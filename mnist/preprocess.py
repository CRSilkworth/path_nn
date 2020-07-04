"""Definition of the TF Transform and input functions."""

from __future__ import division
from __future__ import print_function
from typing import Optional, Dict, List, Text, Any, Callable
from types import FunctionType

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils


def _transformed_name(key: Text) -> Text:
  """Transform a feature key to it's transform's feature key."""
  return key + '_xf'


def _transformed_names(keys: List[Text]) -> List[Text]:
  """Transform a list of feature keys to their transforms' feature keys."""
  return [_transformed_name(key) for key in keys]


def _get_raw_feature_spec(schema: schema_pb2.Schema):
  """Get the feature spec from the schema."""
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames: List[Text]) -> tf.data.TFRecordDataset:
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')


def _fill_in_missing(x: tf.Tensor) -> tf.Tensor:
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

  Parameters
  ----------
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.

  Returns
  -------
    A rank 1 tensor where missing values of `x` have been filled in.

  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value)


def preprocess_factory(
  pixel_key: Text,
  label_key: Text,
  num_pixels: int
) -> Callable:
  """
  Define the preprocessing_fn to feed to the Transform component.

  Parameters
  ----------
  pixel_key: the key of the input data.
  label_key: the key in of the output data.
  num_pixels: size of the array of inputted num_pixels.

  Returns
  -------
  The defined preprocessing_fn

  """
  def preprocessing_fn(inputs: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """tf.transform's callback function for preprocessing inputs.

    Parameters
    ----------
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns
    -------
      Map from string feature key to transformed feature operations.

    """
    outputs = {}
    outputs[_transformed_name(label_key)] = _fill_in_missing(inputs[label_key])

    vocab_file_tensor = tft.vocabulary(
      outputs[_transformed_name(label_key)],
      vocab_filename=label_key
    )

    outputs[_transformed_name(label_key)] = tft.apply_vocabulary(
      outputs[_transformed_name(label_key)],
      vocab_file_tensor
    )

    outputs[_transformed_name(pixel_key)] = tf.concat(
      [_fill_in_missing(inputs[str(i + 1)]) for i in range(num_pixels)],
      axis=1
    )

    # NOTE: This won't be correct in the incremental case since it's only using
    # the new examples to get the mean and variance.
    outputs[_transformed_name('pixels')] = tft.scale_to_0_1(
      outputs[_transformed_name('pixels')]
    )

    return outputs
  return preprocessing_fn


def _example_serving_receiver_fn(
  tf_transform_output: tft.TFTransformOutput,
  schema: schema_pb2.Schema,
  label_key: Text
  ) -> tf.estimator.export.ServingInputReceiver:
  """Build the serving in inputs.

  Parameters
  ----------
    tf_transform_output: A TFTransformOutput.
    schema: the schema of the input data.

  Returns
  -------
    Tensorflow graph which parses examples, applying tf-transform to them.

  """
  # Pull out the feature spec and throwout the label
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(label_key)

  # Define the raw inputs from taken from the user
  receiver_tensors = {}
  for key in raw_feature_spec:
    absl.logging.info("KEY {}".format(key))
    dtype = raw_feature_spec[key].dtype
    receiver_tensors[key] = tf.compat.v1.placeholder(
      dtype=dtype, shape=[None], name='input_' + key
    )

  # Define the inputs into the the graph
  features = {}
  for key in receiver_tensors:
    batch_size = tf.shape(receiver_tensors[key])[0]
    indices = tf.cast(
      tf.expand_dims(tf.range(batch_size), -1),
      tf.int64
    )
    zeros = tf.zeros_like(indices)
    indices = tf.concat([indices, zeros], axis=1)
    features[key] = tf.SparseTensor(
      indices=indices, values=receiver_tensors[key], dense_shape=[batch_size, 1])

  # Transform the features.
  transformed_features = tf_transform_output.transform_raw_features(
      features)
  return tf.estimator.export.ServingInputReceiver(
      transformed_features, receiver_tensors)


def _eval_input_receiver_fn(
  tf_transform_output: tft.TFTransformOutput,
  schema: schema_pb2.Schema,
  label_key: Text
  ) -> tfma.export.EvalInputReceiver:
  """Build everything needed for the tf-model-analysis to run the model.

  Parameters
  ----------
    tf_transform_output: A TFTransformOutput.
    schema: the schema of the input data.

  Returns
  -------
    EvalInputReceiver function, which contains:
      - Tensorflow graph which parses raw untransformed features, applies the
        tf-transform preprocessing operators.
      - Set of raw, untransformed features.
      - Label against which predictions will be compared.

  """
  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = _get_raw_feature_spec(schema)

  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')

  # Add a parse_example operator to the tensorflow graph, which will parse
  # raw, untransformed, tf examples.
  features = tf.io.parse_example(
      serialized=serialized_tf_example, features=raw_feature_spec)

  # Now that we have our raw examples, process them through the tf-transform
  # function computed during the preprocessing step.
  transformed_features = tf_transform_output.transform_raw_features(
      features)

  # The key name MUST be 'examples'.
  receiver_tensors = {'examples': serialized_tf_example}

  # NOTE: Model is driven by transformed features (since training works on the
  # materialized output of TFT, but slicing will happen on raw features.
  features.update(transformed_features)

  return tfma.export.EvalInputReceiver(
      features=features,
      receiver_tensors=receiver_tensors,
      labels=transformed_features[_transformed_name(label_key)])


def get_input_fn(
  filenames: List[Text],
  tf_transform_output: tft.TFTransformOutput,
  batch_size: Optional[int] = 200
  ) -> FunctionType:
  """Build the input funciton from the file_names and tf_transform.

  Parameters
  ----------
    filenames: List of CSV files to read data from.
    tf_transform_output: A TFTransformOutput.
    batch_size: First dimension size of the Tensors returned by input_fn

  Returns
  -------
    input_fn:

  """
  def input_fn(
    mode: tf.estimator.ModeKeys,
    input_context: Optional[Any] = None
    ) -> tf.data.TFRecordDataset:
    """Generate features and labels for training or evaluation.

    Parameters
    ----------
      filenames: [str] list of CSV files to read data from.
      tf_transform_output: A TFTransformOutput.
      batch_size: int First dimension size of the Tensors returned by input_fn

    Returns
    -------
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.

    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

    if input_context:
      dataset = dataset.shard(
        input_context.num_input_pipelines,
        input_context.input_pipeline_id
      )

    return dataset
  return input_fn
