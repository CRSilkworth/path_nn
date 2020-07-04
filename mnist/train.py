"""Trainer function and estimator defintion."""
from __future__ import division
from __future__ import print_function
from typing import Optional, Dict, List, Text, Any, Callable

import tensorflow as tf
import tensorflow_transform as tft

from tensorflow_metadata.proto.v0 import schema_pb2

from mnist import model
from mnist import preprocess as pre
from mnist import metrics
from mlp.tensorflow.learning_rate_schedules import piecewise_learning_rate


def estimator_builder(
  run_config: tf.estimator.RunConfig,
  vocabularies: Dict[Text, List[Text]],
  pixel_key: Text,
  label_key: Text,
  input_dim: int,
  width_dim: int,
  depth_dim: int,
  D: float,
  num_train_steps: int,
  num_warmup_steps: int,
  num_cool_down_steps: int,
  learning_rate: Optional[float] = 2e-5,
  top_ks: Optional[List[int]] = [1, 3, 5, 10],
  warm_start_from: Optional[Text] = None) -> tf.estimator.Estimator:
  """Define the model_fn and build the estimator.

  Parameters
  ----------
    run_config: the training run config for the estimator.
    vocabularies: Mapping from the categorical data key to the list of allowed
      categorical values it can take.
    pixel_key: the key of the input data.
    label_key: the key in of the output data.
    input_dim: dimension of the input before it's fed to the path nn.
    width_dim: dimension of the matrix transformations at each layer in path nn.
    depth_dim: number of layers in the path nn.
    output_dim: dimension of the output to be converted to logits, fed the MSE,
      etc.
    D: the maximum depth value.
    num_train_steps: Total number of steps to train the model
    num_warmup_steps: How fast to warmup to the maximum learning rate.
    num_cool_down_steps: How fast to decrease learning rate to zero.
    learning_rate: Learing rate for the optimizer
    top_ks: The accuracies to include in the evaluation. A top_k accuracy being
      how often the true label is in the top_k predicted labels.
    warm_start_from: path to base model to warm start from.
  Returns
  -------
    estimator: The estimator that will be used for training, eval and prediction

  """
  def model_fn(
    features: Dict[Text, Any],
    labels: tf.Tensor,
    mode: tf.estimator.ModeKeys,
    ) -> tf.estimator.EstimatorSpec:
    """Define model_fn.

    This holds the training, eval and prediction pipeline definitions for the
    estimator.

    Parameters
    ----------
      features: dict of `Tensor`.
      labels: `Tensor` of shape `[batch_size]`.
      mode: Defines whether this is training, evaluation or prediction.
        See `ModeKeys`.

    Returns
    -------
      An `EstimatorSpec` instance.

    Raises
    ------
      ValueError: mode or params are invalid, or features has the wrong type.

    """
    # Pull out all valid labels and their count
    all_labels = vocabularies[pre._transformed_name(label_key)]
    num_labels = len(all_labels)

    # Define the full model to be used in prediction.
    path_nn = model.SimplePathNN(
      input_dim=input_dim,
      width_dim=width_dim,
      depth_dim=depth_dim,
      output_dim=num_labels,
      D=D
    )

    # One hot all the categorical variables
    global_step = tf.compat.v1.train.get_or_create_global_step()
    pixels = features[pre._transformed_name(pixel_key)]
    if mode == tf.estimator.ModeKeys.TRAIN:

      # Get the label indices and their one hots.
      labels = features[pre._transformed_name(label_key)]
      labels_ohs = tf.one_hot(labels, num_labels)

      # Get the loss from the logits
      logits = path_nn(pixels, training=True)
      loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_ohs,
        logits=logits
      )
      loss = tf.reduce_mean(loss)

      # Define the optimizer and minimize the loss
      adj_learning_rate = piecewise_learning_rate(global_step, learning_rate, num_train_steps, num_warmup_steps, num_cool_down_steps)

      optimizer = tf.compat.v1.train.AdamOptimizer(adj_learning_rate)
      train_op = optimizer.minimize(
        loss, tf.compat.v1.train.get_or_create_global_step())

      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
      )
    elif mode == tf.estimator.ModeKeys.EVAL:

      # Get the label indices and their one hots.
      labels = features[pre._transformed_name(label_key)]
      labels_ohs = tf.one_hot(labels, num_labels)

      # Get the logits from the model, calculate the loss and probabilities.
      logits = path_nn(pixels, training=False)
      loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_ohs, logits=logits)
      loss = tf.reduce_mean(loss)
      probs = tf.nn.softmax(logits)

      adj_learning_rate = piecewise_learning_rate(global_step, learning_rate, num_train_steps, num_warmup_steps, num_cool_down_steps)

      # Get the evaluation ops
      eval_metrics = metrics.metric_fn(labels[:, 0], probs, top_ks, adj_learning_rate, path_nn.trainable_variables)

      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metrics
      )
    elif mode == tf.estimator.ModeKeys.PREDICT:
      # Calculate the probabilities for each label from the logits.
      logits = path_nn(pixels, training=False)
      probs = tf.nn.softmax(logits=logits)

      # Get the mapping from label index to string.
      label_lookup = vocabularies[pre._transformed_name(label_key)]

      # Sort and argsort the probabilities
      probs, indices = tf.math.top_k(probs, k=len(label_lookup))

      # Convert the label indices to strings and return the prediction
      predictions = {
        'probabilities': probs,
        'labels': tf.gather(label_lookup, indices)
      }

      return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
      raise ValueError("Invalid mode key: {}".format(mode))

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    warm_start_from=warm_start_from,
  )

  return estimator


def trainer_factory(
  batch_size: int,
  learning_rate: float,
  pixel_key: Text,
  label_key: Text,
  num_pixels: int,
  width_dim: int,
  depth_dim: int,
  D: float,
  warmup_prop: float,
  cooldown_prop: float,
  save_summary_steps: int,
  save_checkpoints_secs: int,
  warm_start_from: Optional[Text] = None,
) -> Callable:
  """
  Define a trainer_fn function to pass to the Trainer component.

  Parameters
  ----------
  batch_size: the mini batch size to train with
  learning_rate: Learing rate for the optimizer
  pixel_key: the key of the input data.
  label_key: the key in of the output data.
  num_pixels: size of the array of inputted num_pixels.
  width_dim: dimension of the matrix transformations at each layer in path nn.
  depth_dim: number of layers in the path nn.
  D: the maximum depth value.
  warmup_prop: the proportion of the training steps that are linearly increasing
    to the maximum rate.
  cooldown_prop: the proportion of the training steps that are linearly decreasing.
    from the maximum rate back to zero.
  warm_start_from: The path of the model to warm start training from.
  save_summary_steps: How often to save a summary and log.
  save_checkpoints_secs: How often to save a checkpoint and evaluate the
    validation set.

  Returns
  -------
  The defined trainer_fn

  """
  def trainer_fn(
    hparams: Any,
    schema: schema_pb2.Schema
    ) -> Dict[Text, Any]:
    """Build the estimator using the high level API.

    Parameters
    ----------
      hparams: Holds hyperparameters used to train the model as name/value pairs.
      schema: Holds the schema of the training examples.

    Returns
    -------
      Dict:
        - estimator: The estimator that will be used for training and eval.
        - train_spec: Spec for training.
        - eval_spec: Spec for eval.
        - eval_input_receiver_fn: Input function for eval.

    """
    # Pull in the transform definition
    tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

    # Build the inputs for the training and eval
    train_input_fn = pre.get_input_fn(
      hparams.train_files,
      tf_transform_output,
      batch_size=batch_size
    )
    eval_input_fn = pre.get_input_fn(
      hparams.eval_files,
      tf_transform_output,
      batch_size=batch_size
    )

    # Define how to receive data during inference and export.
    exporter = tf.estimator.FinalExporter(
      'path_nn',
      lambda: pre._example_serving_receiver_fn(tf_transform_output, schema, label_key)
    )

    # Define the training and eval specifications
    train_spec = tf.estimator.TrainSpec(
      train_input_fn,
      max_steps=hparams.train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=hparams.eval_steps,
      exporters=[exporter],
      name='intent-classifier-eval'
    )

    # How to split up over multiple gpus
    # NOTE: Uncomment for testing on single gpu.
    # strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
    strategy = tf.distribute.MirroredStrategy()

    # Set the other training specs
    run_config = tf.estimator.RunConfig(
      model_dir=hparams.serving_model_dir,
      save_summary_steps=save_summary_steps,
      log_step_count_steps=save_summary_steps,
      save_checkpoints_secs=save_checkpoints_secs,
      keep_checkpoint_max=1,
      train_distribute=strategy,
      eval_distribute=strategy,
    )

    vocabularies = _get_vocabularies(tf_transform_output, [label_key])

    # Define the estimator
    estimator = estimator_builder(
      run_config=run_config,
      learning_rate=learning_rate,
      input_dim=num_pixels,
      width_dim=width_dim,
      depth_dim=depth_dim,
      D=D,
      vocabularies=vocabularies,
      num_train_steps=hparams.train_steps,
      num_warmup_steps=int(hparams.train_steps * warmup_prop),
      num_cool_down_steps=int(hparams.train_steps * cooldown_prop),
      pixel_key=pixel_key,
      label_key=label_key,
      warm_start_from=warm_start_from
    )

    return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': lambda: pre._eval_input_receiver_fn(
        tf_transform_output, schema, label_key)
    }
  return trainer_fn


def _get_vocabularies(tf_transform_output, keys):
  """Get the allowed values for each categorical feature key."""
  vocabularies = {}
  for key in keys:
    trans_key = pre._transformed_name(key)
    vocabularies[trans_key] = tf_transform_output.vocabulary_by_name(key)

  return vocabularies
