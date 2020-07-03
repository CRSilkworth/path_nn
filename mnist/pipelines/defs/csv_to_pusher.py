"""The pipeline defintion which trains the intent classifier from scratch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List, Dict, Optional

from tfx.orchestration import pipeline
from tfx.orchestration import metadata

from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import CsvExampleGen
from mlp.components.always_pusher import AlwaysPusher
from mlp.components.artifact_pusher import SchemaPusher
from mlp.components.artifact_pusher import TransformGraphPusher

from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.utils.dsl_utils import external_input


def create_pipeline(
  pipeline_name: Text,
  pipeline_root: Text,
  pipeline_mod: Text,
  schema_uri: Text,
  transform_graph_uri: Text,
  model_uri: Text,
  example_csv: Text,
  num_train_steps: int,
  num_eval_steps: int,
  metadata_path: Optional[Text] = None
) -> pipeline.Pipeline:
  """Implements the from scratch pipeline.."""

  external = external_input(example_csv)
  # Pull examples from bigquery.

  example_gen = CsvExampleGen(
    input=external,
    output_config=example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=[
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)]
      )
    )
  )

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    infer_feature_shape=False
  )

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=infer_schema.outputs['schema']
  )

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=infer_schema.outputs['schema'],
      preprocessing_fn='{}.preprocessing_fn'.format(pipeline_mod)
  )
  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
    transformed_examples=transform.outputs['transformed_examples'],
    schema=infer_schema.outputs['schema'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=num_train_steps),
    eval_args=trainer_pb2.EvalArgs(num_steps=num_eval_steps),
    trainer_fn='{}.trainer_fn'.format(pipeline_mod),
  )

  # Not depdent on blessing. Always pushes regardless of quality.
  pusher = AlwaysPusher(
      model=trainer.outputs['model'],
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=model_uri
        )
      ),
  )

  # Pushes schema to a particular directory. Only needed if schema is required
  # for other pipelines/processes.
  schema_pusher = SchemaPusher(
      artifact=infer_schema.outputs['schema'],
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=schema_uri
        )
      ),
      instance_name='schema_pusher'
  )

  transform_graph_pusher = TransformGraphPusher(
      artifact=transform.outputs['transform_graph'],
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=transform_graph_uri
        )
      ),
      instance_name='transform_graph_pusher'
  )

  pipeline_kwargs = {}
  if metadata_path is not None:
    pipeline_kwargs = {
      'metadata_connection_config': metadata.sqlite_metadata_connection_config(
        metadata_path),
    }

  return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    components=[
      example_gen,
      statistics_gen,
      infer_schema,
      validate_stats,
      transform,
      trainer,
      pusher,
      schema_pusher,
      transform_graph_pusher
    ],
    enable_cache=True,
    beam_pipeline_args=['--direct_num_workers=1'],
    **pipeline_kwargs

  )
