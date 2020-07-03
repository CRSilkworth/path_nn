"""The pipeline defintion which incrementally trains the intent classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List, Dict, Optional

from tfx.orchestration import pipeline
from tfx.orchestration import metadata

from tfx.components import ImporterNode
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import BigQueryExampleGen
from tfx.components.base import executor_spec
from mlp.components.always_pusher import AlwaysPusher

from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts

from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.types import artifact
from tfx.types import artifact_utils


def create_pipeline(
  pipeline_name: Text,
  pipeline_root: Text,
  pipeline_mod: Text,
  examples_uri: Text,
  schema_uri: Text,
  transform_graph_uri: Text,
  model_uri: Text,
  num_train_steps: int,
  num_eval_steps: int,
  beam_pipeline_args: Optional[List[Text]] = None,
  ai_platform_training_args: Optional[Dict[Text, Text]] = None,
  metadata_path: Optional[Text] = None
) -> pipeline.Pipeline:
  """Implements the incremental pipeline.."""

  schema_importer = ImporterNode(
    instance_name='schema_importer',
    source_uri=schema_uri,
    artifact_type=standard_artifacts.Schema,
    reimport=False
  )
  transform_graph_importer = ImporterNode(
    instance_name='transform_graph_importer',
    source_uri=transform_graph_uri,
    artifact_type=standard_artifacts.TransformGraph,
    reimport=False
  )
  examples_importer = ImporterNode(
    instance_name='examples_importer',
    source_uri=examples_uri,
    artifact_type=standard_artifacts.Examples,
    properties={
      'split_names':
      artifact_utils.encode_split_names(artifact.DEFAULT_EXAMPLE_SPLITS)},
    reimport=False
  )

  trainer_kwargs = {}
  if ai_platform_training_args is not None:
    trainer_kwargs = {
      'custom_executor_spec': executor_spec.ExecutorClassSpec(
          ai_platform_trainer_executor.Executor
        ),
      'custom_config': {
        ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args
      }
    }

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
    transformed_examples=examples_importer.outputs['result'],
    schema=schema_importer.outputs['result'],
    transform_graph=transform_graph_importer.outputs['result'],
    train_args=trainer_pb2.TrainArgs(num_steps=num_train_steps),
    eval_args=trainer_pb2.EvalArgs(num_steps=num_eval_steps),
    trainer_fn='{}.trainer_fn'.format(pipeline_mod),
    **trainer_kwargs
  )

  # Not depdent on blessing. Always pushes regardless of quality.
  pusher = AlwaysPusher(
    model=trainer.outputs['model'],
    push_destination=pusher_pb2.PushDestination(
      filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=model_uri
      )
    )
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
      schema_importer,
      transform_graph_importer,
      examples_importer,
      trainer,
      pusher
    ],
    enable_cache=True,
    beam_pipeline_args=beam_pipeline_args,
    **pipeline_kwargs

  )
