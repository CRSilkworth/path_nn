"""The pipeline defintion which incrementally trains the intent classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List, Dict, Optional

from tfx.orchestration import pipeline
from tfx.orchestration import metadata

from tfx.components import ImporterNode
from mlp.components.transform_with_graph import TransformWithGraph
from tfx.components import Trainer
from tfx.components import BigQueryExampleGen
from tfx.components.base import executor_spec
from mlp.components.always_pusher import AlwaysPusher

from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts

from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor


def create_pipeline(
  pipeline_name: Text,
  pipeline_root: Text,
  pipeline_mod: Text,
  schema_uri: Text,
  transform_graph_uri: Text,
  model_uri: Text,
  query: Text,
  num_train_steps: int,
  num_eval_steps: int,
  beam_pipeline_args: Optional[List[Text]] = None,
  ai_platform_training_args: Optional[Dict[Text, Text]] = None,
  metadata_path: Optional[Text] = None
) -> pipeline.Pipeline:
  """Implements the incremental pipeline.."""

  example_gen = BigQueryExampleGen(
    query=query,
    output_config=example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=[
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=20),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)]
      )
    )
  )

  importer = ImporterNode(
    instance_name='import_schema',
    source_uri=schema_uri,
    artifact_type=standard_artifacts.Schema,
    reimport=False
  )
  graph_importer = ImporterNode(
    instance_name='import_transform_graph',
    source_uri=transform_graph_uri,
    artifact_type=standard_artifacts.TransformGraph,
    reimport=False
  )
  model_importer = ImporterNode(
    instance_name='import_model',
    source_uri=model_uri,
    artifact_type=standard_artifacts.Model,
    reimport=False
  )

  # Performs transformations and feature engineering in training and serving.
  transform = TransformWithGraph(
      examples=example_gen.outputs['examples'],
      schema=importer.outputs['result'],
      transform_graph=graph_importer.outputs['result']
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
    transformed_examples=transform.outputs['transformed_examples'],
    schema=importer.outputs['result'],
    transform_graph=graph_importer.outputs['result'],
    train_args=trainer_pb2.TrainArgs(num_steps=num_train_steps),
    eval_args=trainer_pb2.EvalArgs(num_steps=num_eval_steps),
    trainer_fn='{}.trainer_fn'.format(pipeline_mod),
    base_model=model_importer.outputs['result'],
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
      example_gen,
      importer,
      graph_importer,
      model_importer,
      transform,
      trainer,
      pusher
    ],
    enable_cache=True,
    beam_pipeline_args=beam_pipeline_args,
    **pipeline_kwargs

  )
