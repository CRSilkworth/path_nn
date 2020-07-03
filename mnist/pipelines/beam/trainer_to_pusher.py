"""Run a beam pipeline that starts from pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from mnist.pipelines.defs.trainer_to_pusher import create_pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from mnist import train
import mnist.pipelines.beam.bigquery_to_pusher as full

from mlp.utils.dir import pipeline_dirs

_PIPELINE_TYPE = 'trainer_to_pusher'

_GCP_PROJECT = 'tripla-data'

_RUN_STR = None

_NUM_TRAIN_STEPS = 750
_NUM_EVAL_STEPS = 5
_WARMUP_PROP = 0.1
_COOLDOWN_PROP = 0.1
_SAVE_SUMMARY_STEPS = 1
_SAVE_CHECKPOINT_SECS = 14400
_LEARNING_RATE = 2e-5
_WARM_START_FROM = None

pipeline_name = '-'.join([
  full._MLP_PROJECT,
  full._MLP_SUBPROJECT,
  _PIPELINE_TYPE
])
pipeline_mod = '.'.join([
  full._MLP_PROJECT,
  full._MLP_SUBPROJECT,
  'pipelines',
  full._RUNNER,
  _PIPELINE_TYPE
])
proj_root = os.path.join(full._RUN_DIR, 'tfx', pipeline_name)
serving_root = os.path.join(full._RUN_DIR, 'serving', full._MLP_PROJECT, full._MLP_SUBPROJECT)
schema_uri = os.path.join(serving_root, 'schema')
examples_uri = os.path.join(serving_root, 'transformed_examples')

pipeline_root, _, __, ___ = pipeline_dirs(
  full._RUN_DIR,
  _RUN_STR,
  full._MLP_PROJECT,
  full._MLP_SUBPROJECT,
  pipeline_name
)

trainer_fn = train.trainer_factory(
  batch_size=full._BATCH_SIZE,
  learning_rate=_LEARNING_RATE,
  hidden_layer_dims=full._HIDDEN_LAYER_DIMS,
  categorical_feature_keys=full._CATEGORICAL_FEATURE_KEYS,
  numerical_feature_keys=full._NUMERICAL_FEATURE_KEYS,
  label_key=full._LABEL_KEY,
  # warm_start_from=_WARM_START_FROM,
  warmup_prop=_WARMUP_PROP,
  cooldown_prop=_COOLDOWN_PROP,
  save_summary_steps=_SAVE_SUMMARY_STEPS,
  save_checkpoints_secs=_SAVE_CHECKPOINT_SECS
)

beam_pipeline_args = [
  '--project=' + _GCP_PROJECT
]

if __name__ == "__main__":
  DAG = BeamDagRunner().run(
    create_pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      pipeline_mod=pipeline_mod,
      schema_uri=full.schema_uri,
      transform_graph_uri=full.transform_graph_uri,
      examples_uri=examples_uri,
      model_uri=full.model_uri,
      num_train_steps=_NUM_TRAIN_STEPS,
      num_eval_steps=_NUM_EVAL_STEPS,
      beam_pipeline_args=beam_pipeline_args,
      metadata_path=os.path.join(pipeline_root, 'metadata', 'metadata.db')
      )
  )
