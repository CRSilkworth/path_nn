"""Run a beam pipeline that starts from pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from mnist.pipelines.defs.csv_to_pusher import create_pipeline

from mnist import preprocess
from mnist import train
from mlp.utils.dirs import pipeline_dirs

_MLP_PROJECT = 'path_nn'
_MLP_SUBPROJECT = 'mnist'

_PIPELINE_TYPE = 'csv_to_pusher'
_RUNNER = 'beam'

_EXAMPLE_CSV = os.path.join(os.environ['HOME'], 'data/mnist/train')

# Set to timestamp of previous run if you want to continue old run.
_RUN_STR = None
_RUN_DIR = os.path.join('runs')

# Define the preprocessing/feature parameters
_PIXEL_KEY = 'pixels'
_LABEL_KEY = 'label'
_NUM_PIXELS = 784
_WIDTH_DIM = 256
_DEPTH_DIM = 256
_T = 2.0

# Define the training/model parameters
_BATCH_SIZE = 2048
_NUM_TRAIN_STEPS = 500000
_NUM_EVAL_STEPS = 100
_WARMUP_PROP = 0.05
_COOLDOWN_PROP = 0.2
_WARM_START_FROM = None
_SAVE_SUMMARY_STEPS = 100
_SAVE_CHECKPOINT_SECS = 150
_LEARNING_RATE = 1e-2

pipeline_name = '-'.join([
  _MLP_PROJECT,
  _MLP_SUBPROJECT,
  _PIPELINE_TYPE
])
pipeline_mod = '.'.join([
  _MLP_SUBPROJECT,
  'pipelines',
  _RUNNER,
  _PIPELINE_TYPE
])
pipeline_root, model_uri, schema_uri, transform_graph_uri = pipeline_dirs(
  _RUN_DIR,
  _RUN_STR,
  _MLP_PROJECT,
  _MLP_SUBPROJECT,
  pipeline_name
)


trainer_fn = train.trainer_factory(
  batch_size=_BATCH_SIZE,
  learning_rate=_LEARNING_RATE,
  pixel_key=_PIXEL_KEY,
  label_key=_LABEL_KEY,
  num_pixels=_NUM_PIXELS,
  width_dim=_WIDTH_DIM,
  depth_dim=_DEPTH_DIM,
  T=_T,
  warmup_prop=_WARMUP_PROP,
  cooldown_prop=_COOLDOWN_PROP,
  save_summary_steps=_SAVE_SUMMARY_STEPS,
  save_checkpoints_secs=_SAVE_CHECKPOINT_SECS
)

preprocessing_fn = preprocess.preprocess_factory(
  pixel_key=_PIXEL_KEY,
  label_key=_LABEL_KEY,
  num_pixels=_NUM_PIXELS
)

if __name__ == "__main__":
  DAG = BeamDagRunner().run(
    create_pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      pipeline_mod=pipeline_mod,
      schema_uri=schema_uri,
      transform_graph_uri=transform_graph_uri,
      model_uri=model_uri,
      example_csv=_EXAMPLE_CSV,
      num_train_steps=_NUM_TRAIN_STEPS,
      num_eval_steps=_NUM_EVAL_STEPS,
      metadata_path=os.path.join(pipeline_root, 'metadata', 'metadata.db')
      )
  )
