"""Run a beam pipeline that starts from pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from mnist.pipelines.defs.incremental import create_pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from mnist import preprocess
from mnist import train
import mnist.pipelines.beam.bigquery_to_pusher as full
from mlp.utils.dirs import pipeline_dirs

_PIPELINE_TYPE = 'incremental'

_GCP_PROJECT = 'tripla-data'

_RUN_STR = None

_FREQ_NUM = -1
_FREQ = "DAY"
_NUM_OLD = 5000
_QUERY = """
  (
  SELECT
    example_id,
    language,
    country,
    hotel_id,
    business_type,
    intent_code,
    request
  FROM ml_example.translated_request_response
  WHERE request IS NOT NULL
    AND intent_code IS NOT NULL
    AND DATETIME(created_at) > DATETIME_ADD(CURRENT_DATETIME(), INTERVAL {} {})
)
UNION DISTINCT
(
  SELECT
    example_id,
    language,
    country,
    hotel_id,
    business_type,
    intent_code,
    request
  FROM ml_example.translated_request_response
  WHERE request IS NOT NULL
    AND intent_code IS NOT NULL
  LIMIT {}
 )
""".format(_FREQ_NUM, _FREQ, _NUM_OLD)

_NUM_TRAIN_STEPS = 750
_NUM_EVAL_STEPS = 5
_WARMUP_PROP = 0.1
_COOLDOWN_PROP = 0.1
_SAVE_SUMMARY_STEPS = 1
_SAVE_CHECKPOINT_SECS = 14400
_LEARNING_RATE = 2e-5

# _WARM_START_FROM = os.path.join(full._RUN_DIR, 'tfx', full.pipeline_name, '*/data/Trainer/model/*/serving_model_dir/checkpoint')
# _ASSET_DIR = os.path.join(full._RUN_DIR, 'tfx', full.pipeline_name, '*/data/Transform/transform_graph/*/transform_fn/assets/')

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
  warmup_prop=_WARMUP_PROP,
  cooldown_prop=_COOLDOWN_PROP,
  # warm_start_from=full.model_uri,
  save_summary_steps=_SAVE_SUMMARY_STEPS,
  save_checkpoints_secs=_SAVE_CHECKPOINT_SECS
)

preprocessing_fn = preprocess.preprocess_factory(
  categorical_feature_keys=full._CATEGORICAL_FEATURE_KEYS,
  numerical_feature_keys=full._NUMERICAL_FEATURE_KEYS,
  label_key=full._LABEL_KEY,
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
      model_uri=full.model_uri,
      query=_QUERY,
      num_train_steps=_NUM_TRAIN_STEPS,
      num_eval_steps=_NUM_EVAL_STEPS,
      beam_pipeline_args=beam_pipeline_args,
      metadata_path=os.path.join(pipeline_root, 'metadata', 'metadata.db')
      )
  )
