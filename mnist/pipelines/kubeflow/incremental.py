"""Create a kubeflow pipeline that starts pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from mnist.pipelines.defs.bigquery_to_pusher import create_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from mnist import preprocess
from mnist import train
import mnist.pipelines.kubflow.bigquery_to_pusher as full
from mlp.utils.dir import pipeline_dirs

_PIPELINE_TYPE = 'incremental'
_RUN_STR = None

_FREQ_NUM = -3
_FREQ = "MONTH"
_NUM_OLD = 5000
_QUERY = """
(
  SELECT
    item_description,
    MAX(vendor_name) AS vendor,
    MAX(bottle_volume_ml) AS max_bottle_volume,
    MAX(category_name) AS category
  FROM `bigquery-public-data.iowa_liquor_sales.sales`
  WHERE date > DATE_ADD(CURRENT_DATE(), INTERVAL {} {})
  GROUP BY item_description
)
UNION DISTINCT
(
  SELECT
    item_description,
    MAX(vendor_name) AS vendor,
    MAX(bottle_volume_ml) AS max_bottle_volume,
    MAX(category_name) AS category
  FROM `bigquery-public-data.iowa_liquor_sales.sales`
  GROUP BY item_description
  LIMIT {}
 )
""".format(_FREQ_NUM, _FREQ, _NUM_OLD)

# Define the training/model parameters
_NUM_TRAIN_STEPS = 100
_NUM_EVAL_STEPS = 10
_WARMUP_PROP = 0.1
_COOLDOWN_PROP = 0.1
_SAVE_SUMMARY_STEPS = 10
_SAVE_CHECKPOINT_SECS = 3600
_LEARNING_RATE = 2e-5

# This will take the latest assets/checkpoints from the full/other
# incremental runs of the pipelines.
any_pipeline = '-'.join([
  full._MLP_PROJECT,
  full._MLP_SUBPROJECT,
  '*'
])
# _WARM_START_FROM = os.path.join(full._RUN_DIR, 'tfx', any_pipeline, '*/data/Trainer/model/*/serving_model_dir/checkpoint')
# _ASSET_DIR = os.path.join(full._RUN_DIR, 'tfx', any_pipeline, '*/data/Transform/transform_graph/*/transform_fn/assets/')

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

beam_pipeline_args = full.beam_pipeline_args
ai_platform_training_args = full.ai_platform_training_args

if __name__ == "__main__":
  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=kubeflow_dag_runner.get_default_kubeflow_metadata_config(),
    tfx_image=os.environ.get('KUBEFLOW_TFX_IMAGE', None)
  )

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
    create_pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      pipeline_mod=pipeline_mod,
      query=_QUERY,
      schema_uri=full.schema_uri,
      transform_graph_uri=full.transform_graph_uri,
      model_uri=full.model_uri,
      num_train_steps=_NUM_TRAIN_STEPS,
      num_eval_steps=_NUM_EVAL_STEPS,
      beam_pipeline_args=beam_pipeline_args,
      ai_platform_training_args=ai_platform_training_args
      )
  )
