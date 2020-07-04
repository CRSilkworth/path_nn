from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.trainer import executor as trainer_executor
from official.nlp import bert_modeling as modeling
from mlp.intent_classifier.common import train


class ModelTest(tf.test.TestCase):

  def testTrainerFactory(self):
    bert_dir = 'fake/path'
    vocab_file_path = os.path.join(bert_dir, 'vocab.txt')
    bert_config = modeling.BertConfig.from_dict({
      "attention_probs_dropout_prob": 0.1,
      "directionality": "bidi",
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "max_position_embeddings": 512,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "pooler_fc_size": 768,
      "pooler_num_attention_heads": 12,
      "pooler_num_fc_layers": 3,
      "pooler_size_per_head": 128,
      "pooler_type": "first_token_transform",
      "type_vocab_size": 2,
      "vocab_size": 119547
    })
    bert_checkpoint_dir = os.path.join(bert_dir, 'bert_checkpoint')
    trainer_fn = train.trainer_factory(
      batch_size=16,
      vocab_file_path=vocab_file_path,
      bert_config_path=bert_config,
      bert_checkpoint_dir=bert_checkpoint_dir,
      max_seq_length=8,
      learning_rate=2e-5,
      hidden_layer_dims=[2, 3],
      categorical_feature_keys=['cat_1', 'cat_2'],
      label_key='label',
      warmup_prop=0.1,
      cooldown_prop=0.0,
      non_string_keys=['cat_1'],
      warm_start_from=None,
      save_summary_steps=1000,
      save_checkpoints_secs=100,
      _test_mode=True
    )
    trainer_fn_args = trainer_executor.TrainerFnArgs(
        train_files='/path/to/train.file',
        transform_output='/path/to/transform_output',
        serving_model_dir='/path/to/model_dir',
        eval_files='/path/to/eval.file',
        schema_file='/path/to/schema_file',
        train_steps=1000,
        eval_steps=100,
    )
    schema = schema_pb2.Schema()
    result = trainer_fn(trainer_fn_args, schema)   # pylint: disable=protected-access
    self.assertIsInstance(result['estimator'], tf.estimator.Estimator)
    self.assertIsInstance(result['train_spec'], tf.estimator.TrainSpec)
    self.assertIsInstance(result['eval_spec'], tf.estimator.EvalSpec)
    self.assertTrue(callable(result['eval_input_receiver_fn']))


if __name__ == '__main__':
  tf.test.main()
