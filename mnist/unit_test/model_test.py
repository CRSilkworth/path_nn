from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from official.nlp import bert_modeling as modeling
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.trainer import executor as trainer_executor
from mlp.intent_classifier.common import model


class ModelTest(tf.test.TestCase):
  def testModel(self):
    bert_dir = 'fake/path'
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
    model_1 = model.FineTunedBERT(
      vocab_file_path=os.path.join(bert_dir, 'vocab.txt'),
      bert_config=bert_config,
      max_seq_length=32,
      hidden_layer_dims=[],
      num_labels=5
    )

    model_2 = model.FineTunedBERT(
      vocab_file_path=os.path.join(bert_dir, 'vocab.txt'),
      bert_config=bert_config,
      max_seq_length=16,
      hidden_layer_dims=[20, 5],
      num_labels=5,
      bert_checkpoint_dir=os.path.join(bert_dir, 'bert_checkpoint'),
      non_string_keys=['feature_1', 'feature_2']
    )


if __name__ == '__main__':
  tf.test.main()
