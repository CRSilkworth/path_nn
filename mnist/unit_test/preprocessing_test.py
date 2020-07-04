from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mlp.intent_classifier.common import preprocess


class PreprocessingTest(tf.test.TestCase):

  def testPreprocessFactory(self):
    preprocessing_fn = preprocess.preprocess_factory(
      vocab_feature_keys=['vocab_1'],
      categorical_feature_keys=['cat_1', 'cat_2'],
      label_key='label',
    )
    self.assertTrue(callable(preprocessing_fn))

if __name__ == '__main__':
  tf.test.main()
