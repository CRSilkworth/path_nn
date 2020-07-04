"""Metrics definition to be fed to estimator for evaluation."""
import tensorflow as tf


def metric_fn(label_ids, probs, top_ks, learning_rate, weights):
  """Define the standard evaluation metrics for the classifier."""
  metrics = {}
  metrics['learning_rate'] = tf.keras.metrics.Mean()
  metrics['learning_rate'].update_state(learning_rate)

  for top_k in top_ks:
    # Find whether or not the label is in the top_k predicted labels
    accuracy = tf.math.in_top_k(
      predictions=probs,
      targets=label_ids,
      k=top_k)

    # Take the mean accuracy over the dataset
    running_mean = tf.keras.metrics.Mean()
    running_mean.update_state(accuracy)

    metrics['top_' + str(top_k) + '_accuracy'] = running_mean

  weights_norm = tf.constant(0.)
  weights_min = tf.constant(0.)
  weights_max = tf.constant(0.)
  weights_std = tf.constant(0.)
  for w in weights:
    weights_norm += tf.reduce_sum(w * w)
    weights_std += tf.math.reduce_std(w)*tf.math.reduce_std(w)
    weights_max = tf.maximum(tf.reduce_max(w), weights_max)
    weights_min = tf.maximum(tf.reduce_min(w), weights_min)

  weights_norm = tf.sqrt(weights_norm)
  weights_std = tf.sqrt(weights_std)

  metrics['weights_norm_mean'] = tf.keras.metrics.Mean()
  metrics['weights_norm_mean'].update_state(weights_norm)

  metrics['weights_max'] = tf.keras.metrics.Mean()
  metrics['weights_max'].update_state(weights_max)

  metrics['weights_min'] = tf.keras.metrics.Mean()
  metrics['weights_min'].update_state(weights_min)

  metrics['weights_std'] = tf.keras.metrics.Mean()
  metrics['weights_std'].update_state(weights_std)

  return metrics
