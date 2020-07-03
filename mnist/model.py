"""Model definition for the BasicNN model."""
import tensorflow as tf
from typing import Optional, List


class PathNN(tf.keras.Model):
  """Basic classifier neural network model."""

  def __init__(
    self,
    input_dim: int,
    width_dim: int,
    depth_dim: int,
    output_dim: int,
    T: Optional[float] = 1.0
    ):
    """Construct an BasicNN model.

    Parameters
    ----------
    num_labels: The total number of allowed labels.
    hidden_layer_dims: The dimensions of the dense layers that combine the
      non_string_key data and output from the bert model.

    """
    super(PathNN, self).__init__()
    self.input_dim = input_dim
    self.width_dim = width_dim
    self.depth_dim = depth_dim
    self.output_dim = output_dim
    self.delta_d = T/depth_dim

    self.in_weights = tf.Variable(tf.random.normal([input_dim, width_dim]))
    self.alpha = tf.Variable(tf.zeros([width_dim, depth_dim]))
    self.beta = tf.Variable(tf.random.normal([width_dim, width_dim, depth_dim]))
    self.ouput_weights = tf.Variable(tf.random.normal([width_dim, output_dim]))
    self.n_d = tf.range(0.0, T, self.delta_d)

  def call(self, x, training=False, **kwargs):
    """Get the logits predicted by the model.

    Parameters
    ----------
    inputs: will be ignored. Only there to conform to keras model standard.
    traininig: Whether or not the model is training.
    kwargs: Any data to be used in the prediction.

    Returns
    -------
    The logits predicted by the model.

    """
    x = tf.einsum('nj,jk->nk', x, self.in_weights)
    x_1_n_1 = self._raise_to_power(x, 1./(self.n_d + 1))
    x_n_n_1 = self._raise_to_power(x, self.n_d/(self.n_d + 1))

    y = x + self.delta_d * tf.reduce_sum(
      x_n_n_1 *
      (self.alpha + tf.einsum('ijd, njd -> nid', self.beta, x_1_n_1)) *
      (self.n_d + 1.),
      axis=-1
    )
    y = tf.einsum('nj,jk->nk', x, self.ouput_weights)

    return y

  def _raise_to_power(self, x, n):
    """Tile x so that it can be broadcasted with n and then raise it to n."""
    x_n = tf.tile(
      tf.expand_dims(x, axis=-1),
      multiples=[1, 1, tf.shape(n)[0]]
    )

    x_n = tf.pow(x_n, n)
    return x_n
