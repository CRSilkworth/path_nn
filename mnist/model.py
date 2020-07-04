"""Model definition for the SimplePathNN model."""
import tensorflow as tf
from typing import Optional, List


class SimplePathNN(tf.keras.Model):
  """Basic path neural network model."""

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
    input_dim: dimension of the input before it's fed to the path nn.
    width_dim: dimension of the matrix transformations at each layer in path nn.
    depth_dim: number of layers in the path nn.
    output_dim: dimension of the output to be converted to logits, fed the MSE, etc.
    """
    super(SimplePathNN, self).__init__()
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
    x: the raw input into the path nn.
      shape = [batch_size, input_dim]
    traininig: Whether or not the model is training.

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
