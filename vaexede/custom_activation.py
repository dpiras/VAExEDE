import tensorflow as tf

from typing import Optional
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam


class CPAct(Layer):
  """``CPAct`` activation function."""
  def __init__(self,
               gamma: tf.Tensor = 0.,
               beta: tf.Tensor = 0.,
               trainable: bool = False,
               **kwargs):
    super().__init__(**kwargs)
    self.gamma = gamma
    self.beta = beta
    self.trainable = trainable

  def build(self, input_shape: tf.TensorShape):
    super().build(input_shape)  
    self.gamma_factor = tf.Variable(
      self.gamma,
      dtype=tf.float32,
      trainable=self.trainable,
      name="gamma_factor")
    self.beta_factor = tf.Variable(
      self.beta,
      dtype=tf.float32,
      name="beta_factor")

  def call(self,
           inputs: tf.Tensor,
           mask: Optional[tf.Tensor] = None
           ) -> tf.Tensor:
    neuron = tf.math.sigmoid(inputs*self.beta_factor) * (1 - self.gamma_factor)
    neuron += self.gamma_factor
    neuron *= inputs
    return neuron
  
  def get_config(self):
    config = {
        "gamma": self.get_weights()[0] if self.trainable else self.gamma,
        "beta": self.get_weights()[1] if self.trainable else self.beta,
        "trainable": self.trainable
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
