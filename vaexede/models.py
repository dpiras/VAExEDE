import tensorflow as tf
from vaexede.custom_activation import CPAct

class CVAE(tf.keras.Model):
  """
  Convolutional variational autoencoder
  with custom activation function (taken from CP).
  """

  def __init__(self, input_dim, latent_dim, concat):
    super(CVAE, self).__init__()
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    # these hyperparameters are hardcoded and are to ensure that we do not have
    # to change the architecture every time we change the input size
    # feel free to play with them
    self.s1 = 3 
    self.output_layer1 = 823 # quite random, only to make it smaller
    self.k1 = self.input_dim - (self.output_layer1-1)*self.s1 # output = (input - k)/s + 1
    assert self.k1 > 0, "The input size is smaller than expected, so the first layer has a negative kernel size. " \
                        f"Please ensure that k1 is a positive integer, e.g. by decreasing output_layer1, found k1={self.k1}"
    
    # encoder
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim, 1)),
            tf.keras.layers.Conv1D(
                filters=16, kernel_size=self.k1, strides=self.s1, activation=None),
            CPAct(trainable=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(
                filters=32, kernel_size=16, strides=3, activation=None),
            CPAct(trainable=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, strides=3, activation=None),
            CPAct(trainable=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    # decoder
    extra_params = 6 if concat else 0
    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim+extra_params,)),
            tf.keras.layers.Dense(units=90*64, activation=None),
            CPAct(trainable=True),
            tf.keras.layers.Reshape(target_shape=(90, 64)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1DTranspose(
                filters=64, kernel_size=3, strides=3, padding='valid',
                activation=None),
            CPAct(trainable=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1DTranspose(
                filters=32, kernel_size=16, strides=3, padding='valid',
                activation=None),
            CPAct(trainable=True),
            tf.keras.layers.BatchNormalization(),
            # No activation
            tf.keras.layers.Conv1DTranspose(
                filters=1, kernel_size=self.k1, strides=self.s1, padding='valid'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z):
    logits = self.decoder(z)
    return logits


