# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    utils,
    metrics,
    losses,
    optimizers,
)

from scipy.stats import norm
import pandas as pd

from notebooks.utils import sample_batch, display
from vae_utils import get_vector_from_label, add_vector_to_images, morph_faces

# PARAMETERS
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 128
NUM_FEATURES = 64
DIMMENSIONS = 200 # how many dimmensions it will use
LEARNING_RATE = 0.0005
EPOCHS = 10
BETA = 2000
LOAD_MODEL = False

# GLOBAL VARIABLES
old_shape = -1

# SAMPLING LAYER
class Sampling(layers.Layer): # sampling layer will inherit Keras base layer class attributes and methods
	def call(self, inputs):
		z_mean, z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = td.shape(z_mean)[1]
		epsilon = K.random_normal(shape = (batch, dim))
		return z_mean + tf.exp(0.5 * z_log_var) * epsilon # reparameterization trick


# VAE
class VAE(models.Model):
	def __init__(self, encoder, decoder, **kwargs):
		super(VAE, self).__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.total_loss_tracker = metrics.Mean(name = 'total_loss')
		self.reconstruction_loss_tracker = metrics.Mean(name = 'reconstruction_loss')
		self.kl_loss_tracker = metrics.Mean(name = 'kl_loss')

	@property
	def metrics(self):
		return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker,]
	
	def call(self, inputs):
		z_mean, z_log_var, z = encoder(inputs)
		reconstruction = decoder(z)
		return z_mean, z_log_var, reconstruction

	def train_step(self, data): # one training step of the VAE, including calculation of the loss function
		with tf.GradientTape() as tape:
			z_mean, z_log_var, reconstruction = self(data)
			# beta value of 500 is used in the reconstruction loss
			reconstruction_loss = tf.reduce_mean(500 * losses.binary_crossentropy(data, reconstruction, axis(1, 2, 3)))
			kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis = 1,))
			total_loss = reconstruction_loss + kl_loss # total loss is sum of both

		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.total_loss_tracker.update_state(total_loss)
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		self.kl_loss_tracker.update_state(kl_loss)

		return {m.name: m.result() for m in self.metrics}
	
    def test_step(self, data): # one validation step of the VAE.
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(BETA * losses.mean_squared_error(data, reconstruction))
        kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1,))
        total_loss = reconstruction_loss + kl_loss
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss,}


# VAE ENCODER
def var_encoder():
    encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
	
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
	
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
	
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
	
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
	
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
	
    global old_shape
    old_shape = K.int_shape(x)[1:]  # the decoder will need this!

    x = layers.Flatten()(x)
    z_mean = layers.Dense(DIMMENSIONS, name="z_mean")(x)
    z_log_var = layers.Dense(DIMMENSIONS, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
	# encoder.summary
	return encoder

def var_decoder():
	decoder_input = layers.Input(shape=(DIMMENSIONS,), name="decoder_input")
	
    x = layers.Dense(np.prod(old_shape))(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape(old_shape)(x)

    x = layers.Conv2DTranspose(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    decoder_output = layers.Conv2DTranspose(CHANNELS, kernel_size=3, strides=1, activation="sigmoid", padding="same")(x)
    decoder = models.Model(decoder_input, decoder_output)
    # decoder.summary
    return decoder

