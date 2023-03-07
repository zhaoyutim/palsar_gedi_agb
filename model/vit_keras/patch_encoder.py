import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from model.vit_keras.pos_embed import get_2d_sincos_pos_embed
class PatchEncoder(layers.Layer):
    def __init__(
        self,
        patch_size,
        num_patches,
        projection_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.projection_cnn = layers.Conv2D(filters=self.projection_dim, kernel_size=patch_size, strides=patch_size)
        self.cls_token = tf.Variable(tf.random.normal([1, 1, self.projection_dim], stddev=0.02), trainable=True)
        pos_embed = get_2d_sincos_pos_embed(projection_dim, int(num_patches ** .5), cls_token=True)
        self.position_embedding = tf.convert_to_tensor(pos_embed, dtype=tf.float32)


    def build(self, input_shape):
        self.num_patches = (input_shape[1]//self.patch_size) ** 2

    def call(self, images):
        batch_size = tf.shape(images)[0]
        pos_embeddings = tf.tile(
            self.position_embedding[tf.newaxis], [batch_size, 1, 1]
        )
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        patch_embeddings = self.projection_cnn(images)
        patch_embeddings = tf.reshape(patch_embeddings, [-1, self.num_patches, self.projection_dim])
        patch_embeddings = tf.concat([cls_tokens, patch_embeddings], 1)
        patch_embeddings = (patch_embeddings + pos_embeddings)  # (B, num_patches, projection_dim)
        return patch_embeddings

if __name__=='__main__':
    patch_encoder = PatchEncoder(patch_size=3, num_patches=25, projection_dim=256)
    array = np.zeros((2,15,15,22))
    patch_embeddings = patch_encoder(array)
