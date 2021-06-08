import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from glob import glob
import cv2
import numpy as np
from src.data.make_dataset import IMG_WIDTH, IMG_HEIGHT, n_classes
from src.visualization.visualize import predict_and_save

EPOCHS = 15
SEED = 0
BATCH_SIZE = 8
BUFFER_SIZE = 100
model_name = "mobilenet_unet"

checkpoints_path = os.path.join("models", model_name, str(int(time.time())))
logs_path = os.path.join(checkpoints_path, "logs")
report_fugure_path = os.path.join("reports", "figures", model_name, os.path.split(checkpoints_path)[-1])

os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)
os.makedirs(report_fugure_path, exist_ok=True)

train_images = "data/processed/train/images/"
val_images = "data/processed/validation/images/"

TRAINSET_SIZE = len(glob(train_images + "*.png"))
VALSET_SIZE = len(glob(val_images + "*.png"))

def parse_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    mask_path = tf.strings.regex_replace(img_path, "images", "masks")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.float32)

    return image, mask

train_dataset = tf.data.Dataset.list_files(train_images + "*.png", seed=SEED)
train_dataset = train_dataset.map(parse_image).shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.list_files(val_images + "*.png", seed=SEED)
val_dataset = val_dataset.map(parse_image).batch(BATCH_SIZE)

base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project',
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

class SparseMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

model = unet_model(output_channels=n_classes)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[SparseMeanIOU(num_classes=n_classes)])

callbacks = [
    ModelCheckpoint(
                filepath=os.path.join(checkpoints_path, "{epoch:05d}"),
                save_weights_only=False,
                verbose=True
            ),
    TensorBoard(logs_path)
]

for image, mask in train_dataset.take(1):
  sample_image, sample_mask = image[0], mask[0]

cv2.imwrite(os.path.join(report_fugure_path, "true_mask.png"), np.tile(sample_mask, [1, 1, 3])*50)
predict_and_save(model, sample_image, os.path.join(report_fugure_path, "out_before_training.png"))

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          validation_data=val_dataset,
                          steps_per_epoch=TRAINSET_SIZE//BATCH_SIZE,
                          validation_steps=VALSET_SIZE//BATCH_SIZE,
                          callbacks=callbacks)

predict_and_save(model, sample_image, os.path.join(report_fugure_path, "out_after_training.png"))