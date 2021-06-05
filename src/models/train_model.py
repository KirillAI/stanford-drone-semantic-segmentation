from keras_segmentation.models.unet  import mobilenet_unet
import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
from src.data.make_dataset import IMG_WIDTH, IMG_HEIGHT, n_classes

epochs = 15

checkpoints_path = "models/mobilenet_unet/{}/".format(int(time.time()))
logs_path = "{}/logs".format(checkpoints_path)
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)

model = mobilenet_unet(n_classes=n_classes ,  input_height=IMG_HEIGHT, input_width=IMG_WIDTH)

callbacks = [
    ModelCheckpoint(
                filepath=checkpoints_path+"{epoch:05d}",
                save_weights_only=False,
                verbose=True
            ),
    TensorBoard(logs_path)
]

model.train(
    train_images="data/processed/train/images/",
    train_annotations="data/processed/train/masks/",
    val_images="data/processed/validation/images/",
    val_annotations="data/processed/validation/masks/",
    checkpoints_path=checkpoints_path , epochs=epochs
)