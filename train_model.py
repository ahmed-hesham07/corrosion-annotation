import tensorflow as tf
tf.keras.utils.generic_utils = tf.keras.utils

import os
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image dimensions and batch parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 4
EPOCHS = 20

# Directories for training images and masks (must include a subfolder named "Corrosion")
TRAIN_IMAGES_DIR = "train_images"
TRAIN_MASKS_DIR = "train_masks"

# Ensure TensorFlow uses CPU only (optional)
tf.config.set_visible_devices([], 'GPU')

# Set segmentation_models parameters
sm.set_framework('tf.keras')
sm.framework()

# Choose a backbone – using 'resnet34' (lightweight enough for CPU)
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Define loss and metrics: using a combination of binary crossentropy and Dice loss
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5]))
bce_loss = tf.keras.losses.BinaryCrossentropy()
combined_loss = lambda y_true, y_pred: bce_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

def build_model():
    model = sm.Unet(BACKBONE,
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                    encoder_weights='imagenet',
                    classes=1,
                    activation='sigmoid')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[sm.metrics.iou_score]
    )
    return model

def create_generators(batch_size=BATCH_SIZE):
    data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        rescale=1./255
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Ensure your training data is organized under a subfolder "Corrosion"
    image_generator = image_datagen.flow_from_directory(
        TRAIN_IMAGES_DIR,
        classes=["Corrosion"],
        class_mode=None,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        seed=42
    )
    mask_generator = mask_datagen.flow_from_directory(
        TRAIN_MASKS_DIR,
        classes=["Corrosion"],
        class_mode=None,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=batch_size,
        seed=42
    )

    def combined_gen(img_gen, msk_gen):
        while True:
            imgs = next(img_gen)
            msks = next(msk_gen)
            yield (imgs, msks)

    return combined_gen(image_generator, mask_generator), image_generator.samples

if __name__ == "__main__":
    model = build_model()
    model.summary()

    train_generator, train_samples = create_generators()
    steps_per_epoch = train_samples // BATCH_SIZE

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS
    )

    model.save("model.h5")
    print("Model saved as model.h5")
