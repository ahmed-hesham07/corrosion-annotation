import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dimensions for training images and masks
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Directories for training data
TRAIN_IMAGES_DIR = "train_images"   # This directory should have a subfolder "Corrosion/"
TRAIN_MASKS_DIR = "train_masks"     # This directory should have a subfolder "Corrosion/"

def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = layers.Input(input_shape)

    # --- Encoder ---
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # --- Bottleneck ---
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)

    # --- Decoder ---
    u4 = layers.UpSampling2D((2, 2))(c3)
    concat4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(concat4)
    c4 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c4)

    u5 = layers.UpSampling2D((2, 2))(c4)
    concat5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(concat5)
    c5 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c5)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c5)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def create_datagen(image_dir, mask_dir, batch_size=8):
    """
    Creates two ImageDataGenerators for images and masks, pointing to a single subfolder 'Corrosion'.
    """
    image_datagen = ImageDataGenerator(rescale=1./255,
                                       horizontal_flip=True,
                                       vertical_flip=True)
    mask_datagen = ImageDataGenerator(rescale=1./255,
                                      horizontal_flip=True,
                                      vertical_flip=True)

    image_generator = image_datagen.flow_from_directory(
        image_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode=None,    # For segmentation, no class labels
        classes=["Corrosion"],  # <--- Must match your subfolder name
        seed=1
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode=None,
        color_mode="grayscale",
        classes=["Corrosion"],
        seed=1
    )
    return image_generator, mask_generator

def combined_generator(image_gen, mask_gen):
    """
    A custom Python generator that yields (images, masks) tuples.
    """
    while True:
        # Get next batch of images and masks
        imgs = next(image_gen)
        msks = next(mask_gen)
        yield (imgs, msks)

if __name__ == "__main__":
    # 1) Build and compile the model
    model = build_unet()
    model.compile(optimizer=optimizers.Adam(),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # 2) Create the data generators
    image_gen, mask_gen = create_datagen(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, batch_size=4)
    train_gen = combined_generator(image_gen, mask_gen)

    # 3) Fit the model using the custom generator
    steps_per_epoch = 10  # Adjust based on your dataset size (#batches per epoch)
    model.fit(train_gen,
              steps_per_epoch=steps_per_epoch,
              epochs=20)

    # 4) Save the trained model
    model.save("model.h5")
    print("Model saved as model.h5")
