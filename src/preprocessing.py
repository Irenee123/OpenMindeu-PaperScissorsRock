import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress protobuf warnings

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

def load_data(data_dir, augment=False):
    """
    Loads image data from directory using ImageDataGenerator
    """
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    data = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return data

# Convenience function
def get_train_test_data(train_dir='data/train', test_dir='data/test'):
    train_data = load_data(train_dir, augment=True)
    test_data = load_data(test_dir, augment=False)
    return train_data, test_data
