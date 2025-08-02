import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from glob import glob

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import class_weight
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

df = pd.read_csv(r"C:\Users\rohan\Downloads\archive\HAM10000_metadata.csv")
print(df.head())
df.info()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

lesion_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
skin_dir = r"C:\Users\rohan\Downloads\archive"

image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(skin_dir, '*', '*.jpg'))}

df['path'] = df['image_id'].map(image_path_dict.get)
df = df.dropna(subset=['path'])
df['cell_type'] = df['dx'].map(lesion_dict.get)
df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

df.head()

print(df.isna().sum())
df['age'] = df['age'].fillna(df['age'].mean())
df['sex'] = df['sex'].fillna('unknown')
df['localization'] = df['localization'].fillna('unknown')

df = pd.get_dummies(df, columns=['sex', 'localization'])

df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

metadata_features = ['age'] + [col for col in df.columns if col.startswith('sex_') or col.startswith('localization_')]

print(df.isna().sum())

print(df['path'].isna().sum())


def cnn():
    cnn_model = tf.keras.models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        # Block 1
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(64, (5, 5), padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),  # 0.4

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),  # 0.4

        # Fully Connected
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),  # 0.5
    ])
    return cnn_model


def meta(input_dim):
    meta_model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(32, activation='relu'),
    ])
    return meta_model


def augment_image(img):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])
    return data_augmentation(img)


def preprocess_combined(path, metadata, label):
    # Image
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)
    img = img / 255.0
    img = augment_image(img)
    metadata = tf.cast(metadata, tf.float32)
    return (img, metadata), label


def create_dataset(df, metadata_features, label_ser, batch_size=32, shuffle=True):
    # 1) drop bad rows (just in case)
    df = df.dropna(subset=['path'])
    # 2) pull out three parallel lists/arrays
    paths    = df['path'].tolist()                                # list of file‑paths
    meta_np  = df[metadata_features].to_numpy(dtype='float32')    # (N, D)
    labels   = to_categorical(label_ser.to_numpy(), num_classes=7)  # (N,7)

    # 3) build a 3‑column dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, meta_np, labels))

    # 4) parse each row
    def _parse(path, meta, lab):
        # load + normalize + augment image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])  # simpler than resize_with_pad
        img = img / 255.0
        img = augment_image(img)
        # metadata is already float32
        return (
            {'image_input': img,  # matches Input(name='image_input', …)
             'meta_input': meta},
            lab
        )

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1_000, seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['cell_type_idx'], random_state=42)

train_ds = create_dataset(train_df, metadata_features, train_df['cell_type_idx'], batch_size=8, shuffle=True)
val_ds = create_dataset(test_df, metadata_features, test_df['cell_type_idx'], batch_size=8, shuffle=False)

cnn_branch = cnn()
meta_branch = meta(input_dim=len(metadata_features))

image_input = Input(shape=(224, 224, 3), name='image_input')
meta_input = Input(shape=(len(metadata_features),), name='meta_input')

cnn_features = cnn_branch(image_input)
meta_features = meta_branch(meta_input)

combined = layers.concatenate([cnn_features, meta_features])

x = layers.Dense(64, activation='relu')(combined)
output = layers.Dense(7, activation='softmax')(x)

model = tf.keras.Model(inputs=[image_input, meta_input], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stop],
)

# model_path = r"C:\Users\rohan\Documents\ham10000_multimodal_model.h5"
# model.save(model_path)

# for (img_batch, meta_batch), label_batch in val_ds.take(1):
#     predictions = model.predict([img_batch, meta_batch])
#
#     predicted_classes = np.argmax(predictions, axis=1)
#     true_classes = np.argmax(label_batch.numpy(), axis=1)
#
#     inv_lesion_dict = {v: k for k, v in lesion_dict.items()}
#     index_to_label = dict(enumerate(df['cell_type'].astype('category').cat.categories))
#
#     print("Inference Results:\n")
#     for i in range(len(predicted_classes)):
#         print(f"Sample {i+1}:")
#         print(f"  Predicted: {index_to_label[predicted_classes[i]]}")
#         print(f"  Actual:    {index_to_label[true_classes[i]]}")
#         print(f"  Probabilities: {np.round(predictions[i], 3)}\n")
#
#         plt.imshow(img_batch[i].numpy())
#         plt.title(f"Predicted: {index_to_label[predicted_classes[i]]} | Actual: {index_to_label[true_classes[i]]}")
#         plt.axis('off')
#         plt.show()

# test_loss, test_acc = model.evaluate(val_ds)
# print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
#
# plt.plot(history.history['accuracy'], label='train accuracy')
# plt.plot(history.history['val_accuracy'], label='val accuracy')
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.legend()
# plt.show()
