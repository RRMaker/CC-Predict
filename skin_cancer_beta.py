import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import itertools
import math
import joblib
import random

from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from keras.utils.vis_utils import plot_model

df = pd.read_csv(r"C:\Users\rohan\Downloads\archive\HAM10000_metadata.csv")
df.head()
df.info()

lesion_type_dict = {
   'nv': 'Melanocytic nevi',
   'mel': 'Melanoma',
   'bkl': 'Benign keratosis-like lesions ',
   'bcc': 'Basal cell carcinoma',
   'akiec': 'Actinic keratoses',
   'vasc': 'Vascular lesions',
   'df': 'Dermatofibroma'
}
base_skin_dir = r"C:\Users\rohan\Downloads\archive"

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

df['path'] = df['image_id'].map(imageid_path_dict.get)
df['cell_type'] = df['dx'].map(lesion_type_dict.get)
df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

print(df.head())
print(df.isna().sum())

df = df[df['path'].notna()]

df['age'] = df['age'].fillna(df['age'].mean())
df['sex'] = df['sex'].fillna('unknown')
df['localization'] = df['localization'].fillna('unknown')

le_sex = LabelEncoder()
le_loc = LabelEncoder()
df['sex_enc'] = le_sex.fit_transform(df['sex'])
df['loc_enc'] = le_loc.fit_transform(df['localization'])

metadata_features = df[['age', 'sex_enc', 'loc_enc']].copy()
scaler = StandardScaler()
metadata_scaled = scaler.fit_transform(metadata_features)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_sex, "le_sex.pkl")
joblib.dump(le_loc, "le_loc.pkl")

df['metadata'] = list(metadata_scaled)

print(df.isna().sum().sum())

df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
df['image'].map(lambda x: x.shape).value_counts()

features = df[['image', 'metadata']]
target = df['cell_type_idx']

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=1234
)

x_train_images = np.stack(x_train_o['image'].values)
x_test_images = np.stack(x_test_o['image'].values)

x_train_meta = np.stack(x_train_o['metadata'].values)
x_test_meta = np.stack(x_test_o['metadata'].values)

x_train_images = (x_train_images - np.mean(x_train_images)) / np.std(x_train_images)
x_test_images = (x_test_images - np.mean(x_test_images)) / np.std(x_test_images)

y_train = to_categorical(y_train_o, num_classes=7)
y_test = to_categorical(y_test_o, num_classes=7)

x_train_images, x_val_images, x_train_meta, x_val_meta, y_train, y_val = train_test_split(
   x_train_images, x_train_meta, y_train, test_size=0.1, random_state=2
)

x_train_images = x_train_images.reshape(x_train_images.shape[0], 75, 100, 3)
x_test_images = x_test_images.reshape(x_test_images.shape[0], 75, 100, 3)
x_val_images = x_val_images.reshape(x_val_images.shape[0], 75, 100, 3)

image_input = Input(shape=(75, 100, 3))
x = Conv2D(32, (3,3), activation='relu', padding='same')(image_input)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.1)(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.1)(x)

x = Flatten()(x)


meta_input = Input(shape=(3,))
m = Dense(64, activation='relu')(meta_input)
m = Dropout(0.3)(m)


combined = Concatenate()([x, m])
z = Dense(128, activation='relu')(combined)
z = Dropout(0.25)(z)
output = Dense(7, activation='softmax')(z)

model = Model(inputs=[image_input, meta_input], outputs=output)
model.summary()

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)


def multimodal_generator(images, metadata, labels, batch_size):
   datagen = ImageDataGenerator(
       featurewise_center=False,
       samplewise_center=False,
       featurewise_std_normalization=False,
       samplewise_std_normalization=False,
       zca_whitening=False,
       rotation_range=10,
       zoom_range=0.1,
       width_shift_range=0.1,
       height_shift_range=0.1,
       horizontal_flip=True,
       vertical_flip=True,
       fill_mode='nearest'
   )
   image_gen = datagen.flow(images, labels, batch_size=batch_size, seed=42)
   while True:
       img_batch, label_batch = next(image_gen)
       batch_indices = image_gen.index_array[:img_batch.shape[0]]
       meta_batch = metadata[batch_indices]
       yield [img_batch, meta_batch], label_batch


epochs = 50
batch_size = 32

train_generator = multimodal_generator(x_train_images, x_train_meta, y_train, batch_size)

history = model.fit(
   train_generator,
   steps_per_epoch=math.ceil(x_train_images.shape[0] / batch_size),
   epochs=epochs,
   validation_data=([x_val_images, x_val_meta], y_val),
   callbacks=[learning_rate_reduction],
   verbose=1
)

plot_model(model, to_file='model_plot_2.png', show_shapes=True, show_layer_names=True)

test_loss, test_acc = model.evaluate([x_test_images, x_test_meta], y_test, verbose=1)
val_loss,  val_acc = model.evaluate([x_val_images, x_val_meta], y_val, verbose=1)
print(f"Test  → accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
print(f"Val   → accuracy: {val_acc:.4f}, loss: {val_loss:.4f}")
model.save("model.h5")

y_pred_probs = model.predict([x_test_images, x_test_meta])
y_test_bin = y_test

roc_auc_weighted = roc_auc_score(
    y_test_bin,
    y_pred_probs,
    multi_class='ovr',
    average='weighted'
)

print(f"ROC AUC (weighted, OVR): {roc_auc_weighted:.4f}")

# history_dict = history.history
# epochs_range = range(1, len(history_dict['loss'])+1)
# plt.figure(figsize=(8,6))
# plt.plot(epochs_range, history_dict['loss'],    label='Train Loss')
# plt.plot(epochs_range, history_dict['val_loss'],label='Val   Loss')
# plt.hlines(test_loss, epochs_range[0], epochs_range[-1],
#            linestyles='--', label=f'Test Loss ({test_loss:.2%})')
# plt.title('Loss over Epochs')
# plt.xlabel('Epoch'); plt.ylabel('Loss')
# plt.legend(); plt.grid(True); plt.show()

# y_pred_probs = model.predict([x_test_images, x_test_meta])
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)
#
# cm = confusion_matrix(y_true, y_pred)
#
# labels = list(lesion_type_dict.values())
#
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# plt.figure(figsize=(10, 8))
# disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()
#
# print(classification_report(y_true, y_pred, target_names=labels))
