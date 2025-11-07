import os
import re
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models


# If GPU, comment out this line.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
CLASS_NAMES = ['benign', 'malignant']
label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

# Paths
DATA_DIR = "BreakHis_v1/BreaKHis_40x_200x/BreaKHis_v1/histology_slides/breast"
OUTPUT_DIR = "./trained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using devices:", tf.config.list_physical_devices())

# ---------------------------
# 1) Build dataframe of images
# ---------------------------

def extract_patient_id(parts):
    """Extract patient ID from a path split into components."""
    for part in parts:
        if re.match(r"SOB_[A-Z]_[A-Z0-9-]+", part):
            return part
    return parts[-3] if len(parts) >= 3 else "unknown"

def build_image_dataframe(data_dir):
    records = []
    for label in ["benign", "malignant"]:
        subdir = os.path.join(data_dir, label)
        if not os.path.exists(subdir):
            continue
        for root, _, files in os.walk(subdir):
            for f in files:
                if f.lower().endswith(".png"):
                    path = os.path.join(root, f)
                    parts = path.split(os.sep)
                    mag = parts[-2] if len(parts) >= 2 else "unknown"
                    mag_norm = re.sub(r'[^0-9]', '', mag)
                    patient_id = extract_patient_id(parts)
                    records.append({
                        "filepath": path,
                        "label": label,
                        "mag": mag_norm,
                        "patient_id": patient_id
                    })
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No images found under {data_dir}. Check DATA_DIR.")
    print(f"Total images: {len(df)}")
    print("Magnification distribution:\n", df['mag'].value_counts())
    return df

df = build_image_dataframe(DATA_DIR)
df = df[df['mag'].isin(['40', '200'])].reset_index(drop=True)
print("Filtered to 40x and 200x images:", len(df))
print(df['mag'].value_counts())

# ---------------------------
# 2) Patient-level split
# ---------------------------

def patient_stratified_split(df, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=SEED):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    splits = {}
    for mag in sorted(df['mag'].unique()):
        sub = df[df['mag'] == mag].copy()
        patients = sub['patient_id'].unique().tolist()
        random.Random(seed).shuffle(patients)
        n = len(patients)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train_pat = set(patients[:n_train])
        val_pat = set(patients[n_train:n_train + n_val])
        test_pat = set(patients[n_train + n_val:])
        splits[mag] = {
            'train': sub[sub['patient_id'].isin(train_pat)].reset_index(drop=True),
            'val': sub[sub['patient_id'].isin(val_pat)].reset_index(drop=True),
            'test': sub[sub['patient_id'].isin(test_pat)].reset_index(drop=True)
        }
    return splits

splits = patient_stratified_split(df)
for mag in splits:
    print(f"\nMagnification {mag}x")
    print("Images per split:", {k: len(v) for k, v in splits[mag].items()})
    print("Patients per split:", {k: v['patient_id'].nunique() for k, v in splits[mag].items()})

# Sanity check: no patient overlap
for mag in splits:
    p_train = set(splits[mag]['train']['patient_id'])
    p_val = set(splits[mag]['val']['patient_id'])
    p_test = set(splits[mag]['test']['patient_id'])
    assert p_train.isdisjoint(p_val)
    assert p_train.isdisjoint(p_test)
    assert p_val.isdisjoint(p_test)
print("\nPatient split verified: no overlap.")

# ---------------------------
# 3) TensorFlow data pipeline
# ---------------------------

AUTOTUNE = tf.data.AUTOTUNE

def path_to_tensor(path, img_size=IMG_SIZE, preprocess=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)
    img = img / 255.0 if preprocess is None else preprocess(img)
    return img

data_augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.05, 0.05),
])

def dataframe_to_dataset(df, preprocess=None, shuffle=False, repeat=False, drop_remainder=True):
    paths = df['filepath'].values
    labels = np.array([label_to_idx[l] for l in df['label']], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    def _load(path, label):
        img = path_to_tensor(path, IMG_SIZE, preprocess)
        return img, tf.cast(label, tf.int32)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=drop_remainder).prefetch(AUTOTUNE)
    return ds

# ---------------------------
# 4) Model definitions
# ---------------------------

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = data_augment(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name='custom_cnn')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_mobilenet(input_shape=(224,224,3), num_classes=2):
    base = tf.keras.applications.MobileNetV2(include_top=False,
                                             weights='imagenet',
                                             input_shape=input_shape)
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = data_augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name='mobilenet_v2')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------
# 5) Training and evaluation
# ---------------------------

def train_and_evaluate(model_fn, splits, mag, name):
    print(f"\nTraining {name} on {mag}x images")

    train_df = splits[mag]['train']
    val_df = splits[mag]['val']
    test_df = splits[mag]['test']

    train_ds = dataframe_to_dataset(train_df, shuffle=True, repeat=False, drop_remainder=True)
    val_ds = dataframe_to_dataset(val_df, shuffle=False, repeat=False, drop_remainder=True)
    test_ds = dataframe_to_dataset(test_df, shuffle=False, repeat=False, drop_remainder=False)

    model = model_fn()
    ckpt_dir = os.path.join(OUTPUT_DIR, f"{mag}x_{name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_model.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        verbose=2)

    # Evaluate
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    y_true = []
    for _, y in test_ds:
        y_true.extend(y.numpy())
    y_true = np.array(y_true)[:len(y_pred)]

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    try:
        auc = float(roc_auc_score(y_true, probs[:, 1]))
    except Exception:
        auc = None

    model.save(os.path.join(ckpt_dir, "model.h5"))
    out = {
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "report": report,
        "auc": auc,
        "test_size": int(len(y_true))
    }

    with open(os.path.join(ckpt_dir, "report.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved model and report in {ckpt_dir}")
    return history, out

# ---------------------------
# 6) Run experiments
# ---------------------------

for mag in ['40', '200']:
    if mag not in splits or len(splits[mag]['train']) == 0:
        print(f"Skipping magnification {mag}: no images found.")
        continue

    print(f"\nMagnification {mag}x")
    train_and_evaluate(build_custom_cnn, splits, mag, "CustomCNN")
    train_and_evaluate(build_mobilenet, splits, mag, "MobileNetV2")

print("\nTraining complete. Models and reports saved under ./trained_models/")
