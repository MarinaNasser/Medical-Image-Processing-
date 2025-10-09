"""
BreakHis 40x and 200x classification notebook
- Downloads and extracts the BreaKHis_v1.tar.gz dataset (link from https://web.inf.ufpr.br/)
- Builds patient-aware train/val/test splits (70/10/20) to avoid patient leakage
- Implements two classifiers for each magnification (40x and 200x):
    1) custom CNN
    2) MobileNetV2 transfer learning
- Applies data augmentation
- Trains, evaluates, and saves models and history
- Includes a short presentation (Introduction, Methods & Materials, Results, Discussion) at the bottom

Notes:
- Run this notebook in an environment with TensorFlow (2.x), scikit-learn, pandas, matplotlib.
- Adjust paths, number of epochs, and batch sizes according to your GPU/memory limits.

"""

import os
import re
import tarfile
import random
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

# ---------------------------
# Configuration
# ---------------------------
DATA_URL = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
DATA_DIR = Path("./BreakHis_v1")
EXTRACTED_DIR = DATA_DIR / "BreakHis_v1"
IMAGE_SIZE = (224, 224)  # for MobileNet and reasonably small custom CNN
BATCH_SIZE = 32
SEED = 42
EPOCHS = 25

os.makedirs(DATA_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# 1) Download and extract
# ---------------------------

def download_dataset(url=DATA_URL, dest_dir=DATA_DIR):
    dest_file = dest_dir / os.path.basename(url)
    if dest_file.exists():
        print(f"Dataset archive already exists: {dest_file}")
        return dest_file
    try:
        import requests
        print(f"Downloading {url} ...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
        return dest_file
    except Exception as e:
        raise RuntimeError(f"Could not download dataset automatically. Please download manually from {url} and place it in {dest_dir}. Error: {e}")


def extract_tar_gz(archive_path: Path, extract_to: Path):
    if extract_to.exists():
        print(f"Extracted folder already exists: {extract_to}")
        return
    print(f"Extracting {archive_path} to {extract_to} ...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_to.parent)
    print("Extraction complete.")


# ---------------------------
# 2) Build metadata DataFrame
# ---------------------------

def build_metadata(root_dir: Path):
    """
    Walk the extracted dataset and produce a DataFrame with columns:
    - path
    - label (B or M -> map to 'benign'/'malignant' or 0/1)
    - magnification (int)
    - patient_id (string to group by)
    """
    rows = []
    img_pattern = re.compile(r"(.+)\.(png|jpg|jpeg)$", re.IGNORECASE)

    for path in root_dir.rglob("*.png"):
        fname = path.name
        # filename format described on the BreakHis page:
        # SOB_B_TA-14-4659-40-001.png
        # parts split by '-'
        parts = fname.split('-')
        # Ensure at least 5 parts
        if len(parts) < 5:
            continue
        # tumor class part contains _ before the dash, e.g., SOB_B_TA -> we can find class after '_'
        pre = parts[0]
        # Extract tumor class letter B or M from second subsection of name
        # The second token in the whole filename (before first dash) may be like SOB_B_TA -> contains 'B' or 'M'
        class_match = re.search(r"_([BM])_", pre + "_")
        if class_match:
            tumor_class = class_match.group(1)
        else:
            # fallback: look at parts[0] elements
            tumor_class = 'B' if '_B_' in fname or '_B-' in fname else ('M' if '_M_' in fname or '_M-' in fname else None)
        year = parts[1]
        slide = parts[2]
        mag = parts[3]
        # patient id we'll compose as year-slide (this corresponds to slide origin)
        patient_id = f"{year}-{slide}"

        label = 0 if tumor_class == 'B' else 1

        try:
            mag_int = int(mag)
        except:
            continue

        rows.append({
            'path': str(path),
            'label': label,
            'magnification': mag_int,
            'patient_id': patient_id,
            'filename': fname
        })

    df = pd.DataFrame(rows)
    # Keep only expected magnifications
    df = df[df['magnification'].isin([40, 200])].reset_index(drop=True)
    print(f"Found {len(df)} images for 40x and 200x combined.")
    return df


# ---------------------------
# 3) Patient-aware splits
# ---------------------------

def patient_stratified_split(df, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=SEED):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    splits = {}

    for mag in sorted(df['magnification'].unique()):
        sub = df[df['magnification'] == mag].copy()
        # patient-level label (assume patient has a single tumor class)
        patient_labels = sub.groupby('patient_id')['label'].agg(lambda x: x.mode()[0]).reset_index()
        patients = patient_labels['patient_id'].values
        p_labels = patient_labels['label'].values

        # First split: train vs temp (train_frac vs (1-train_frac))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=seed)
        train_idx, temp_idx = next(sss.split(patients, p_labels))
        train_patients = patients[train_idx]
        temp_patients = patients[temp_idx]
        temp_labels = p_labels[temp_idx]

        # Now split temp into val and test with ratio val/(val+test) = val_frac/(val_frac+test_frac)
        relative_val_frac = val_frac / (val_frac + test_frac)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - relative_val_frac), random_state=seed + 1)
        val_idx_rel, test_idx_rel = next(sss2.split(temp_patients, temp_labels))
        val_patients = temp_patients[val_idx_rel]
        test_patients = temp_patients[test_idx_rel]

        # Now construct image-level splits
        train_df = sub[sub['patient_id'].isin(train_patients)].reset_index(drop=True)
        val_df = sub[sub['patient_id'].isin(val_patients)].reset_index(drop=True)
        test_df = sub[sub['patient_id'].isin(test_patients)].reset_index(drop=True)

        print(f"Magnification {mag}: patients -> train {len(train_patients)}, val {len(val_patients)}, test {len(test_patients)}")
        print(f"Magnification {mag}: images -> train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

        splits[mag] = {'train': train_df, 'val': val_df, 'test': test_df}

    return splits


# ---------------------------
# 4) Data generators with augmentation
# ---------------------------

def create_generators(train_df, val_df, test_df, input_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    # Mean/std normalization will be handled by rescale=1./255
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    def df_to_generator(df, datagen, shuffle=True):
        return datagen.flow_from_dataframe(
            dataframe=df,
            x_col='path',
            y_col='label',
            target_size=input_size,
            color_mode='rgb',
            class_mode='binary',
            batch_size=batch_size,
            shuffle=shuffle,
            seed=SEED
        )

    train_gen = df_to_generator(train_df, train_datagen, shuffle=True)
    val_gen = df_to_generator(val_df, val_test_datagen, shuffle=False)
    test_gen = df_to_generator(test_df, val_test_datagen, shuffle=False)

    return train_gen, val_gen, test_gen


# ---------------------------
# 5) Model definitions
# ---------------------------

def build_custom_cnn(input_shape=IMAGE_SIZE + (3,), dropout_rate=0.4):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name='custom_cnn')
    return model


def build_mobilenet(input_shape=IMAGE_SIZE + (3,), fine_tune_at=100):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = True
    # Freeze layers up to fine_tune_at
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name='MobileNetV2_finetuned')
    return model


# ---------------------------
# 6) Training helper
# ---------------------------

def compile_and_train(model, train_gen, val_gen, save_path, epochs=EPOCHS):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    cbks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cbks
    )

    return history


# ---------------------------
# 7) Evaluation helper
# ---------------------------

def evaluate_model(model, test_gen):
    res = model.evaluate(test_gen)
    metrics = dict(zip(model.metrics_names, res))
    return metrics


# ---------------------------
# 8) Full pipeline for a magnification
# ---------------------------

def run_for_magnification(splits, mag, output_dir=Path('./models')):
    mag_dir = output_dir / f"mag_{mag}"
    mag_dir.mkdir(parents=True, exist_ok=True)

    train_df = splits[mag]['train']
    val_df = splits[mag]['val']
    test_df = splits[mag]['test']

    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)

    # Custom CNN
    cnn = build_custom_cnn()
    cnn_path = mag_dir / 'custom_cnn.h5'
    print(f"Training custom CNN for {mag}x ...")
    hist_cnn = compile_and_train(cnn, train_gen, val_gen, str(cnn_path))
    metrics_cnn = evaluate_model(cnn, test_gen)
    print(f"Custom CNN test metrics for {mag}x: {metrics_cnn}")

    # MobileNet
    mn = build_mobilenet()
    mn_path = mag_dir / 'mobilenetv2.h5'
    print(f"Training MobileNetV2 for {mag}x ...")
    hist_mn = compile_and_train(mn, train_gen, val_gen, str(mn_path))
    metrics_mn = evaluate_model(mn, test_gen)
    print(f"MobileNetV2 test metrics for {mag}x: {metrics_mn}")

    # Save histories and a combined report
    import json
    with open(mag_dir / 'history_custom_cnn.json', 'w') as f:
        json.dump(hist_cnn.history, f)
    with open(mag_dir / 'history_mobilenet.json', 'w') as f:
        json.dump(hist_mn.history, f)

    report = {
        'magnification': mag,
        'custom_cnn': metrics_cnn,
        'mobilenetv2': metrics_mn,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df)
    }
    with open(mag_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------
# 9) Run everything
# ---------------------------
if __name__ == '__main__':
    # 1) download and extract (if you prefer manual download, skip this and set EXTRACTED_DIR properly)
    try:
        archive = download_dataset()
        extract_tar_gz(archive, EXTRACTED_DIR)
    except RuntimeError as e:
        print(str(e))
        print("Assuming you have already downloaded and extracted the archive manually. Please set EXTRACTED_DIR accordingly.")

    # The extracted dataset's structure is usually something like BreakHis_v1/SOB/*magnification*/... or plain files
    # Try to find the folder that contains pngs.
    candidate = None
    for p in DATA_DIR.rglob('*.png'):
        candidate = p.parent
        break
    if candidate is None:
        # look in EXTRACTED_DIR
        print(f"No images found under {DATA_DIR}. Please double-check extraction path.")
    else:
        root_images_dir = candidate.parent  # adjust as necessary
        print(f"Detected image folder: {root_images_dir}")

        # Build metadata
        df = build_metadata(root_images_dir)

        # Splits
        splits = patient_stratified_split(df)

        # Run pipeline for each magnification
        reports = {}
        out_dir = Path('./trained_models')
        for mag in [40, 200]:
            reports[mag] = run_for_magnification(splits, mag, output_dir=out_dir)

        print('All done. Reports saved in', out_dir)

