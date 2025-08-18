# src/autoencoder_train.py
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Paths & Defaults
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_DIR      = PROJECT_ROOT / "images" / "segmented_images"
MODELS_DIR   = PROJECT_ROOT / "models"
LOGS_DIR     = PROJECT_ROOT / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Utilities
def set_determinism(seed: int):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_gpu(mixed_precision: bool):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")
        print("[info] Mixed precision enabled (float16 compute)")

# robust dataset builders 
def list_image_paths(img_dir: Path):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [str(p) for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"No segmented images found in {img_dir}. Run segmentation first.")
    return files

def build_datasets(img_dir: Path, img_size=(256, 256), batch_size=16, val_split=0.2, seed=42):
    import math, random
    files = list_image_paths(img_dir)
    random.Random(seed).shuffle(files)

    n_total = len(files)
    n_val = max(1, int(n_total * val_split))
    n_train = max(1, n_total - n_val)
    train_files = files[:n_train]
    val_files   = files[n_train:] if n_total > 1 else files[:1]

    def _loader(path):
        img_b = tf.io.read_file(path)
        img = tf.io.decode_image(img_b, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img, img  # AE target = input

    ds_train = (tf.data.Dataset.from_tensor_slices(train_files)
                .shuffle(buffer_size=len(train_files), seed=seed, reshuffle_each_iteration=True)
                .map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    ds_val = (tf.data.Dataset.from_tensor_slices(val_files)
              .map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))

    steps_per_epoch = math.ceil(len(train_files) / batch_size)
    validation_steps = math.ceil(len(val_files) / batch_size)

    return ds_train, ds_val, steps_per_epoch, validation_steps, len(train_files), len(val_files)

# -------------------------------------------------------------------------
def build_autoencoder():
    inp = Input(shape=(256, 256, 3))
    # Encoder
    x = Conv2D(128, 3, activation='relu', padding='same')(inp); x = BatchNormalization()(x); x = MaxPooling2D(2, padding='same')(x); x = Dropout(0.2)(x)
    x = Conv2D(64,  3, activation='relu', padding='same')(x);  x = BatchNormalization()(x); x = MaxPooling2D(2, padding='same')(x); x = Dropout(0.2)(x)
    x = Conv2D(32,  3, activation='relu', padding='same')(x);  x = BatchNormalization()(x); x = MaxPooling2D(2, padding='same')(x); x = Dropout(0.2)(x)
    x = Conv2D(16,  3, activation='relu', padding='same')(x);  x = BatchNormalization()(x); enc = MaxPooling2D(2, padding='same')(x); enc = Dropout(0.2)(enc)
    # Decoder
    y = Conv2D(16, 3, activation='relu', padding='same')(enc); y = BatchNormalization()(y); y = UpSampling2D(2)(y); y = Dropout(0.2)(y)
    y = Conv2D(32, 3, activation='relu', padding='same')(y);   y = BatchNormalization()(y); y = UpSampling2D(2)(y); y = Dropout(0.2)(y)
    y = Conv2D(64, 3, activation='relu', padding='same')(y);   y = BatchNormalization()(y); y = UpSampling2D(2)(y); y = Dropout(0.2)(y)
    y = Conv2D(128,3, activation='relu', padding='same')(y);   y = BatchNormalization()(y); y = UpSampling2D(2)(y)
    out = Conv2D(3, 3, activation='sigmoid', padding='same')(y)

    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-4), loss='mean_squared_error')
    return model

def save_model_summary(model: Model, path: Path):
    with open(path, "w") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

def get_encoder_from_autoencoder(ae: Model) -> Model:
    # Take the layer just BEFORE the first UpSampling2D (start of decoder)
    from tensorflow.keras.layers import UpSampling2D as _Up
    up_idx = None
    for i, lyr in enumerate(ae.layers):
        if isinstance(lyr, _Up):
            up_idx = i
            break
    bottleneck_name = ae.layers[up_idx - 1].name if up_idx is not None else ae.layers[-1].name
    enc = Model(inputs=ae.input, outputs=ae.get_layer(bottleneck_name).output)
    return enc

def plot_training_curves(history, out_png: Path):
    plt.figure(figsize=(7,5))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history.get('val_loss', []), label='val')
    plt.title('Autoencoder Training')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed_precision", action="store_true", help="Enable float16 mixed precision (GPU)")
    args = ap.parse_args()

    set_determinism(args.seed)
    setup_gpu(args.mixed_precision)

    # Build datasets (split BEFORE batching; compute steps explicitly)
    ds_train, ds_val, steps_per_epoch, validation_steps, n_train, n_val = build_datasets(
        SEG_DIR, img_size=(256, 256), batch_size=args.batch_size, val_split=args.val_split, seed=args.seed
    )
    print(f"[data] train images: {n_train}, val images: {n_val}")

    # Model
    model = build_autoencoder()
    save_model_summary(model, LOGS_DIR / "autoencoder_summary.txt")

    ckpt = MODELS_DIR / "best_autoencoder.weights.h5"
    cb = [
        ModelCheckpoint(str(ckpt), save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=cb,
        verbose=1
    )

    # Save final artifacts
    model.save(str(MODELS_DIR / "autoencoder.h5"))
    enc = get_encoder_from_autoencoder(model)
    enc.save(str(MODELS_DIR / "encoder.h5"))

    # Logs
    import pandas as pd
    hist_csv = LOGS_DIR / "autoencoder_history.csv"
    pd.DataFrame(history.history).to_csv(hist_csv, index=False)
    plot_training_curves(history, LOGS_DIR / "training_loss.png")

    # Run info
    run_meta = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_split": args.val_split,
        "seed": args.seed,
        "n_images": n_train + n_val,
        "mixed_precision": bool(args.mixed_precision),
    }
    (LOGS_DIR / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    print(f"[done] Saved: models/autoencoder.h5, models/encoder.h5, logs/*")

if __name__ == "__main__":
    main()

