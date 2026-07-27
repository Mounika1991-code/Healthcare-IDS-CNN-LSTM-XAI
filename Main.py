import argparse
import hashlib
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks as kc
from tensorflow.keras import layers as kl
from tensorflow.keras import models as km
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Dataset"
OUTPUT_DIR = ROOT / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_ehr_data():
    """Load and standardize healthcare access logs."""
    print("Step 1: Loading the ER log dataset...")
    path = DATA_DIR / "ER_logs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["hour"] = df["Timestamp"].dt.hour
        df["day_of_week"] = df["Timestamp"].dt.dayofweek
        df["month"] = df["Timestamp"].dt.month
    if "Patient_Age" in df.columns:
        df["Patient_Age"] = pd.to_numeric(df["Patient_Age"], errors="coerce")
    if "Duration_Seconds" in df.columns:
        df["Duration_Seconds"] = pd.to_numeric(df["Duration_Seconds"], errors="coerce")
    if "Access_Success" in df.columns:
        df["Access_Success"] = df["Access_Success"].astype(str).str.strip().str.lower()
        df["Access_Success"] = df["Access_Success"].map({"yes": 1, "no": 0, "true": 1, "false": 0}).fillna(0.5)
    if "Patient_ID" in df.columns:
        df["Patient_ID"] = df["Patient_ID"].astype(str)
    else:
        df["Patient_ID"] = [f"row_{i}" for i in range(len(df))]

    if "Anomaly" not in df.columns:
        notes = df.get("Notes", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.lower()
        suspicious_words = ["unauthorized", "snooping", "possible", "false positive", "curiosity", "unknown", "suspicious"]
        df["Anomaly"] = notes.str.contains("|".join(suspicious_words), regex=True).astype(int)
    else:
        df["Anomaly"] = pd.to_numeric(df["Anomaly"], errors="coerce").fillna(0).astype(int)

    for col in ["Access_Type", "Access_Reason", "Notes", "Attachments"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["source"] = "synthetic"
    df["is_attachment_present"] = df.get("Attachments", "").astype(str).str.len() > 0
    df["duration_sq"] = df["Duration_Seconds"].fillna(0) ** 2
    df["age_duration"] = df["Patient_Age"].fillna(0) * df["Duration_Seconds"].fillna(0)
    df["access_success_flag"] = df.get("Access_Success", 0).fillna(0)
    df["patient_id_hash"] = df["Patient_ID"].apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    print(f"Loaded {len(df)} synthetic EHR rows.")
    return df


def load_mimic_profiles():
    print("Step 2: Building MIMIC-III behavioral profiles...")
    patients = pd.read_csv(DATA_DIR / "MIMIC-III" / "patients.csv", usecols=["subject_id", "gender", "anchor_age"], nrows=12000)
    admissions = pd.read_csv(
        DATA_DIR / "MIMIC-III" / "admissions.csv",
        usecols=["subject_id", "admission_type", "admission_location", "insurance", "marital_status", "race", "hospital_expire_flag"],
        nrows=20000,
    )
    diagnoses = pd.read_csv(DATA_DIR / "MIMIC-III" / "diagnoses_icd.csv", usecols=["subject_id", "icd_code"], nrows=20000)
    prescriptions = pd.read_csv(DATA_DIR / "MIMIC-III" / "prescriptions.csv", usecols=["subject_id", "drug", "route", "dose_val_rx"], nrows=20000)
    drgcodes = pd.read_csv(DATA_DIR / "MIMIC-III" / "drgcodes.csv", usecols=["subject_id", "description", "drg_severity", "drg_mortality"], nrows=20000)

    prescriptions["dose_val_rx"] = pd.to_numeric(prescriptions["dose_val_rx"], errors="coerce")

    patients = patients.dropna(subset=["subject_id"]).copy()
    patients["gender"] = patients["gender"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).fillna(0.5)
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce")

    admissions = admissions.dropna(subset=["subject_id"]).copy()
    admissions["admission_type"] = admissions["admission_type"].fillna("UNK")
    admissions["admission_location"] = admissions["admission_location"].fillna("UNK")
    admissions["insurance"] = admissions["insurance"].fillna("UNK")
    admissions["marital_status"] = admissions["marital_status"].fillna("UNK")
    admissions["race"] = admissions["race"].fillna("UNK")
    admissions["hospital_expire_flag"] = pd.to_numeric(admissions["hospital_expire_flag"], errors="coerce").fillna(0)

    diagnose_stats = diagnoses.groupby("subject_id").agg(diag_count=("icd_code", "count"), diag_unique=("icd_code", "nunique")).reset_index()
    prescription_stats = (
        prescriptions.groupby("subject_id")
        .agg(med_count=("drug", "count"), med_unique=("drug", "nunique"), route_unique=("route", "nunique"), dose_mean=("dose_val_rx", "mean"))
        .reset_index()
    )
    drg_stats = (
        drgcodes.groupby("subject_id")
        .agg(drg_count=("description", "count"), drg_severity_mean=("drg_severity", "mean"), drg_mortality_mean=("drg_mortality", "mean"))
        .reset_index()
    )
    admission_stats = (
        admissions.groupby("subject_id")
        .agg(
            n_admissions=("admission_type", "count"),
            admission_type_unique=("admission_type", "nunique"),
            location_unique=("admission_location", "nunique"),
            insurance_unique=("insurance", "nunique"),
            marital_unique=("marital_status", "nunique"),
            race_unique=("race", "nunique"),
            hospital_expire_flag_any=("hospital_expire_flag", "max"),
        )
        .reset_index()
    )

    profiles = patients.merge(admission_stats, on="subject_id", how="left")
    profiles = profiles.merge(diagnose_stats, on="subject_id", how="left")
    profiles = profiles.merge(prescription_stats, on="subject_id", how="left")
    profiles = profiles.merge(drg_stats, on="subject_id", how="left")

    for col in [
        "n_admissions",
        "admission_type_unique",
        "location_unique",
        "insurance_unique",
        "marital_unique",
        "race_unique",
        "hospital_expire_flag_any",
        "diag_count",
        "diag_unique",
        "med_count",
        "med_unique",
        "route_unique",
        "dose_mean",
        "drg_count",
        "drg_severity_mean",
        "drg_mortality_mean",
    ]:
        profiles[col] = pd.to_numeric(profiles[col], errors="coerce").fillna(0)
    profiles = profiles.fillna(0)
    profiles = profiles.rename(columns={"subject_id": "mimic_subject_id"})
    return profiles


def combine_features(ehr_df, mimic_profiles):
    print("Step 3: Combining features and encoding categories...")
    subject_ids = mimic_profiles["mimic_subject_id"].astype(int).tolist()
    if not subject_ids:
        raise ValueError("No MIMIC profiles were created")

    ehr_df = ehr_df.copy()
    ehr_df["mimic_subject_id"] = ehr_df["patient_id_hash"].apply(lambda x: subject_ids[int(x[:8], 16) % len(subject_ids)])
    combined = ehr_df.merge(mimic_profiles, on="mimic_subject_id", how="left")

    combined["mimic_age_gap"] = combined["Patient_Age"].fillna(0) - combined["anchor_age"].fillna(0)
    combined["mimic_duration_admission"] = combined["Duration_Seconds"].fillna(0) * combined["n_admissions"].fillna(0)
    combined["mimic_diagnosis_medication"] = combined["diag_count"].fillna(0) * combined["med_count"].fillna(0)
    combined["mimic_high_risk"] = ((combined["hospital_expire_flag_any"].fillna(0) > 0) | (combined["drg_mortality_mean"].fillna(0) > 1.0)).astype(int)

    feature_columns = [
        "Patient_Age",
        "Duration_Seconds",
        "Access_Success",
        "hour",
        "day_of_week",
        "month",
        "is_attachment_present",
        "duration_sq",
        "age_duration",
        "access_success_flag",
        "gender",
        "anchor_age",
        "n_admissions",
        "admission_type_unique",
        "location_unique",
        "insurance_unique",
        "marital_unique",
        "race_unique",
        "hospital_expire_flag_any",
        "diag_count",
        "diag_unique",
        "med_count",
        "med_unique",
        "route_unique",
        "dose_mean",
        "drg_count",
        "drg_severity_mean",
        "drg_mortality_mean",
        "mimic_age_gap",
        "mimic_duration_admission",
        "mimic_diagnosis_medication",
        "mimic_high_risk",
    ]

    categorical_columns = []
    for column in ["Hospital", "ER_Room", "User_Role", "Patient_Gender", "Access_Type", "Access_Reason", "Location", "Device_ID", "Access_Method", "Shift", "Department", "Disease", "Insurance_Code", "Treatment", "source"]:
        if column in combined.columns:
            categorical_columns.append(column)
    feature_columns.extend(categorical_columns)

    frame = combined[feature_columns + ["Anomaly"]].copy().fillna(0)
    for column in categorical_columns:
        frame[column] = frame[column].astype(str)

    encoded = pd.get_dummies(frame.drop(columns=["Anomaly"]), columns=categorical_columns, dummy_na=False)
    encoded["Anomaly"] = frame["Anomaly"].astype(int)
    encoded = encoded.astype(float)
    return encoded


def bootstrap_multiple_linear_imputation(df, numeric_columns=None, n_boot=12, random_state=42):
    """Impute missing values using multiple linear regression with bootstrap resampling."""
    df = df.copy()
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    rng = np.random.RandomState(random_state)
    for col in numeric_columns:
        if df[col].isna().sum() == 0:
            continue
        predictors = [c for c in numeric_columns if c != col]
        if len(predictors) == 0:
            df[col] = df[col].fillna(df[col].median())
            continue

        complete = df[predictors + [col]].dropna().copy()
        if complete.shape[0] < 5:
            df[col] = df[col].fillna(df[col].median())
            continue

        X_complete = complete[predictors].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_complete = pd.to_numeric(complete[col], errors="coerce").fillna(0.0)
        predictions = []

        for _ in range(n_boot):
            sample_idx = rng.choice(len(complete), size=len(complete), replace=True)
            sample_x = X_complete.iloc[sample_idx]
            sample_y = y_complete.iloc[sample_idx]
            try:
                model = LinearRegression()
                model.fit(sample_x, sample_y)
                missing_idx = df.index[df[col].isna()].tolist()
                if not missing_idx:
                    break
                missing_x = df.loc[missing_idx, predictors].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                predictions.append(model.predict(missing_x))
            except Exception:
                continue

        if predictions:
            pred_array = np.vstack(predictions).mean(axis=0)
            df.loc[df[col].isna(), col] = pred_array
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def apply_zscore_scaling(df):
    """Apply z-score normalization across all numeric features."""
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        return df
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_columns])
    return pd.DataFrame(scaled, columns=numeric_columns, index=df.index)


def build_gradcam_model(model, conv_layer_name="conv2d_4"):
    """Create a model that outputs the last conv layer activations and final predictions."""
    last_conv_layer = model.get_layer(conv_layer_name)
    if not hasattr(last_conv_layer, "output") or last_conv_layer.output is None:
        input = tf.zeros((1,) + model.input_shape[1:], dtype=tf.float32)
        _ = model(input)
    return tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])


def gradcam_plus_plus_heatmap(model, image, class_index, conv_layer_name="conv2d_4"):
    """Compute a Grad-CAM++ heatmap for a single input image."""
    grad_model = build_gradcam_model(model, conv_layer_name)
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    grads = tf.maximum(grads, 0.0)

    first_term = tf.square(grads)
    second_term = grads * conv_outputs
    alpha_num = first_term
    alpha_denom = 2.0 * first_term + tf.reduce_sum(second_term, axis=[1, 2], keepdims=True) + 1e-8
    alpha = alpha_num / alpha_denom
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=[1, 2])

    cam = tf.reduce_sum(tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_outputs), axis=-1)
    cam = tf.nn.relu(cam)
    cam = tf.image.resize(cam[..., tf.newaxis], (image.shape[1], image.shape[2]))
    heatmap = cam[..., 0] / (tf.reduce_max(cam[..., 0]) + 1e-8)
    return tf.squeeze(heatmap).numpy()


def scorecam_heatmap(model, image, class_index, conv_layer_name="conv2d_4", max_maps=64):
    """Compute a Score-CAM heatmap for a single input image."""
    activation_model = tf.keras.models.Model(model.inputs, model.get_layer(conv_layer_name).output)
    activation_maps = activation_model(image)
    activation_maps = tf.cast(activation_maps, tf.float32)

    if activation_maps.shape[-1] > max_maps:
        activation_maps = activation_maps[..., :max_maps]

    upsampled_maps = tf.image.resize(activation_maps, (image.shape[1], image.shape[2]))
    min_val = tf.reduce_min(upsampled_maps, axis=[1, 2], keepdims=True)
    max_val = tf.reduce_max(upsampled_maps, axis=[1, 2], keepdims=True)
    norm_maps = (upsampled_maps - min_val) / (max_val - min_val + 1e-8)

    weights = []
    for i in range(norm_maps.shape[-1]):
        mask = tf.expand_dims(norm_maps[..., i], axis=-1)
        masked_input = image * mask
        preds = model(masked_input)
        weights.append(preds[0, class_index].numpy())

    weights = np.maximum(np.array(weights, dtype=np.float32), 0.0)
    weights = weights / (np.sum(weights) + 1e-8)
    cam = tf.reduce_sum(upsampled_maps * weights[None, None, :], axis=-1)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.squeeze(cam).numpy()


def combined_explainability_heatmap(model, image, class_index, conv_layer_name="conv2d_4"):
    """Combine Grad-CAM++ and Score-CAM heatmaps into a single explainability map."""
    gradcam = gradcam_plus_plus_heatmap(model, image, class_index, conv_layer_name)
    scorecam = scorecam_heatmap(model, image, class_index, conv_layer_name)
    combined = (gradcam + scorecam) / 2.0
    return np.clip(combined, 0.0, 1.0)


def generate_explainability_maps(model, X_test, test_prob, sample_count=3, conv_layer_name="conv2d_4"):
    """Generate combined explainability heatmaps for a few test samples."""
    explain_maps = {}
    for idx in range(min(sample_count, len(X_test))):
        image = np.expand_dims(X_test[idx], axis=0)
        class_idx = int(np.argmax(test_prob[idx]))
        try:
            explain_maps[f"sample_{idx}"] = combined_explainability_heatmap(model, image, class_idx, conv_layer_name)
        except Exception as exc:
            explain_maps[f"sample_{idx}"] = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
    return explain_maps


def reshape_to_pseudo_image(X, target_size=128):
    """Pad and reshape the feature matrix into a pseudo-image tensor of size 128x128x1."""
    X_arr = X.astype(np.float32).values
    total_pixels = target_size * target_size
    if X_arr.shape[1] < total_pixels:
        pad_width = total_pixels - X_arr.shape[1]
        X_arr = np.pad(X_arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    else:
        X_arr = X_arr[:, :total_pixels]
    return X_arr.reshape(-1, target_size, target_size, 1)


def select_features(X, max_features=16384):
    """Reduce the feature space to a manageable size for training."""
    if X.shape[1] <= max_features:
        return X
    rng = np.random.RandomState(42)
    selected_idx = rng.choice(X.shape[1], size=max_features, replace=False)
    selected_idx = np.sort(selected_idx)
    return X.iloc[:, selected_idx]


class EpochProgressCallback(tf.keras.callbacks.Callback):

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.spinner = ["|", "/", "-", "\\"]

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        print(f"\rTraining epoch {(epoch + 1) * 10}/{self.total_epochs * 10} {self.spinner[epoch % len(self.spinner)]}", end="", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        print(f"\rEpoch {(epoch + 1) * 10}/{self.total_epochs * 10} complete | loss={loss:.4f} | val_loss={val_loss:.4f} | time={elapsed:.1f}s", flush=True)
        
def build_model(input_shape):
    tf.keras.backend.clear_session()
    model = km.Sequential(
        [
            kl.Input(shape=input_shape),
            kl.Conv2D(32, 3, activation="relu", padding="same", name="conv2d_1"),
            kl.MaxPooling2D(2),
            kl.Conv2D(64, 3, activation="relu", padding="same", name="conv2d_2"),
            kl.MaxPooling2D(2),
            kl.Conv2D(128, 3, activation="relu", padding="same", name="conv2d_3"),
            kl.MaxPooling2D(2),
            kl.Conv2D(256, 3, activation="relu", padding="same", name="conv2d_4"),
            kl.MaxPooling2D(2),
            kl.Flatten(),
            kl.Dense(256, activation="relu"),
            kl.Reshape((16, 16)),
            kl.LSTM(128, return_sequences=True),
            kl.LSTM(64),
            kl.Dense(128, activation="relu"),
            kl.Dropout(0.3),
            kl.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(X, y, epochs=10, batch_size=32):
    """Train the CNN-LSTM model and return predictions, metrics, and history."""
    print("Step 5: Reshaping features into a pseudo-image and training the CNN-LSTM model...")
    X_img = reshape_to_pseudo_image(X)
    y_binary = y.astype(int).values
    y_cat = to_categorical(y_binary, num_classes=2)

    train_idx, temp_idx = train_test_split(np.arange(len(y_binary)), test_size=0.30, random_state=42, stratify=y_binary)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y_binary[temp_idx])

    X_train, X_val, X_test = X_img[train_idx], X_img[val_idx], X_img[test_idx]
    y_train, y_val, y_test = y_cat[train_idx], y_cat[val_idx], y_cat[test_idx]

    class0_count = int(np.sum(y_train[:, 0]))
    class1_count = int(np.sum(y_train[:, 1]))
    class_weight = {0: 1.0, 1: max(1.0, class0_count / max(class1_count, 1))}
    model = build_model(input_shape=(128, 128, 1))

    early_stop = kc.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    progress_callback = EpochProgressCallback(total_epochs=epochs)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, progress_callback],
        class_weight=class_weight,
        verbose=0,
    )
    print()

    test_prob = model.predict(X_test, verbose=0)
    test_pred = np.argmax(test_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    tn, fp, fn, tp = confusion_matrix(y_true, test_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, test_pred)),
        "precision": float(precision_score(y_true, test_pred, zero_division=0)),
        "recall": float(recall_score(y_true, test_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, test_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, test_pred)),
        "auc": float(roc_auc_score(y_true, test_prob[:, 1])),
        "confusion_matrix": confusion_matrix(y_true, test_pred).tolist(),
    }

    return model, metrics, history.history, X_test, y_test, test_pred, test_prob


def read_metrics_file(output_dir):
    metrics_path = output_dir / "metrics"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as fp:
        return fp.read()


def save_outputs(model, metrics, history, y_test, test_pred, test_prob, X_test):
    print("Step 7: final results...")
    print_metrics = read_metrics_file(OUTPUT_DIR)
    print(print_metrics)

    print("Step 8: generating explainability heatmaps...")
    explain_maps = generate_explainability_maps(model, X_test, test_prob, sample_count=3)
    explain_path = OUTPUT_DIR / "explainability_heatmaps.npz"
    np.savez(explain_path, **explain_maps)
    print(f"Saved explainability heatmaps for {len(explain_maps)} test samples to {explain_path}")


def main():
    """Run the full preprocessing, training, evaluation, and explainability workflow."""
    parser = argparse.ArgumentParser(description="Train the CNN-LSTM intrusion detection pipeline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=16384)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    print("Starting the CNN-LSTM intrusion detection workflow...")
    ehr = load_ehr_data()
    mimic_profiles = load_mimic_profiles()
    combined = combine_features(ehr, mimic_profiles)

    if args.sample_size is not None:
        print(f"Sampling down to {args.sample_size} rows for a faster run...")
        combined = combined.sample(n=min(args.sample_size, len(combined)), random_state=42)

    print("Step 4: Preparing features and applying preprocessing...")
    X = combined.drop(columns=["Anomaly"])
    y = combined["Anomaly"].astype(int)

    X = bootstrap_multiple_linear_imputation(X, numeric_columns=X.select_dtypes(include=[np.number]).columns.tolist())
    X = apply_zscore_scaling(X)
    X = select_features(X, max_features=args.max_features)

    model, metrics, history, X_test, y_test, test_pred, test_prob = train_model(X, y, epochs=args.epochs, batch_size=args.batch_size)
    save_outputs(model, metrics, history, y_test, test_pred, test_prob, X_test)


if __name__ == "__main__":
    main()
    import graph
