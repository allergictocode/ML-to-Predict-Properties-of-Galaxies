import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras import regularizers   #type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop, SGD   # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN    # type: ignore
from tensorflow.keras.layers import Dense, Input, concatenate   # type: ignore
from tensorflow.keras.models import Model   # type: ignore
TF_ENABLE_ONEDNN_OPTS = 0

# ==============================================================================
# 1. FUNGSI PRA-PEMROSESAN DAN UTILITAS DATA
# ==============================================================================

def load_and_clean_data(csv_path):
    """Memuat data dari CSV, menghapus nilai NaN dan duplikat."""
    df = pd.read_csv(csv_path)
    df = df.dropna().drop_duplicates()
    print(f"\nUkuran berkas setelah dibersihkan: {df.shape}")
    return df

def preprocess(csv_path, feature_columns, target_column, scaler_name):
    """
    Fungsi utama untuk memuat, membersihkan, menskalakan, dan membagi data.
    """
    df = load_and_clean_data(csv_path)

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    scaler = scalers.get(scaler_name)
    if not scaler:
        raise ValueError("Nama scaler tidak valid. Pilih dari: 'StandardScaler', 'MinMaxScaler', 'RobustScaler'")

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# ==============================================================================
# 2. FUNGSI PEMBUATAN DAN PELATIHAN MODEL
# ==============================================================================

# --- Deep Learning (DL) ---

def model_dl(input_shape, layers, activation, regularizer, optimizer_name, learning_rate):
    """Membangun arsitektur model Deep Learning Keras."""
    reg = None
    if regularizer == 'l1':
        reg = regularizers.l1(0.01)
    elif regularizer == 'l2':
        reg = regularizers.l2(0.01)

    inputs = Input(shape=(input_shape,))
    x = inputs
    for units in layers:
        x = Dense(units, activation=activation, kernel_regularizer=reg)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer_map = {
        'adam': Adam,
        'rmsprop': RMSprop,
        'sgd': SGD
    }
    # Buat objek optimizer dengan learning rate
    optimizer_instance = optimizer_map.get(optimizer_name)(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss='mean_absolute_error', metrics=['mae']) # Gunakan metrics mae

    return model

def run_dl(X_train, X_test, y_train, y_test, validation_split, layers, activation, regularizer, optimizer, learning_rate, batch_size, epochs):
    """Melatih dan mengevaluasi model Deep Learning."""
    model = model_dl(X_train.shape[1], layers, activation, regularizer, optimizer, learning_rate)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        TerminateOnNaN()
    ]
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    
    y_pred = model.predict(X_test)
    elapsed_time = f"{end_time - start_time:.2f} s"
    
    return history, elapsed_time, y_test, y_pred.flatten(), model

# --- Wide & Deep Neural Network (WDNN) ---

def model_wdnn(input_shape, layers, activation, regularizer, optimizer_name, learning_rate):
    """Membangun arsitektur model WDNN Keras."""
    reg = None
    if regularizer == 'l1':
        reg = regularizers.l1(0.01)
    elif regularizer == 'l2':
        reg = regularizers.l2(0.01)

    inputs = Input(shape=(input_shape,))
    deep_path = inputs
    for units in layers:
        deep_path = Dense(units, activation=activation, kernel_regularizer=reg)(deep_path)
    
    # Wide path terhubung langsung ke output
    wide_path = inputs 
    
    combined = concatenate([wide_path, deep_path])
    outputs = Dense(1)(combined)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer_map = {
        'adam': Adam,
        'rmsprop': RMSprop,
        'sgd': SGD
    }
    optimizer_instance = optimizer_map.get(optimizer_name)(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss='mean_absolute_error', metrics=['mae'])
    
    return model

def run_wdnn(X_train, X_test, y_train, y_test, validation_split, layers, activation, regularizer, optimizer, learning_rate, batch_size, epochs):
    """Melatih dan mengevaluasi model WDNN."""
    model = model_wdnn(X_train.shape[1], layers, activation, regularizer, optimizer, learning_rate)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        TerminateOnNaN()
    ]
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    
    y_pred = model.predict(X_test)
    elapsed_time = f"{end_time - start_time:.2f} s"
    
    return history, elapsed_time, y_test, y_pred.flatten(), model


# --- XGBoost ---

def cv_xgboost(X_train, y_train, n_splits, **params):
    """Menjalankan cross-validation untuk XGBoost (digunakan oleh hyperparam_tune.py)."""
    model = XGBRegressor(random_state=42, **params)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_validate(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv)
    return -np.mean(cv_scores['test_score'])

def run_xgboost(X_train, X_test, y_train, y_test, **params):
    """Melatih dan mengevaluasi model XGBoost."""
    model = XGBRegressor(random_state=42, eval_metric='mae', **params)
    
    # Tambahkan eval_set agar riwayat training bisa di-plot
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    start_time = time.time()
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    end_time = time.time()
    
    y_pred = model.predict(X_test)
    elapsed_time = f"{end_time - start_time:.2f} s"
    
    # Objek history tidak lagi dibutuhkan, kita akan teruskan None
    return None, elapsed_time, y_test, y_pred, model


# --- CatBoost ---

def cv_catboost(X_train, y_train, n_splits, **params):
    """Menjalankan cross-validation untuk CatBoost (digunakan oleh hyperparam_tune.py)."""
    model = CatBoostRegressor(random_state=42, verbose=0, **params)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_validate(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv)
    return -np.mean(cv_scores['test_score'])

def run_catboost(X_train, X_test, y_train, y_test, **params):
    """Melatih dan mengevaluasi model CatBoost."""
    model = CatBoostRegressor(random_state=42, verbose=0, eval_metric='MAE', **params)
    
    # Tambahkan eval_set untuk merekam riwayat
    eval_set = [(X_test, y_test)]
    
    start_time = time.time()
    model.fit(X_train, y_train, eval_set=eval_set)
    end_time = time.time()
    
    y_pred = model.predict(X_test)
    elapsed_time = f"{end_time - start_time:.2f} s"

    # Objek history tidak lagi dibutuhkan, kita akan teruskan None
    return None, elapsed_time, y_test, y_pred, model


# ==============================================================================
# 3. FUNGSI PENYIMPANAN HASIL DAN VISUALISASI
# ==============================================================================

# --- Evaluation Metrics ---

def evaluation_metrics(true, pred):
    """Fungsi placeholder untuk metrik evaluasi."""
    rmse = root_mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    # Placeholder untuk metrik lainnya
    nrmse = rmse / (true.max() - true.min())
    bias = np.mean(pred - true)
    sigma = np.std(pred - true)
    eta = np.median(np.abs(pred - true))
    return rmse, nrmse, mae, bias, sigma, eta, r2

# --- Learning Curve ---

def plot_learning_curve(history, model_select, model, target_column, output_path):
    plt.figure(figsize=(10, 6))
    if history is not None and hasattr(history, 'history'):
        # Logika untuk Keras/TensorFlow
        # Pastikan metrik yang benar ada (misalnya 'mae' dan 'val_mae')
        if 'mae' in history.history and 'val_mae' in history.history:
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
        else: # Fallback ke 'loss' jika 'mae' tidak ada
            plt.plot(history.history['loss'], label='Train MAE')
            plt.plot(history.history['val_loss'], label='Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
    elif model_select == "xgboost":
        results = model.evals_result()
        epochs = len(results['validation_0']['mae'])
        x_axis = range(0, epochs)
        plt.plot(x_axis, results['validation_0']['mae'], label='Train MAE')
        plt.plot(x_axis, results['validation_1']['mae'], label='Validation MAE')
        plt.xlabel('Iteration')
        plt.ylabel('MAE')
    elif model_select == "catboost":
        evals_result = model.get_evals_result()
        train_loss = evals_result['learn']['MAE']
        test_loss = evals_result['validation']['MAE']
        iterations = np.arange(1, len(train_loss) + 1)
        plt.plot(iterations, train_loss, label='Train MAE')
        plt.plot(iterations, test_loss, label='Validation MAE')
        plt.xlabel('Iteration')
        plt.ylabel('MAE')
    else:
        print(f"Learning curve plot is not available for model type '{model_select}'.")
        plt.close()
        return # Keluar dari fungsi jika tidak bisa di-plot
    
    plt.title(f'{target_column} {model_select.upper()} Learning Curve')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(output_path)
    plt.close()

# --- True Value vs. Predicted Value ---

def plot_prediction(df_name, y_test, y_pred, model_select, target_name, plot_config, output_path):
    true = y_test if isinstance(y_test, np.ndarray) else y_test.values
    true = true.flatten()
    pred = y_pred.flatten()

    residuals = pred - true
    rmse, nrmse, mae, bias, sigma, eta, r2 = evaluation_metrics(true, pred)

    xy = np.vstack([true, pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z, res = true[idx], pred[idx], z[idx], residuals[idx]

    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    norm = Normalize(vmin=z.min(), vmax=z.max())
    sc_main = ax_main.scatter(x, y, c=z, cmap='viridis', s=8, norm=norm, alpha=0.8)
    sc_res = ax_res.scatter(x, res, c=z, cmap='viridis', s=8, norm=norm, alpha=0.8)

    lims = [np.min([ax_main.get_xlim(), ax_main.get_ylim()]), np.max([ax_main.get_xlim(), ax_main.get_ylim()])]
    ax_main.plot(lims, lims, color='orange', linewidth=1, label=r'$45\degree$ line')
    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)

    p_main = np.poly1d(np.polyfit(x, y, 1))
    ax_main.plot(lims, p_main(lims), linestyle='dashed', color='red', lw=1, label='Best fit line')
    p_res = np.poly1d(np.polyfit(x, res, 1))
    ax_res.plot(lims, p_res(lims), linestyle='dashed', color='red', lw=1)
    ax_res.axhline(0, color='orange', lw=1)
    
    _3sigma = 3 * sigma
    x_unique = np.linspace(lims[0], lims[1], 100)
    ax_main.fill_between(x_unique, x_unique + _3sigma, x_unique - _3sigma, color='pink', alpha=0.2, label='3σ Region')
    ax_main.legend(loc='upper left', fontsize=9)
    fig.colorbar(sc_main, ax=[ax_main, ax_res], location='right', pad=0.05, label='Density')

    model_display_name = model_select.upper()
    
    # Ambil label dari plot_config, gunakan target_name sebagai fallback
    target_label = plot_config.get('display_unit', target_name)
    
    ax_main.set_ylabel(f"Predicted {target_label}")
    ax_res.set_ylabel(f"Residual")
    ax_res.set_xlabel(f"True {target_label}")
    
    ax_main.text(0.02, 0.95, rf"$\sigma$: {sigma:.2f}, $\eta$: {eta:.2f}", transform=ax_main.transAxes, fontsize=12, va='top')
    
    for ax in [ax_main, ax_res]:
        ax.grid(True, linestyle=':', linewidth=0.5)

    fig.suptitle(f'Prediction vs. Actual for {model_display_name} on {df_name.upper()}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()

# --- Save All Plots and Evaluation Metrics ---

def save_results(dataset_name, history, training_time, y_test, y_pred, target_column, scaler_name, model_name, model, plot_config, **kwargs):
    """
    Menyimpan semua hasil: metrik, plot, dan informasi model ke dalam direktori.
    """
    # Membuat direktori output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/output/{dataset_name}/{target_column}/{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Hitung Metrik Evaluasi
    rmse, nrmse, mae, bias, sigma, eta, r2 = evaluation_metrics(y_test, y_pred)
    
    # 2. Simpan Metrik ke Berkas Teks
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write(f"--- Model Training Summary ---\n")
        f.write(f"Dataset        : {dataset_name}\n")
        f.write(f"Target Variable: {target_column}\n")
        f.write(f"Model          : {model_name.upper()}\n")
        f.write(f"Scaler         : {scaler_name}\n")
        f.write(f"Training Time  : {training_time}\n\n")
        f.write("--- Evaluation Metrics ---\n")
        f.write(f"MAE   : {mae:.4f}\n")
        f.write(f"RMSE  : {rmse:.4f}\n")
        f.write(f"NRMSE : {nrmse:.4f}\n")
        f.write(f"Bias  : {bias:.4f}\n")
        f.write(f"Sigma : {sigma:.4f}\n")
        f.write(f"Eta   : {eta:.4f}\n")
        f.write(f"R²    : {r2:.4f}\n\n")
        f.write("--- Hyperparameters ---\n")
        for key, value in kwargs.items():
            f.write(f"  - {key}: {value}\n")

    # 3. Panggil fungsi plot baru
    
    # Plot Kurva Pembelajaran
    lc_path = f"{output_dir}/learning_curve.png"
    plot_learning_curve(history, model_name, model, target_column, lc_path)

    # Plot Prediksi vs Aktual
    pv_path = f"{output_dir}/prediction_vs_actual.png"
    if y_pred.ndim == 1:
        y_pred_reshaped = y_pred.reshape(-1, 1)
    else:
        y_pred_reshaped = y_pred

    # Teruskan plot_config ke fungsi plot_prediction
    plot_prediction(dataset_name, y_test, y_pred_reshaped, model_name, target_column, plot_config, pv_path)
    
    print(f"Results saved in directory: {output_dir}")