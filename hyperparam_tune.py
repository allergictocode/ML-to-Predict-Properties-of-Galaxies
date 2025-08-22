import argparse
import yaml
import optuna
from models import preprocess, run_dl, run_wdnn, cv_xgboost, cv_catboost
from sklearn.metrics import mean_absolute_error

# --- Fungsi untuk memuat konfigurasi dari berkas YAML ---
def load_config(config_path="config.yaml"):
    """Creating configuration from YAML file..."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# --- Fungsi objective utama untuk Optuna ---
def objective(trial, model_name, X_train, y_train, X_test, y_test, validation_split, k_fold_splits):
    """
    Fungsi objective umum yang dipanggil oleh Optuna untuk setiap trial.
    Fungsi ini akan memilih dan melatih model berdasarkan 'model_name'.
    """
    if model_name == "dl":
        # Mendefinisikan ruang pencarian (search space) untuk Deep Learning
        params = {
            'layers': [trial.suggest_int(f'layer_{i}', 16, 128, step=16) for i in range(trial.suggest_int('n_layers', 3, 6))],
            'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'tanh']),
            'regularizer': trial.suggest_categorical('regularizer', [None, 'l1', 'l2']),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 100, 1000, step=100),
        }
        # Menjalankan model DL dan mendapatkan MAE
        history, _, _, y_pred, _ = run_dl(X_train, X_test, y_train, y_test, validation_split=validation_split, **params)
        return mean_absolute_error(y_test, y_pred)

    elif model_name == "wdnn":
        # Mendefinisikan ruang pencarian untuk Wide & Deep Neural Network
        params = {
            'layers': [trial.suggest_int(f'layer_{i}', 16, 128, step=16) for i in range(trial.suggest_int('n_layers', 3, 6))],
            'activation': trial.suggest_categorical('activation', ['relu', 'elu']),
            'regularizer': trial.suggest_categorical('regularizer', [None, 'l1', 'l2']),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 100, 1000, step=100),
        }
        # Menjalankan model WDNN dan mendapatkan MAE
        history, _, _, y_pred, _ = run_wdnn(X_train, X_test, y_train, y_test, validation_split=validation_split, **params)
        return mean_absolute_error(y_test, y_pred)

    elif model_name == "xgboost":
        # Mendefinisikan ruang pencarian untuk XGBoost
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('learning_rate', 0.01, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        # Menjalankan cross-validation untuk XGBoost dan mendapatkan MAE
        return cv_xgboost(X_train, y_train, n_splits=k_fold_splits, **params)

    elif model_name == "catboost":
        # Mendefinisikan ruang pencarian untuk CatBoost
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        }
        # Menjalankan cross-validation untuk CatBoost dan mendapatkan MAE
        return cv_catboost(X_train, y_train, n_splits=k_fold_splits, **params)

    else:
        raise ValueError(f"Model '{model_name}' unsupported.")

# --- Fungsi utama untuk menjalankan proses tuning ---
def main(args):
    """Fungsi utama yang mengatur seluruh proses hyperparameter tuning."""
    
    # 1. Muat konfigurasi
    config = load_config(args.config_path)
    try:
        # Coba akses konfigurasi dataset dan pastikan target ada di dalamnya
        dataset_config = config['data'][args.dataset]
        if args.target not in dataset_config['targets']:
            # Jika target tidak ada, picu KeyError secara manual
            raise KeyError(f"Target '{args.target}' not found for dataset '{args.dataset}'")

    except KeyError as e:
        # Tangkap error jika dataset atau target tidak ditemukan
        print(f"Error: Configuration invalid or not found in {args.config_path}.")
        print(f"Detail: Make sure dataset '{args.dataset}' and target '{args.target}' are defined correctly.")
        print(f"Error message: {e}")
        return # Hentikan eksekusi jika konfigurasi tidak ada

    # 2. Persiapan data
    print(f"Processing data for dataset: {args.dataset}, target: {args.target}")
    X_train, X_test, y_train, y_test = preprocess(
        csv_path=dataset_config['path'],
        feature_columns=dataset_config['features'],
        target_column=args.target,
        scaler_name=config['scaler']
    )

    # 3. Setup dan jalankan studi Optuna
    study_name = f"tuning_{args.dataset}_{args.target}_{args.model}"
    storage_name = f"sqlite:///{args.dataset}_tuning.db"
    
    study = optuna.create_study(
        direction="minimize",  # Kita ingin meminimalkan MAE
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )

    validation_split = config.get('validation_split', 0.1)  # Default 0.1 jika tidak ada di config
    k_fold_splits = config.get('k_fold', 5) # Default 5 jika tidak ada di config

    # Lambda digunakan untuk meneruskan argumen tambahan ke fungsi objective
    objective_func = lambda trial: objective(trial, args.model, X_train, y_train, X_test, y_test, validation_split, k_fold_splits)
    
    # Tentukan metode validasi berdasarkan jenis model
    validation_method = f"{validation_split} validation split" if args.model in ["dl", "wdnn"] else f"{k_fold_splits}-Fold CV"
    print(f"\nHyperparameter tuning for model: {args.model.upper()} with {validation_method}...")

    study.optimize(objective_func, n_trials=args.n_trials)

    # 4. Tampilkan hasil
    print("\n--- Hyperparameter tuning done ---")
    print(f"Study: {study.study_name}")
    print(f"Number of trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"\nBest MAE: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")
        
    # 5. Simpan hasil ke berkas
    output_filename = f"hyperparameter_tuning/{args.dataset}_{args.target}_{args.model}_best_params.txt"
    with open(output_filename, "w") as f:
        f.write(f"Best MAE: {best_trial.value}\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")
    print(f"\nBest hyperparameters saved in: {output_filename}\n\n")


if __name__ == "__main__":
    # Mengatur parser untuk argumen baris perintah
    parser = argparse.ArgumentParser(description="Hyperparameter tuning using Optuna")
    
    parser.add_argument("--dataset", type=str, required=True, choices=['magphys', 'mpa-jhu'], 
                        help="Choose dataset (example: magphys)")
    parser.add_argument("--target", type=str, required=True, 
                        help="Target column (example: SFR_TOT_P50)")
    parser.add_argument("--model", type=str, required=True, choices=['dl', 'wdnn', 'xgboost', 'catboost'],
                        help="Choose model to tune (example: xgboost)")
    parser.add_argument("--n_trials", type=int, default=50, 
                        help="Number of trials for hyperparameter tuning (default: 50)")
    parser.add_argument("--config_path", type=str, default="config.yaml", 
                        help="Path to YAML configuration file (default: config.yaml)")
                        
    args = parser.parse_args()
    main(args)


# run hyperparam_tune.py
# python hyperparam_tune.py --dataset magphys --target SFR_0_1Gyr_percentile50 --model xgboost --n_trials 75