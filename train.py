import argparse
import yaml
import joblib  # Untuk menyimpan model scikit-learn
from models import preprocess, run_dl, run_wdnn, run_xgboost, run_catboost, save_results

def load_config(config_path="config.yaml"):
    """Memuat konfigurasi dari berkas YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    """Fungsi utama untuk melatih dan menyimpan model akhir."""

    # 1. Muat konfigurasi
    config = load_config(args.config_path)
    try:
        dataset_config = config['data'][args.dataset]
        target_config = dataset_config['targets'][args.target]
        model_params = target_config[args.model]
        plot_config = target_config.get('plotting', {})
    except KeyError:
        print(f"Error: Configuration for dataset '{args.dataset}', target '{args.target}', or model '{args.model}' not found.")
        return

    # 2. Persiapan data
    print(f"Processing data for dataset: {args.dataset}, target: {args.target}")
    X_train, X_test, y_train, y_test = preprocess(
        csv_path=dataset_config['path'],
        feature_columns=dataset_config['features'],
        target_column=args.target,
        scaler_name=config['scaler']
    )

    # 3. Latih model dengan hyperparameter terbaik
    print(f"\nTraining for model: {args.model.upper()}...")
    
    # Pilih fungsi pelatihan yang sesuai berdasarkan model
    training_function = {
        "dl": run_dl,
        "wdnn": run_wdnn,
        "xgboost": run_xgboost,
        "catboost": run_catboost
    }.get(args.model)

    if args.model in ["dl", "wdnn"]:
        model_params['validation_split'] = config.get('validation_split', 0.1)

    if not training_function:
        raise ValueError(f"Model '{args.model}' unsupported.")

    # Melatih model
    history, training_time, y_test, y_pred, trained_model = training_function(
        X_train, X_test, y_train, y_test, **model_params
    )

    print(f"Training done in {training_time}.")

    # 4. Simpan hasil dan model
    print("Saving results and trained model...")
    
    # Menyimpan plot dan metrik evaluasi (menggunakan fungsi Anda yang sudah ada)
    save_results(
        dataset_name=args.dataset.upper(),
        history=history,
        training_time=training_time,
        y_test=y_test,
        y_pred=y_pred,
        target_column=args.target,
        scaler_name=config['scaler'],
        model_name=args.model,
        model=trained_model,
        plot_config=plot_config,
        **model_params 
    )

    # Menyimpan objek model yang sudah dilatih
    model_filename = f"data/output/{args.dataset}/saved_models/{args.dataset}_{args.target}_{args.model}.pkl"
    if args.model in ["xgboost", "catboost"]:
        joblib.dump(trained_model, model_filename)
    elif args.model in ["dl", "wdnn"]:
        model_params['validation_split'] = config.get('val_split', 0.1)
        # Model Keras/TensorFlow disimpan dengan cara berbeda
        model_filename = f"data/output/{args.dataset}/saved_models/{args.dataset}_{args.target}_{args.model}.keras"
        trained_model.save(model_filename)

    print(f"Model saved in: {model_filename}")
    print("\n--- Training done ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model estimator training script")
    
    parser.add_argument("--dataset", type=str, required=True, choices=['magphys', 'mpa-jhu'],
                        help="Choose dataset to use.")
    parser.add_argument("--target", type=str, required=True,
                        help="Target column to predict.")
    parser.add_argument("--model", type=str, required=True, choices=['dl', 'wdnn', 'xgboost', 'catboost'],
                        help="Choose model to train.")
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to YAML configuration file.")
                        
    args = parser.parse_args()

    # Pastikan direktori untuk menyimpan model ada
    import os
    os.makedirs(f"data/output/{args.dataset}/saved_models", exist_ok=True)
    
    main(args)


# run train.py
# python train.py --dataset magphys --target SFR_0_1Gyr_percentile50 --model xgboost