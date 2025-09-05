import optuna   # type: ignore
import yaml # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from .data_loader import preprocess
from .models import run_dl, run_wdnn, cv_xgboost, cv_catboost


def run_tuning(config, config_path):
    """
    Run the hyperparameter tuning pipeline.
    """
    feature_columns = config['feature_columns']
    scaler_cls = config['scaler_cls']
    models_to_tune = config['models']
    
    for target in config['target_columns']:
        print(f"\n===== Starting Tuning for Target: {target} =====")
        X_train, X_test, y_train, y_test = preprocess(
            config['csv_path'], feature_columns, target, scaler_cls
        )
        
        # Objective function that accepts the model name as an argument
        def objective(trial, model_name):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'eta': trial.suggest_float('eta', 0.01, 0.3, step=0.001),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.01),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0, step=0.0001),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0, step=0.0001),
                }
                mae = cv_xgboost(X_train, y_train, n_splits=config['k_fold'], **params)

            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.001),
                }
                mae = cv_catboost(X_train, y_train, n_splits=config['k_fold'], **params)

            elif model_name == 'ann':
                params = {
                    'layer_units': [trial.suggest_int(f'layer_{i}', 16, 128, step=16) for i in range(trial.suggest_int('n_layers', 3, 5))],
                    'activation': trial.suggest_categorical('activation', ['relu']),
                    'regularizer_name': trial.suggest_categorical('regularizer', [None, 'l2', 'l1']),
                    'optimizer_name': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, step=1e-4),
                    'batch_size': trial.suggest_int('batch_size', 32, 256, step=32),
                    'epochs': trial.suggest_int('epochs', 100, 1000, step=100),
                }
                _, _, y_true, y_pred, _ = run_dl(
                    X_train, X_test, y_train, y_test, val_split=config['val_split'], patience=config['targets'][target][model_name], **params
                )
                mae = mean_absolute_error(y_true, y_pred)
            
            elif model_name == 'wdnn':
                params = {
                    'layer_units': [trial.suggest_int(f'layer_{i}', 16, 128, step=16) for i in range(trial.suggest_int('n_layers', 3, 5))],
                    'activation': trial.suggest_categorical('activation', ['relu']),
                    'regularizer_name': trial.suggest_categorical('regularizer', [None, 'l2', 'l1']),
                    'optimizer_name': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, step=1e-4),
                    'batch_size': trial.suggest_int('batch_size', 32, 256, step=32),
                    'epochs': trial.suggest_int('epochs', 100, 1000, step=100),
                }

                _, _, y_true, y_pred, _ = run_wdnn(
                    X_train, X_test, y_train, y_test, val_split=config['val_split'], patience=config['targets'][target][model_name], **params
                )
                mae = mean_absolute_error(y_true, y_pred)
            
            else:
                raise ValueError(f"Model {model_name} is not recognized.")

            return mae

        # Loop to run the study for each model
        for model_name in models_to_tune:
            print(f"\n--- Starting tuning for model: {model_name} ---")
            study_name = f"{config['project_name']}_{target}_{model_name}"
            study = optuna.create_study(
                direction="minimize",
                storage=config['tuning']['storage_db'],
                study_name=study_name,
                load_if_exists=True
            )
            
            # Using a lambda function to pass model_name to the objective
            study.optimize(lambda trial: objective(trial, model_name=model_name), 
                           n_trials=config['tuning']['n_trials'])
            
            print(f"\nTuning finished for {model_name}.")
            trial = study.best_trial
            print(f"  Best MAE: {trial.value}")
            print("  Best Hyperparameters:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

            # Saving the best parameters back to the config
            # Ensure the structure in the YAML already exists
            if target not in config['targets']:
                config['targets'][target] = {}
            if model_name not in config['targets'][target]:
                config['targets'][target][model_name] = {}
            
            # Using update() to save all parameters
            config['targets'][target][model_name].update(trial.params)

            with open(config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False, indent=2)

    print(f"\nAll tuning completed. Updated configuration file at: {config_path}.\n\n")
