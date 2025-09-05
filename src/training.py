from .data_loader import preprocess
from .models import run_dl, run_wdnn, run_xgboost, run_catboost
from .utils import save_results


def run_training(config):
    """
    Run training pipeline for every target and chosen model in config.
    """
    feature_columns = config['feature_columns']
    scaler_cls = config['scaler_cls']
    models = config['models']
    df_name = config['output_dir']
    label = config['label']

    for target in config['target_columns']:
        print(f"\n===== Processing Target: {target} =====")
        
        # 1. Preprocessing Data
        X_train, X_test, y_train, y_test = preprocess(
            config['csv_path'], feature_columns, target, scaler_cls
        )
        
        for model_select in models:
            print(f"\n--- Training Model: {model_select} ---")
            
            # 2. Take Hyperparameters
            try:
                params = config['targets'][target][model_select]
            except KeyError:
                print(f"Warning: Hyperparameter for {model_select} on target {target} not found. Skipping...")
                continue

            # 3. Train Model
            history = training_time = y_pred = model = None
            # Model-specific hyperparameters
            layer_units = activation = regularizer = optimizer = learning_rate = val_split = batch_size = epochs = patience = None
            xg_eta = n_estimators = max_depth = reg_lambda = reg_alpha = colsample_bytree = None
            iterations = depth = cat_eta = None

            if model_select == "xgboost":
                xg_eta = params['eta']
                n_estimators = params['n_estimators']
                max_depth = params['max_depth']
                reg_lambda = params['reg_lambda']
                reg_alpha = params['reg_alpha']
                colsample_bytree = params['colsample_bytree']
                history, training_time, y_true, y_pred, model = run_xgboost(
                    X_train, X_test, y_train, y_test,
                    eta=xg_eta,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    lmbda=reg_lambda,
                    alpha=reg_alpha,
                    colsample_bytree=colsample_bytree,
                )
            
            elif model_select == "catboost":
                iterations = params['iterations']
                depth = params['depth']
                cat_eta = params['learning_rate']
                history, training_time, y_true, y_pred, model = run_catboost(
                    X_train, X_test, y_train, y_test,
                    iterations=iterations,
                    depth=depth,
                    learning_rate=cat_eta,
                )

            elif model_select == "ann":
                layer_units = params['layers']
                activation = params['activation']
                regularizer = params.get('regularizer', None)
                optimizer = params['optimizer']
                learning_rate = params['learning_rate']
                val_split = params['val_split']
                batch_size = params['batch_size']
                epochs = params['epochs']
                patience = params['patience']
                history, training_time, y_true, y_pred, model = run_dl(
                    X_train, X_test, y_train, y_test,
                    layers=layer_units,
                    activation=activation,
                    regularizer=regularizer,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    val_split=val_split,
                    batch_size=batch_size,
                    epochs=epochs,
                    patience=patience                    
                )

            elif model_select == "wdnn":
                layer_units = params['layers']
                activation = params['activation']
                regularizer = params.get('regularizer', None)
                optimizer = params['optimizer']
                learning_rate = params['learning_rate']
                val_split = params['val_split']
                batch_size = params['batch_size']
                epochs = params['epochs']
                patience = params['patience']
                history, training_time, y_true, y_pred, model = run_wdnn(
                    X_train, X_test, y_train, y_test,
                    layers=layer_units,
                    activation=activation,
                    regularizer=regularizer,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    val_split=val_split,
                    batch_size=batch_size,
                    epochs=epochs,
                    patience=patience    
                )

            # 4. Save Results for each model and target
            save_results(
                df_name, label, history, training_time, y_true, y_pred, target, scaler_cls, model_select, model,
                layer_units, activation, regularizer, optimizer, learning_rate, val_split, batch_size, epochs, patience,
                xg_eta, n_estimators, max_depth, reg_lambda, reg_alpha, colsample_bytree,
                iterations, depth, cat_eta
            )

            print(f"\nTraining {model_select} model for target {target} done.\n")

