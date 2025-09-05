import time
import numpy as np  # type: ignore
from sklearn.model_selection import KFold, cross_validate  # type: ignore
from xgboost import XGBRegressor # type: ignore
from catboost import CatBoostRegressor  # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.layers import Dense, Input, concatenate # type: ignore
from tensorflow.keras.models import Model # type: ignore
from .data_loader import get_regularizer, get_optimizer


def model_dl(input_shape, layers, activation, regularizer, optimizer):
    inputs = Input(shape=(input_shape,))
    x = inputs
    for unit in layers:
        x = Dense(unit, activation=activation, kernel_regularizer=regularizer)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model

def model_wdnn(input_dim, layer_units, activation, regularizer, optimizer):
    wide_input = Input(shape=(input_dim,))
    wide = Dense(1)(wide_input)

    deep_input = Input(shape=(input_dim,))
    x = deep_input
    for units in layer_units:
        x = Dense(units, activation=activation, kernel_regularizer=regularizer)(x)
    deep = x

    merged = concatenate([wide,deep])
    output = Dense(1, activation='linear')(merged)

    model = Model(inputs=[wide_input,deep_input], outputs=output)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model

def model_xgboost(eta, n_estimators, max_depth, reg_lambda, reg_alpha, colsample_bytree):
    model = XGBRegressor(objective ='reg:squarederror', learning_rate=eta, max_depth=max_depth,
                         reg_lambda=reg_lambda, reg_alpha=reg_alpha, n_estimators=n_estimators,
                         colsample_bytree=colsample_bytree, eval_metric='mae', seed = 123)
    return model

def model_catboost(iterations, depth, learning_rate):
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function='MAE',
        eval_metric='MAE',
        random_state=42,
        verbose=0)

    return model

def run_dl(X_train, X_test, y_train, y_test, layer_units, activation, regularizer_name,
              optimizer_name, learning_rate, val_split, batch_size, epochs, patience):

    optimizer = get_optimizer(optimizer_name, learning_rate)
    regularizer = get_regularizer(regularizer_name)

    model = model_dl(X_train.shape[1], layer_units, activation, regularizer, optimizer)
    # Start training
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=val_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
        verbose=0
    )
    end_time = time.time()  # End training
    y_pred = model.predict(X_test)

    elapsed_time = end_time - start_time
   
    return history, f"{elapsed_time} s", y_test, y_pred, model

def run_wdnn(X_train, X_test, y_train, y_test, layer_units, activation, regularizer_name,
              optimizer_name, learning_rate, val_split, batch_size, epochs, patience):

    optimizer = get_optimizer(optimizer_name, learning_rate)
    regularizer = get_regularizer(regularizer_name)

    model = model_wdnn(X_train.shape[1], layer_units, activation, regularizer, optimizer)
    # Start training
    start_time = time.time()
    history = model.fit(
        [X_train,X_train], y_train,
        validation_split=val_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
        verbose=0
    )
    end_time = time.time()  # End training
    y_pred = model.predict([X_test, X_test])

    elapsed_time = end_time - start_time
   
    return history, f"{elapsed_time} s", y_test, y_pred, model

def cv_xgboost(X_train, y_train, n_splits, eta, n_estimators,
               max_depth, reg_lambda, reg_alpha, colsample_bytree):

    # train-test split
    # X_train, X_test, y_train, y_test = preprocess(csv_path, feature_columns, target_column, scaler_name)

    model = model_xgboost(eta, n_estimators, max_depth, reg_lambda, reg_alpha, colsample_bytree)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_validate(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, return_train_score=True)
    mae_scores = -np.mean(cv_scores['test_score'])

    return mae_scores

def run_xgboost(X_train, X_test, y_train, y_test, eta, n_estimators,
               max_depth, lmbda, alpha, colsample_bytree):

    # train-test split
    # X_train, X_test, y_train, y_test = preprocess(csv_path, feature_columns, target_column, scaler_name)

    model = model_xgboost(eta, n_estimators, max_depth, lmbda, alpha, colsample_bytree)
    # Start training
    start_time = time.time()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=0)
    end_time = time.time()  # End training

    y_pred = model.predict(X_test).reshape(-1, 1)

    history = None  # XGBoost does not return a history object like Keras models
    xgboost_model = model

    elapsed_time = end_time - start_time

    return history, f"{elapsed_time} s", y_test, y_pred, xgboost_model

def cv_catboost(X_train, y_train, n_splits, iterations, depth, learning_rate):

    # train-test split
    # X_train, X_test, y_train, y_test = preprocess(csv_path, feature_columns, target_column, scaler_name)

    model = model_catboost(iterations, depth, learning_rate)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_validate(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, return_train_score=True)
    mae_scores = -np.mean(cv_scores['test_score'])

    return mae_scores

def run_catboost(X_train, X_test, y_train, y_test, iterations, depth, learning_rate):

    # train-test split
    # X_train, X_test, y_train, y_test = preprocess(csv_path, feature_columns, target_column, scaler_name)

    model = model_catboost(iterations, depth, learning_rate)
    # Start training
    start_time = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        verbose=0)
    end_time = time.time()  # End training

    y_pred = model.predict(X_test).reshape(-1, 1)

    history = None  # CatBoost does not return a history object like Keras
    catboost_model = model

    elapsed_time = end_time - start_time

    return history, f"{elapsed_time} s", y_test, y_pred, catboost_model
