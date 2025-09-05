import os
from datetime import datetime
import joblib  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score  # type: ignore
from scipy.stats import gaussian_kde    # type: ignore
from matplotlib.colors import Normalize # type: ignore


def plot_learning_curve(history, model_select, model, target_column, output_path):
    if history is not None:
        plt.plot(history.history['mae'], label='Train loss')
        plt.plot(history.history['val_mae'], label='Val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    elif model_select == "xgboost":
        results = model.evals_result()
        epochs = len(results['validation_0']['mae'])
        x_axis = range(0, epochs)
        plt.plot(x_axis, results['validation_0']['mae'], label='Train MAE')
        plt.plot(x_axis, results['validation_1']['mae'], label='Validation MAE')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
    elif model_select == "catboost":
        evals_result = model.get_evals_result()
        train_loss = evals_result['learn']['MAE']
        test_loss = evals_result['validation']['MAE']
        iterations = np.arange(1, len(train_loss) + 1)
        plt.plot(iterations, train_loss, label='Train MAE')
        plt.plot(iterations, test_loss, label='Validation MAE')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
    else:
        raise ValueError("Invalid model type for plotting history.")
    plt.title(f'{target_column} {model_select} Learning Curve')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(output_path)
    plt.close()

def plot_prediction(df, label, y_test, y_pred, model_select, target_name, output_path):
    true = y_test.values.flatten()
    pred = y_pred[:, 0]
    residuals = pred - true

    rmse, nrmse, mae, bias, sigma, eta, r2 = evaluation_metrics(true, pred)

    # Density calculation
    xy = np.vstack([true, pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = true[idx], pred[idx], z[idx]
    res = residuals[idx]

    # Plot setup
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})

    # Normalize color scale
    norm = Normalize(vmin=z.min(), vmax=z.max())

    # Main scatter plot
    sc_main = ax_main.scatter(x, y, c=z, cmap='viridis', s=8, norm=norm, alpha=0.8)
    # Residual scatter plot
    sc_res = ax_res.scatter(x, res, c=z, cmap='viridis', s=8, norm=norm, alpha=0.8)

    # Garis identitas y = x
    lims = [
        np.min([ax_main.get_xlim(), ax_main.get_ylim()]),
        np.max([ax_main.get_xlim(), ax_main.get_ylim()])
    ]
    ax_main.plot(lims, lims, color='orange', linewidth=1, label=rf"$45\degree$ line")  # Garis y = x
    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)

    # Best fit lines
    p_main = np.poly1d(np.polyfit(x, y, 1))
    ax_main.plot(lims, p_main(lims), linestyle='dashed', color='red', lw=1, label='Best fit line')
    p_res = np.poly1d(np.polyfit(x, res, 1))
    ax_res.plot(lims, p_res(lims), linestyle='dashed', color='red', lw=1, label='Best fit line')
    ax_res.axhline(0, color='orange', lw=1)

    # Shaded 3σ region
    _3sigma = 3 * sigma
    x_unique = np.unique(x)
    ax_main.fill_between(x_unique, x_unique + _3sigma, x_unique - _3sigma, color='pink', alpha=0.2)

    # IQR lines parallel to y=x
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    median = np.median(y)
    offset_q1 = q1 - median
    offset_q3 = q3 - median
    ax_main.plot(lims, [l + offset_q1 for l in lims], color='blue', linestyle='--', linewidth=1, label='Q1 (IQR)')
    ax_main.plot(lims, [l + offset_q3 for l in lims], color='green', linestyle='--', linewidth=1, label='Q3 (IQR)')
    ax_main.legend(loc='upper right', fontsize=9)

    # Color bar
    fig.colorbar(sc_main, ax=[ax_main, ax_res], location='right', pad=0.05, label='Density')

    # Labels 
    if model_select == "deep_learning":
        model = "Deep learning"
    elif model_select == "wdnn":
        model = "WDNN"
    elif model_select == "xgboost":
        model = "XGBoost"
    elif model_select == "catboost":
        model = "CatBoost"
    else:
        return ValueError(f"Invalid model select: {model_select}")
    
    if target_name == "SFR_0_1Gyr_percentile50":
        target = "SFR"
        ax_main.set_ylabel(rf"{target} {model} $\log({target})$ $(M_{{\odot}}\,/\,\mathrm{{yr}})$")
        ax_res.set_ylabel(r"Residual $\log(\mathrm{SFR})$ $(M_{\odot}\,/\,\mathrm{yr})$")
        ax_res.set_xlabel(rf"{target} {label} $\log({target})$ $(M_{{\odot}}\,/\,\mathrm{{yr}})$")
    elif target_name == "mass_stellar_percentile50":
        target = "SM"
        ax_main.set_ylabel(rf"{target} {model} $\log(M_{{*}}[M_{{\odot}}])$")
        ax_res.set_ylabel(r"Residual $\log(M_{*}[M_{\odot}])$")
        ax_res.set_xlabel(rf"{target} {label} $\log(M_{{*}}[M_{{\odot}}])$")
    elif target_name == "SFR_TOT_P50":
        target = "SFR"
        ax_main.set_ylabel(rf"{target} {model} $\log({target})$ $(M_{{\odot}}\,/\,\mathrm{{yr}})$")
        ax_res.set_ylabel(r"Residual $\log(\mathrm{SFR})$ $(M_{\odot}\,/\,\mathrm{yr})$")
        ax_res.set_xlabel(rf"{target} {label} $\log({target})$ $(M_{{\odot}}\,/\,\mathrm{{yr}})$")
    elif target_name == "LGM_TOT_P50":
        target = "SM"
        ax_main.set_ylabel(rf"{target} {model} $\log(M_{{*}}[M_{{\odot}}])$")
        ax_res.set_ylabel(r"Residual $\log(M_{*}[M_{\odot}])$")
        ax_res.set_xlabel(rf"{target} {label} $\log(M_{{*}}[M_{{\odot}}])$")
    else:
        return ValueError(f"Invalid target name: {target_name}")

    # Style tweaks
    for ax in [ax_main, ax_res]:
        ax.grid(True, linestyle=':', linewidth=0.5)

    # Std and Eta
    ax_main.text(0.02, 0.95, rf"$\sigma$: {sigma:.2f}, $\eta$: {eta:.2f}",
             transform=ax_main.transAxes, fontsize=12, fontstyle='italic')
    
    plt.savefig(output_path)
    plt.close()

def evaluation_metrics(y_true, y_pred):
    residuals = y_pred - y_true
    rmse = root_mean_squared_error(y_true, y_pred)
    nrmse = rmse / (np.max(y_true) - np.min(y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(residuals)
    sigma = np.std(residuals, axis=0)
    catastrophic_outliers = np.sum(np.abs(residuals) > 3 * sigma)
    eta = (catastrophic_outliers / len(residuals))
    r2 = r2_score(y_true, y_pred)

    return rmse, nrmse, mae, bias, sigma, eta, r2

def save_results(df, label, history, training_time, y_true, y_pred, target_column, scaler_cls, model_select, model,
                 layer_units, activation, regularizer, optimizer, learning_rate, val_split, batch_size, epochs, patience,
                 xg_eta, n_estimators, max_depth, reg_lambda, reg_alpha, colsample_bytree,
                 iterations, depth, cat_eta):
    output_folder = create_output_folder(df)

    if model_select in ["xgboost", "catboost"]:
        joblib.dump(model, output_folder + f'/{model_select}_model_{target_column}.pkl')
    elif model_select in ["dl", "wdnn"]:
        model.save(output_folder + f'/{model_select}_model_{target_column}.h5')
    
    plot_learning_curve(history, model_select, model, target_column, output_folder + f'/{model_select}_learning curve_{target_column}.png')
    plot_prediction(df, label, y_true, y_pred, model_select, target_column, output_folder + f'/{model_select}_prediction_result_{target_column}.png')

    rmse, nrmse, mae, bias, sigma, eta, r2 = evaluation_metrics(y_true, y_pred)

    results = ''
    results += f'Target        : {target_column}\n'
    results += f'Training time : {training_time}\n\n'
    results += '--- Model & Training Parameters ---\n'
    results += f'Scaler        : {scaler_cls}\n'
    results += f'Model         : {model_select}\n'
    if model_select in ["deep_learning", "wdnn"]:
        results += f'Layer units   : {layer_units}\n'
        results += f'Activation    : {activation}\n'
        results += f'Regularizer   : {regularizer}\n'
        results += f'Optimizer     : {optimizer}\n'
        results += f'Learning rate : {learning_rate}\n'
        results += f'Val split     : {val_split}\n'
        results += f'Batch size    : {batch_size}\n'
        results += f'Epochs        : {epochs}\n'
        results += f'Patience      : {patience}\n\n'
    elif model_select == "xgboost":
        results += f'Number of estimators : {n_estimators}\n'
        results += f'Max depth            : {max_depth}\n'
        results += f'Learning rate        : {xg_eta}\n'
        results += f'L2 regularization    : {reg_lambda}\n'
        results += f'L1 regularization    : {reg_alpha}\n'
        results += f'Colsample by tree    : {colsample_bytree}\n\n'
    elif model_select == "catboost":
        results += f'Iterations             : {iterations}\n'
        results += f'Depth                  : {depth}\n'
        results += f'Learning rate          : {cat_eta}\n\n'
    else:
        raise ValueError("Invalid model selection. Choose from 'deep_learning', 'wdnn', 'xgboost', or 'catboost'.")
    results += '--- Evaluation Metrics ---\n'
    results += f'RMSE          : {rmse}\n'
    results += f'NRMSE         : {nrmse}\n'
    results += f'MAE           : {mae}\n'
    results += f'Bias          : {bias}\n'
    results += f'Sigma         : {sigma}\n'
    results += f'Eta           : {eta}\n'
    results += f'R2            : {r2}\n'

    with open(output_folder + f'/{model_select}_evaluation_metrics_{target_column}', 'w') as file:
        file.write(results)

def create_output_folder(df):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')   #-%S
    output_folder = f'./{df}/{timestamp}'
    os.makedirs(output_folder, exist_ok=True)
    return output_folder