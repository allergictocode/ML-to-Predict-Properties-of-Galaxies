# ML Models for Predicting Galaxy Properties

A machine learning pipeline for predicting galactic properties using various models including XGBoost, CatBoost, Artificial Neural Networks (ANN), and Wide & Deep Neural Networks (WDNN).

## Overview

This project implements machine learning models to predict galactic properties from two different datasets:

- **MAGPHYS Dataset**: 21 Panchromatic GAMA Dataset containing flux measurements across multiple wavelengths
- **MPA-JHU Dataset**: SDSS+AllWISE dataset with photometric measurements

## Features

- Multiple ML model implementations:
  - XGBoost
  - CatBoost  
  - Artificial Neural Networks (ANN)
  - Wide & Deep Neural Networks (WDNN)
- Automated hyperparameter tuning using Optuna
- Model training pipeline
- Performance visualization and metrics tracking
- YAML-based configuration system

## Project Structure

```
├── configs/                  # Configuration files
│   ├── magphys_config.yaml  # MAGPHYS dataset config
│   └── mpa_jhu_config.yaml  # MPA-JHU dataset config
├── data/
│   ├── input/               # Input datasets
│   └── output/              # Model outputs and results
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── models.py           # ML model implementations
│   ├── training.py         # Training pipeline
│   ├── tuning.py          # Hyperparameter tuning
│   └── utils.py           # Helper functions
├── main.py                 # Main execution script
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone this repository
2. Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

The pipeline can be run in two modes:

1. Hyperparameter tuning:
```sh
python main.py tune configs/magphys_config.yaml [model(s)]
```

2. Model training:
```sh
python main.py train configs/mpa_jhu_config.yaml [model(s)]
```

Where `[model(s)]` can be one or more of: `xgboost`, `catboost`, `ann`, `wdnn`

## Configuration

The project uses YAML configuration files to specify:
- Dataset paths and properties
- Feature and target columns
- Model parameters
- Training parameters
- Hyperparameter tuning settings

Example configs are provided in the `configs/` directory.

## Output

The pipeline generates:
- Trained model files
- Learning curves
- Prediction visualizations
- Detailed evaluation metrics including:
  - RMSE (Root Mean Square Error)
  - NRMSE (Normalized RMSE)
  - MAE (Mean Absolute Error)
  - R² score
  - Additional statistical metrics

Results are saved in timestamped directories under `data/output/`.

## Dependencies

Main dependencies include:
- TensorFlow
- XGBoost
- CatBoost
- scikit-learn
- Optuna
- NumPy
- Pandas
- Matplotlib

See `requirements.txt` for complete list and versions.