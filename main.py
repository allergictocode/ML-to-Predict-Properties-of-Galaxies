import yaml # type: ignore
import argparse
from src.tuning import run_tuning
from src.training import run_training

def main():
    parser = argparse.ArgumentParser(description="Pipeline for Galactic Properties Prediction.")
    parser.add_argument("action", choices=['tune', 'train'], 
                        help="Action done: 'tune' for hyperparameter tuning or 'train' for training final model.")
    parser.add_argument("config", type=str, 
                        help="Path to YAML configuration file (example: configs/magphys_config.yaml).")
    parser.add_argument("model", nargs='+', choices=['xgboost', 'catboost', 'ann', 'wdnn'],
                        help="Model(s) to be used in tuning or training. You can specify multiple models separated by space.")
    
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update config with selected model
    config['models'] = args.model

    # Run chosen action
    if args.action == 'tune':
        print(f"\n\n--- Starting Hyperparameter Tuning for: {config['project_name']} using {config['models']} ---")
        run_tuning(config, args.config)
    elif args.action == 'train':
        print(f"\n\n--- Starting Model Training for: {config['project_name']} using {config['models']} ---")
        run_training(config)


if __name__ == "__main__":
    main()


# python main.py tune configs/magphys_config.yaml model(s)
# python main.py train configs/mpa_jhu_config.yaml model(s)