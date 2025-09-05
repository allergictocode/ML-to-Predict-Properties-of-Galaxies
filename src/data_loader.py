import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split    # type: ignore
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler    # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import regularizers # type: ignore


def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna().drop_duplicates()
    print("\nFile size:",df.shape)
    return df

def get_scaler(scaler_name):
    scalers = {
        'StandardScaler': StandardScaler,
        'MinMaxScaler': MinMaxScaler,
        'RobustScaler': RobustScaler
    }
    return scalers[scaler_name]()

def get_regularizer(name):
    return {
        'l1': regularizers.l1(1e-4),
        'l2': regularizers.l2(1e-4),
        'l1_l2': regularizers.l1_l2(1e-4, 1e-4),
        None: None
    }[name]

def get_optimizer(name, lr):
    return {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD,
        'rmsprop': tf.keras.optimizers.RMSprop
    }[name](lr)

def preprocess(csv_path, features, target, scaler_name):
    df = load_and_clean_data(csv_path)
    scaler = get_scaler(scaler_name)
    X = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    y = df[[target]]
    return train_test_split(X, y, test_size=0.2, random_state=42)