import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

def scale_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    logger.info("Feature scaling: Fit on train, transform on train and test...")

    byte_columns = [f'byte{i}' for i in range(1, 9)]

    bytes_scaler = MinMaxScaler()
    time_scaler = MinMaxScaler()

    df_train[byte_columns] = bytes_scaler.fit_transform(df_train[byte_columns])
    df_train[['delta_t']] = time_scaler.fit_transform(df_train[['delta_t']])

    df_val[byte_columns] = bytes_scaler.transform(df_val[byte_columns])
    df_val[['delta_t']] = time_scaler.transform(df_val[['delta_t']])

    logger.success("Scaling done!")
    return df_train, df_val, bytes_scaler, time_scaler