import pandas as pd
from loguru import logger
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def scale_features(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> tuple[
    DataFrame, DataFrame, DataFrame, MinMaxScaler, MinMaxScaler]:

    logger.info("Feature scaling...")

    byte_columns = [f'byte{i}' for i in range(1, 9)]

    bytes_scaler = MinMaxScaler()
    time_scaler = MinMaxScaler()

    df_train[byte_columns] = bytes_scaler.fit_transform(df_train[byte_columns])
    df_train[['delta_t']] = time_scaler.fit_transform(df_train[['delta_t']])

    df_val[byte_columns] = bytes_scaler.transform(df_val[byte_columns])
    df_val[['delta_t']] = time_scaler.transform(df_val[['delta_t']])

    df_test[byte_columns] = bytes_scaler.transform(df_test[byte_columns])
    df_test[['delta_t']] = time_scaler.transform(df_test[['delta_t']])

    logger.success("Scaling done!")
    return df_train, df_val, df_test, bytes_scaler, time_scaler