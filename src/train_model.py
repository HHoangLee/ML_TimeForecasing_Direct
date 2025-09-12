import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_timeseries_data(data, window_size, target_size):
    i = 1
    while i < window_size:
        data[f"co2_{i}"] = data["co2"].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data[f"target_{i}"] = data["co2"].shift(-i - window_size)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data


data = pd.read_csv("data/co2.csv")
data["time"] = pd.to_datetime(data["time"], yearfirst=True)
data["co2"] = data["co2"].interpolate()

window_size = 10
target_size = 5
train_ratio = 0.8

data = create_timeseries_data(data, window_size, target_size)
targets = [f"target_{i}" for i in range(target_size)]
x = data.drop(["time"] + targets, axis=1)
y = data[targets]

num_samples = len(x)
x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

models = [LinearRegression() for _ in range(target_size)]


def root_mean_squared_error(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


for idx, model in enumerate(models):
    model.fit(x_train, y_train[f"target_{idx}"])
    y_predict = model.predict(x_test)
    print(f"Model {idx + 1}")
    print(f"MAE: {mean_absolute_error(y_test[f'target_{idx}'], y_predict)}")
    print(f"MSE: {mean_squared_error(y_test[f'target_{idx}'], y_predict)}")
    print(f"RMSE: {root_mean_squared_error(y_test[f'target_{idx}'], y_predict)}")
    print(f"R2: {r2_score(y_test[f'target_{idx}'], y_predict)}")
