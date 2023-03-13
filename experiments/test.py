
from datetime import datetime, timedelta
import model, dataset
import torch
import pandas as pd
import os
import numpy as np
from time import gmtime, strftime
import csv

CUDA_DEVICE = ""

parent_dir = os.path.abspath(os.path.dirname(__file__))
results_dir = parent_dir + "/output/predictions"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


if __name__ == '__main__':
    data_path = "data/Energy/Energy_tst.csv"
    batch_size = 32
    weights_name = "lstm_03_13_2023_16_56_58.pth"
    time_now = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

    device = torch.device("cuda" + str(CUDA_DEVICE) if torch.cuda.is_available() else "cpu")

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1

    model = model.LSTM(input_dim, hidden_dim, num_layers, output_dim,future_look = 15)
    model = model.double()
    model.load_model(weights_name)
    model.to(device)

    stock_df = pd.read_csv(data_path)
    lookback = 70
    beg = 0
    first_sample = stock_df[beg:beg+lookback]
    first_sample = np.array(first_sample)
    first_sample_tickers = first_sample[...,1].astype(float)

    dates = first_sample[...,0].astype(str)
    start_date_str = dates[-1]
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    date = start_date + timedelta(days=1)


    model.eval()
    sample = torch.from_numpy(np.expand_dims(first_sample_tickers, axis=(0,-1)))

    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates ]
    pred_tickers = first_sample_tickers.tolist()

    input_series = sample.to(device)
    outputs = model(input_series)

    y_pred = outputs.cpu().detach().numpy().squeeze()
    for elem in y_pred:
        dates.append(date)
        date += timedelta(days=1)
        pred_tickers.append(elem)

    dates = [tmp_date.strftime('%Y-%m-%d') for tmp_date in dates]
    df = pd.DataFrame(list(zip(dates, pred_tickers)), columns=['Date', 'AVG'])

    df.to_csv(results_dir + "/pred_y_" + time_now + '.csv', index=True, sep=',')



