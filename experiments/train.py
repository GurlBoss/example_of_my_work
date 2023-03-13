import model as md
import dataset
import torch
import datetime
from Utils import Utils
import matplotlib.pyplot as plt
import os
import numpy as np
from time import gmtime, strftime

CUDA_DEVICE = ""


def train():
    # get main word and datapath
    data_path = "data/Energy/Energy_trn.csv"
    time_now = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

    # load the settings
    settings_path = "settings.txt"
    settings = Utils.load_settings(settings_path)
    try:
        batch_size = settings['batch_size']
        epochs = settings['epochs']
        future_look = settings['future_look']
        look_back = settings['look_back']
        input_dim = settings['input_dim']
        hidden_dim = settings['hidden_dim']
        num_layers = settings['num_layers']
        output_dim = settings['output_dim']
        learning_rate = settings['learning_rate']
    except Exception as e:
        print("Something went wrong while loading the settings. Maybe some value is missing?")
        print("The reason is", e)
    print("Setting loaded")

    # set the device
    device = torch.device("cuda" + str(CUDA_DEVICE) if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # create datasets
    training_dataset = dataset.myDataset(data_path, look_back=look_back, future_look=future_look)
    validation_dataset = dataset.myDataset(data_path, look_back=look_back, future_look=future_look, train=False)

    # define model
    model = md.LSTM(input_dim, hidden_dim, num_layers, output_dim, future_look=future_look)
    model = model.double()
    model.to(device)
    loss_fc = torch.nn.MSELoss()

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=6)
    print("Dataset loaded")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("starting training")
    for epoch in range(epochs):
        loss_val = 0
        loss_train = 0

        # enumerate training dataset
        model.train()
        for i, data in enumerate(train_loader):
            input_series = data['x_series'].to(device)
            label = data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_series)

            loss = loss_fc(outputs, label)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch % 4 == 0:
            model.eval()
            for i, data in enumerate(val_loader):
                input_series = data['x_series'].to(device)
                label = data['label'].to(device)

                with torch.no_grad():
                    output = model(input_series)

                loss = loss_fc(outputs, label)
                loss_val += loss.item()
                if i < 5:
                    visualise_val(label,output,input_series,i,epoch)
            ep_trn_mse_loss = loss_train / len(train_loader)
            ep_val_mse_loss = loss_val / len(val_loader)
            print(f"e: {epoch:2d}   Trn_MSE: {ep_trn_mse_loss:.6f}   Val_MSE: {ep_val_mse_loss:.6f} ")

        model.save_model(time_now)


def visualise_val(y, y_pred, x, i, e):
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    graph_dir = parent_dir + "/output/graphs/val_data"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    y = y.cpu().detach().numpy().squeeze()
    y_pred = y_pred.cpu().detach().numpy().squeeze()
    x = x.cpu().detach().numpy().squeeze()
    xpoints_ori = [i for i in range(len(y) + len(x))]
    xpoints_pred = [i  for i in range(len(y_pred))]
    plt.clf()
    full_data = np.concatenate((x,y))
    plt.plot(xpoints_ori, full_data)
    plt.plot(xpoints_pred, y_pred)
    total_path = graph_dir + f"/{i}_{e}.png"
    plt.ylim(0.2, 0.6)
    plt.tight_layout()
    plt.savefig(total_path, format="png", dpi=300)


if __name__ == '__main__':
    train()
