import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from torch import nn
import torch.utils.data as tdata
import utils
import model as mod
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence

# Delcare Gs for import data
Gs_static = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
Gs_dynamic = ['pin', 'rotate', 'touch', 'swipe_left', 'swipe_right', 'swipe_up', 'swipe_down']
Gs_actual = Gs_dynamic  # change for your actual gestures
# Path to the dataset default set to
# /home/<user>/<workspace>/src/mirracle_gestures/include/data/learning/
PATH = os.path.abspath(os.path.dirname(__file__)) + "/Dataset/learning/"
work_dir = os.path.abspath(os.path.dirname(__file__))
print_batch = False
save_best_model = True
enough_acc = 0.95
print_epochs = True


class gesture():
    def __init__(self, name):
        self.name = name
        self.gest_list = None
    def get_len(self):
        return len(self.gest_list)


class Dataset(tdata.Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data
        self.longest = self.get_longest(data)

    def __len__(self):
        return len(self.data)

    def get_longest(self,data):
        shapes = [a.shape[0] for a in data]
        return max(shapes)

    def __getitem__(self, item):
        tmp_data = self.data[item]
        tmp_label = self.labels[item]
        sample = {"data": tmp_data, "label": tmp_label}
        # TODO: ASK FOR TRANSPOSE?
        # sample = {"data": tmp_data.transpose((1, 0)), "label": tmp_label}
        return sample


def get_prediction_order(prediction, label):
    prediction = prediction.detach()  # detach from computational graph (no grad)
    label = label.detach()

    prediction_sorted = torch.argsort(prediction, 1, True)
    finder = (
            label[:, None] == prediction_sorted
    )  # None as an index creates new dimension of size 1, so that broadcasting works as expected
    order = torch.nonzero(finder)[:, 1]  # returns a tensor of indices, where finder is True.

    return order

def my_key(data):
    return data

def collate_batch(batch):
    label_list, data_list, = [], []
    lengths = []

    for sample in batch:
        label_list.append(sample['label'])
        processed_data = torch.tensor(sample['data'], dtype=torch.double)
        data_list.append(processed_data)
        lengths.append(processed_data.size()[0])

    label_list = torch.tensor(label_list, dtype=torch.int64)
    data_list = pad_sequence(data_list, batch_first=True, padding_value=0)
    data_list = pack_padded_sequence(data_list,batch_first=True,lengths=lengths,enforce_sorted=False)
    return {"data": data_list, "label": label_list}

def get_pred_label(prediction):
    prediction = prediction.detach()
    prediction_sorted = torch.argsort(prediction, 1, True)
    label = prediction_sorted[0, 0].tolist()
    return label


def list_of_gestures(data, labels):
    num_of_Gs = int(max(labels) + 1)
    tmp_list = []
    for i in range(num_of_Gs):
        tmp_gs = gesture(Gs_actual[i])
        tmp_gs.gest_list = [data[j, :, :] for j in range(len(labels)) if labels[j] == i]
        tmp_list.append(tmp_gs)
    return tmp_list


def plot_train(x1, x2, x3, x4):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x2)
    axs[0].set_title("Loss train per epoch")
    axs[0].set_ylim([1, 2.3])
    axs[0].set_xlim([0, 50])
    axs[0].set(xlabel='epoch', ylabel='Cross Ent. loss')
    axs[1].plot(x4)
    axs[1].set_title("Loss validation per epoch")
    axs[1].set_ylim([1, 2.3])
    axs[1].set_xlim([0, 50])
    axs[1].set(xlabel='epoch', ylabel='Cross Ent. loss')
    plt.tight_layout()


def train(trn_data, trn_label, val_data, val_label, b_size=20,
          epochs=10, my_model=None, model_type="lstm", learning_rate=0.0001,
          num_layers = 1,hidden_size = 100,word = ''):
    name = word
    # create dataset
    val_dataset = Dataset(val_data, val_label)
    trn_dataset = Dataset(trn_data, trn_label)

    if my_model is None:
        num_classes = int(max(trn_dataset.labels) + 1)
        last_dim = len(trn_dataset.data[0].shape) - 1
        input_size = int(trn_dataset.data[0].shape[last_dim])
        if model_type == "lstm":
            my_model = mod.LSTM(num_classes, input_size, hidden_size, num_layers)
        else:
            my_model = mod.GRU(num_classes, input_size, hidden_size, num_layers)
        my_model = my_model.double()

    device_id = torch.cuda.device_count()
    device = "cuda"
    my_model = my_model.to(device)

    trn_loader = tdata.DataLoader(trn_dataset, batch_size=b_size, shuffle=True, num_workers=2,collate_fn=collate_batch)
    val_loader = tdata.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2,collate_fn=collate_batch)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss(reduction="mean") # mean-squared error for regression
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=1e-6)

    trn_loss = []
    val_loss = []
    trn_loss_per_epoch = []
    val_loss_per_epoch = []
    save_acc = 0
    loss_control = 20000
    for epoch in range(epochs):

        trn_total = 0
        my_model = my_model.train()
        for i, batch in enumerate(trn_loader):
            input = batch['data'].to(device)
            labels = batch['label'].to(device)

            outputs = my_model(input)

            optimizer.zero_grad()

            outputs = outputs.to(torch.float)
            labels = labels.to(torch.int64)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            trn_loss.append(loss.item())
            trn_total += loss.item()

            if i % 1 == 0 and print_batch:
                print('Epoch %3d id: %3d/%3d Training '
                      'loss: %.3f' % (epoch + 1,i,len(trn_loader), loss.item()))
        trn_loss_per_epoch.append(trn_total / len(trn_loader))

        val_total = 0
        my_model = my_model.eval()
        pred_lab = []

        for i, batch in enumerate(val_loader):
            input = batch['data'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = my_model(input)
            loss = criterion(outputs, labels)

            # print statistics
            val_loss.append(loss.item())
            val_total += loss.item()
            pred_lab.append(get_pred_label(outputs))

            # if epoch % 5 == 0:
            # print('Epoch %3d Validation loss: %.3f' % (epoch + 1, loss.item()))

        val_loss_per_epoch.append(val_total / len(val_loader))
        acc = accuracy_score(val_label, pred_lab)
        if acc > save_acc:
            save_acc = acc
            if save_best_model:
                my_model.save_model(name)
        if save_acc > enough_acc:
            break

        if epoch % 5 and print_epochs:
            print("Epoch: %3d Trn loss per Ep: %.3f Val loss per Ep: %.3f Val accuracy: %.3f" % (
                epoch+1, trn_total / len(trn_loader), val_total / len(val_loader), acc))
    print("Final acc = %.3f" % save_acc)
    plot_train(trn_loss, trn_loss_per_epoch, val_loss, val_loss_per_epoch)
    return save_acc

def test(test_data, test_label, my_model=None, model_type="lstm",
         hidden_size = 100,num_layers = 1,word = ''):
    test_dataset = Dataset(test_data, test_label)
    name = word
    if my_model is None:
        num_classes = int(max(test_dataset.labels) + 1)
        last_dim = len(test_dataset.data[0].shape) - 1
        input_size = int(test_dataset.data[0].shape[last_dim])
        if model_type == "lstm":
            my_model = mod.LSTM(num_classes, input_size, hidden_size, num_layers)
        else:
            my_model = mod.GRU(num_classes, input_size, hidden_size, num_layers)
        my_model = my_model.double()
    my_model.load_model(name)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Running on GPU")
    else:
        device = torch.device('cpu')
        print("Running on CPU")

    my_model = my_model.to(device)
    my_model = my_model.eval()
    test_loader = tdata.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,collate_fn=collate_batch)

    orders = []
    pred_lab = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input = batch['data'].to(device)
            labels = batch['label'].to(device)

            prediction = my_model(input)
            order = get_prediction_order(prediction, labels).cpu().numpy()
            orders.append(order)
            pred_lab.append(get_pred_label(prediction))

    orders = np.concatenate(orders, 0)
    TOP1 = (orders == 0).mean()
    TOP1 = round(TOP1 * 100)
    return TOP1, pred_lab


if __name__ == "__main__":
    with open(work_dir + '/rocords.npy', 'rb') as f:
        rec = np.load(f)
    f.close()
    with open(work_dir + '/xpalm.npy', 'rb') as f:
        a = np.load(f)
    f.close()
    val_data, val_rec, train_data, train_rec = utils.create_val_data(a, rec, c=9)

    # set parameters
    epochs = 50
    batch = 60
    lr = 0.01
    model_t = "lstm"

    train(train_data, train_rec, val_data, val_rec, epochs=epochs, b_size=batch, learning_rate=lr,model_type=model_t)
    _, pred_labels = test(val_data, val_rec,model_type=model_t)
    acc = accuracy_score(val_rec, pred_labels)
    plt.suptitle('LSTM')
    plt.tight_layout()
    plt.savefig(work_dir + '/LSTM.png', format='png')

    model_t = "gru"
    train(train_data, train_rec, val_data, val_rec, epochs=epochs, b_size=batch, learning_rate=lr,model_type=model_t)
    _, pred_labels = test(val_data, val_rec,model_type=model_t)
    acc = accuracy_score(val_rec, pred_labels)
    plt.suptitle('GRU')
    plt.tight_layout()
    plt.savefig(work_dir + '/GRU.png', format='png')

    # order=test(val_dataset)
