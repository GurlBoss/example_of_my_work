from torch import nn
import torch
import os
DEVICE = 'cuda:0'
from torch.nn.utils.rnn import pad_packed_sequence
#LSTM inspired by
#https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=_BcDEjcABRVz
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.sig = torch.nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # Propagate input through LSTM


        out, (hn, cn) = self.lstm(x)
        #out = self.soft(out)
        output, input_sizes = pad_packed_sequence(out, batch_first=True)
        out = output[:, -1, :]

        out = self.fc(out)
        out = self.soft(out)
        return out

    def save_model(self,word):
        directory = os.path.abspath(os.path.dirname(__file__))
        torch.save(self.state_dict(), directory + '/output/models/weights_lstm'+word+'.pth')
        print("saved weights")
        print(directory + '/output/models/weights.pth')

    def load_model(self,word):
        directory = os.path.abspath(os.path.dirname(__file__))

        # The model should be trained in advance and in this function, you should instantiate model and load the weights into it:
        self.load_state_dict(torch.load(directory + '/output/models/weights_lstm'+word+'.pth', map_location='cpu'))

        # For more info on storing and loading weights, see https://pytorch.org/tutorials/beginner/saving_loading_models.html
        return self


class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.sig = torch.nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # Propagate input through LSTM


        out, hn = self.gru(x)
        output, input_sizes = pad_packed_sequence(out, batch_first=True)
        #out = self.soft(out)
        out = output[:, -1, :]

        out = self.fc(out)
        out = self.soft(out)
        return out

    def save_model(self,word):
        directory = os.path.abspath(os.path.dirname(__file__))
        torch.save(self.state_dict(), directory + '/output/models/weights_gru'+word+'.pth')
        print("saved weights")
        print(directory + '/output/models/weights_gru.pth')

    def load_model(self,word):
        directory = os.path.abspath(os.path.dirname(__file__))

        # The model should be trained in advance and in this function, you should instantiate model and load the weights into it:
        self.load_state_dict(torch.load(directory + '/output/models/weights_gru'+word+'.pth', map_location='cpu'))

        # For more info on storing and loading weights, see https://pytorch.org/tutorials/beginner/saving_loading_models.html
        return self