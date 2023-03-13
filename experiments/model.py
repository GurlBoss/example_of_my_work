from torch import nn
import torch
import os


input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, future_look = 0):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.fl = future_look

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def save_model(self,version):
        parent_dir = os.path.abspath(os.path.dirname(__file__))
        model_dir = parent_dir+"/output/weights"
        weights_name = "lstm_" + version + '.pth'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.state_dict(), model_dir + "/" + weights_name)

    def load_model(self,version):
        parent_dir = os.path.abspath(os.path.dirname(__file__))
        model_dir = parent_dir + "/output/weights/"
        weights_name =  version
        weights_path = model_dir + weights_name
        try:
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict)
        except Warning:
            print("Weights to the model does not exist")

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().type(torch.DoubleTensor)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().type(torch.DoubleTensor)


        if x.is_cuda:
            h0 = h0.to("cuda:0")
            c0 = c0.to("cuda:0")
        outputs= []
        for i in range(self.fl):
            out, (hn, cn) = self.lstm(x, (h0, c0))
            out = self.fc(out[:,-1,:])
            outputs.append(out)
            out = out.unsqueeze(dim=2)
            x = torch.cat((x, out), dim=1)
            #print(out.size())
        # out.size() --> 100, 10
        out = torch.cat((outputs), dim = 1)
        out = out.unsqueeze(dim = 2)
        return out

class LSTM2(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM2, self).__init__()
        self.hidden_layers = hidden_layers
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, x, future_preds=0):
        outputs, num_samples = [], x.size(0)
        h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float64)
        c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float64)
        h_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float64)
        c_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float64)


        for time_step in x.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(time_step, (h_t, c_t))  # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))  # new hidden and cell states
            output = self.linear(h_t2)  # output from the last FC layer
            outputs.append(output)


        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor
        outputs = torch.cat(outputs, dim=1)
        return outputs
    def save_model(self,version):
        parent_dir = os.path.abspath(os.path.dirname(__file__))
        model_dir = parent_dir+"/output/weights"
        weights_name = "lstm_" + version + '.pth'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.state_dict(), model_dir + "/" + weights_name)

    def load_model(self,version):
        parent_dir = os.path.abspath(os.path.dirname(__file__))
        model_dir = parent_dir + "/output/weights/"
        weights_name =  version
        weights_path = model_dir + weights_name
        try:
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict)
        except Warning:
            print("Weights to the model does not exist")