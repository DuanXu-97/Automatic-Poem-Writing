import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from torchvision.models import densenet


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, path):
        t.save(self.state_dict(), path)


class AttLSTM(BasicModule):

    def __init__(self, config):
        super(AttLSTM, self).__init__()

        self.config = config
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.hidden_dim, num_layers=2, batch_first=False, dropout=self.config.dropout_rate)
        self.fc = nn.Linear(self.config.hidden_dim, self.config.vocab_size)

    def attention(self, lstm_output, final_state):
        print(lstm_output.size())
        print(final_state.size())

        # [seq_len, batch, hidden_size]->[batch, seq_len, hidden_size]
        lstm_output = lstm_output.permute(1, 0, 2)
        batch_size, seq_len, hidden_size = lstm_output.size()
        print(lstm_output.size())

        # [num_layers, batch, hidden_size]->[batch, hidden_size]
        merged_state = t.mean(final_state, 0)
        print(merged_state.size())

        # [batch, hidden_size]->[batch, hidden_size, 1]
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        print(merged_state.size())

        ogema = nn.Parameter(t.zeros(batch_size, hidden_size, hidden_size), requires_grad=True)
        # [batch, hidden_size, hidden_size], [batch, hidden_size, 1] -> [batch, hidden_size, 1]
        weights = t.bmm(ogema, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        print('att_weights', weights.size())
        # [batch, seq_len, hidden_size] * [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]
        out = lstm_output.mul(weights.transpose(1, 2).repeat(1, seq_len, 1))
        return out

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()
        print(x.size())
        if hidden is None:
            h_0 = x.data.new(2, batch_size, self.config.hidden_dim).fill_(0).float()
            c_0 = x.data.new(2, batch_size, self.config.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden

        x = self.embeddings(x)
        lstm_output, lstm_hidden = self.lstm(x, (h_0, c_0))
        x = self.attention(lstm_output, lstm_hidden[0])
        print(x.size())

        output = self.fc(x.reshape(seq_len*batch_size, -1))
        return output, lstm_hidden

