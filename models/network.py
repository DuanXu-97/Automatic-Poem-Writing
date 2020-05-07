import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from torchvision.models import densenet


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path, use_gpu=False):
        if use_gpu:
            self.load_state_dict(t.load(path))
        else:
            self.load_state_dict(t.load(path, map_location=t.device('cpu')))


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
        # [seq_len, batch, hidden_size]->[batch, seq_len, hidden_size]
        lstm_output = lstm_output.permute(1, 0, 2)
        batch_size, seq_len, hidden_size = lstm_output.size()

        # [num_layers, batch, hidden_size]->[batch, hidden_size]
        merged_state = t.mean(final_state, 0)

        # [batch, hidden_size]->[batch, hidden_size, 1]
        merged_state = merged_state.unsqueeze(2)

        ogema = nn.Parameter(t.zeros(batch_size, hidden_size, hidden_size), requires_grad=True)
        if self.config.use_gpu:
            ogema = ogema.cuda()
        # [batch, hidden_size, hidden_size], [batch, hidden_size, 1] -> [batch, hidden_size, 1]
        weights = t.bmm(ogema, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        # [batch, seq_len, hidden_size] * [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]
        out = lstm_output.mul(weights.transpose(1, 2).repeat(1, seq_len, 1))
        return out

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()
        if hidden is None:
            h_0 = x.data.new(2, batch_size, self.config.hidden_dim).fill_(0).float()
            c_0 = x.data.new(2, batch_size, self.config.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden

        x = self.embeddings(x)
        lstm_output, lstm_hidden = self.lstm(x, (h_0, c_0))

        x = self.attention(lstm_output, lstm_hidden[0])

        output = self.fc(x.reshape(seq_len*batch_size, -1))
        return output, lstm_hidden


class LSTM(BasicModule):

    def __init__(self, config):
        super(LSTM, self).__init__()

        self.config = config
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.hidden_dim, num_layers=self.config.num_layers, batch_first=False)
        self.fc = nn.Linear(self.config.hidden_dim, self.config.vocab_size)

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()
        if hidden is None:
            h_0 = x.data.new(2, batch_size, self.config.hidden_dim).fill_(0).float()
            c_0 = x.data.new(2, batch_size, self.config.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden

        x = self.embeddings(x)
        x, hidden = self.lstm(x, (h_0, c_0))
        output = self.fc(x.reshape(seq_len*batch_size, -1))
        return output, hidden


class GRU(BasicModule):

    def __init__(self, config):
        super(GRU, self).__init__()

        self.config = config
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.gru = nn.GRU(self.config.embedding_dim, self.config.hidden_dim, num_layers=self.config.num_layers, batch_first=False)
        self.fc = nn.Linear(self.config.hidden_dim, self.config.vocab_size)

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()
        if hidden is None:
            h_0 = x.data.new(2, batch_size, self.config.hidden_dim).fill_(0).float()
            h_0 = Variable(h_0)
        else:
            h_0 = hidden

        x = self.embeddings(x)
        x, hidden = self.gru(x, h_0)
        output = self.fc(x.reshape(seq_len*batch_size, -1))
        return output, hidden


class BiGRU(BasicModule):

    def __init__(self, config):
        super(BiGRU, self).__init__()

        self.config = config
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.gru = nn.GRU(self.config.embedding_dim, self.config.hidden_dim, num_layers=self.config.num_layers, batch_first=False, bidirectional=True)
        self.fc1 = nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2)
        self.fc2 = nn.Linear(self.config.hidden_dim // 2, self.config.vocab_size)

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()
        if hidden is None:
            h_0 = x.data.new(4, batch_size, self.config.hidden_dim).fill_(0).float()
            h_0 = Variable(h_0)
        else:
            h_0_backward = x.data.new(self.config.num_layers, 1, batch_size, self.config.hidden_dim).fill_(0).float()
            h_0_forward = t.index_select(hidden.reshape(self.config.num_layers, 2, batch_size, self.config.hidden_dim), dim=1, index=t.LongTensor([0]))
            h_0 = t.cat([h_0_forward, h_0_backward], dim=1).reshape(self.config.num_layers*2, batch_size, self.config.hidden_dim)

        x = self.embeddings(x)
        x, hidden = self.gru(x, h_0)
        x = t.mean(x.reshape(seq_len*batch_size, 2, self.config.hidden_dim), dim=1)
        x = self.fc1(x)
        output = self.fc2(x)

        return output, hidden


class ShortcutGRU(BasicModule):

    def __init__(self, config):
        super(ShortcutGRU, self).__init__()

        self.config = config
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.gru1 = nn.GRU(self.config.embedding_dim, self.config.hidden_dim_1, num_layers=self.config.num_layers, batch_first=False)
        self.gru2 = nn.GRU(self.config.embedding_dim + self.config.hidden_dim_1, self.config.hidden_dim_2, num_layers=self.config.num_layers, batch_first=False)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(self.config.hidden_dim_2, self.config.vocab_size)

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()
        if hidden is None:
            h1_0 = x.data.new(2, batch_size, self.config.hidden_dim_1).fill_(0).float()
            h2_0 = x.data.new(2, batch_size, self.config.hidden_dim_2).fill_(0).float()
            h1_0, h2_0 = Variable(h1_0), Variable(h2_0)
        else:
            h1_0, h2_0 = hidden

        x = self.embeddings(x)
        index_1 = t.LongTensor([0, 1]).cuda() if self.config.use_gpu else t.LongTensor([0, 1])
        gru1_output, gru1_hidden = self.gru1(x, t.index_select(h1_0, dim=0, index=index_1))
        x = t.cat([x, gru1_output], dim=-1)
        x = self.dropout(x)
        index_2 = t.LongTensor([2, 3]).cuda() if self.config.use_gpu else t.LongTensor([0, 1])
        x, gru2_hidden = self.gru2(x, t.index_select(h2_0, dim=0, index=index_2))
        x = self.dropout(x)
        output = self.fc(x)
        hidden = (gru1_hidden, gru2_hidden)

        return output, hidden


