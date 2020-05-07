
class AttLSTMConfig(object):
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10

    data_path = "../data/tang.npz"

    vocab_size = 0
    ix2word = None
    word2ix = None

    num_layers = 2
    embedding_dim = 128
    hidden_dim = 256
    dropout_rate = 0.1
    max_len = 125

    batch_size = 128
    epoch = 20
    lr = 0.001
    seed = 10


class LSTMConfig(object):
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10

    data_path = "../data/tang.npz"

    vocab_size = 0
    ix2word = None
    word2ix = None

    num_layers = 2
    embedding_dim = 128
    hidden_dim = 256
    dropout_rate = 0.1
    max_len = 125

    batch_size = 128
    epoch = 20
    lr = 0.001
    seed = 10


class GRUConfig(object):
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10

    data_path = "../data/tang.npz"

    vocab_size = 0
    ix2word = None
    word2ix = None

    num_layers = 2
    embedding_dim = 128
    hidden_dim = 256
    dropout_rate = 0.1
    max_len = 125

    batch_size = 128
    epoch = 20
    lr = 0.001
    seed = 10


class BiGRUConfig(object):
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10

    data_path = "../data/tang.npz"

    vocab_size = 0
    ix2word = None
    word2ix = None

    num_layers = 2
    embedding_dim = 128
    hidden_dim = 256
    dropout_rate = 0.05
    max_len = 125

    batch_size = 128
    epoch = 20
    lr = 0.001
    seed = 10


class ShortcutGRUConfig(object):
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10

    data_path = "../data/tang.npz"

    vocab_size = 0
    ix2word = None
    word2ix = None

    num_layers = 2
    embedding_dim = 128
    hidden_dim_1 = 128
    hidden_dim_2 = 256
    dropout_rate = 0.05
    max_len = 125

    batch_size = 128
    epoch = 30
    lr = 0.001
    seed = 10


