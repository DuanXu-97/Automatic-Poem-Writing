
class ResNet34Config(object):
    model = 'ResNet'
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10
    num_classes = 2

    data_path = "../data/tang.npz"

    seed = 10
    batch_size = 16
    epoch = 20
    lr = 0.001


class ResNet50Config(object):
    model = 'ResNet'
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10
    num_classes = 2

    train_path = "../data/dogs-cats-images/dataset/training_set"
    test_path = "../data/dogs-cats-images/dataset/test_set"

    train_image_nums = 8000
    test_image_nums = 2000

    seed = 10
    batch_size = 16
    epoch = 20
    lr = 0.001


class DenseNet121Config(object):
    model = 'DenseNet'
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10

    train_path = "../data/dogs-cats-images/dataset/training_set"
    test_path = "../data/dogs-cats-images/dataset/test_set"

    growth = 32
    blocks = [6, 12, 24, 16]
    num_init_features = 64
    bn_size = 4
    dropout_rate = 0.1
    num_classes = 2

    train_image_nums = 8000
    test_image_nums = 2000

    seed = 10
    batch_size = 16
    epoch = 20
    lr = 0.001




