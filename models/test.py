import sys
sys.path.append('../')
import os
import argparse
import torch as t
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from models import network
from utils.dataset import PoemDataset
from models import configs


def generate_with_beginning(args):
    config = getattr(configs, args.model + 'Config')()
    dataset = PoemDataset(data_path=config.data_path, config=config)
    config.vocab_size = dataset.vocab_size
    config.word2ix = dataset.word2ix
    config.ix2word = dataset.ix2word
    config.use_gpu = args.use_gpu
    model = getattr(network, args.model)(config).eval()

    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        model.cuda()

    poem = list(args.given_words)
    given_words_len = len(args.given_words)
    input = Variable(t.Tensor([config.word2ix['<START>']]).view(1,1).long())

    if args.use_gpu:
        input = input.cuda()
    hidden = None

    if len(args.prefix_words) > 0:
        for word in args.prefix_words:
            output, hidden = model(input, hidden)
            input = Variable(input.data.new([config.word2ix[word]]).view(1,1))

    for i in range(config.max_len):
        output, hidden = model(input, hidden)

        if i < given_words_len:
            word = args.given_words[i]
            input = Variable(input.data.new([config.word2ix[word]]).view(1,1))

        else:
            word_index = output.data[0].topk(1)[1][0].item()
            word = config.ix2word[word_index]
            poem.append(word)
            input = Variable(input.data.new([word_index]).view(1,1))

        if word == '<EOP>':
            break

    return poem


def generate_with_acrostic(args):
    config = getattr(configs, args.model + 'Config')()
    dataset = PoemDataset(data_path=config.data_path, config=config)
    config.vocab_size = dataset.vocab_size
    config.word2ix = dataset.word2ix
    config.ix2word = dataset.ix2word
    config.use_gpu = args.use_gpu
    model = getattr(network, args.model)(config).eval()

    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        model.cuda()

    poem = []
    given_words_len = len(args.given_words)
    input = Variable(t.Tensor([config.word2ix['<START>']]).view(1,1).long())
    if args.use_gpu:
        input = input.cuda()
    hidden = None

    acrostic_index = 0
    pre_word = '<START>'

    if len(args.prefix_words) > 0:
        for word in args.prefix_words:
            output, hidden = model(input, hidden)
            input = Variable(input.data.new([config.word2ix[word]]).view(1,1))

    for i in range(config.max_len):
        output, hidden = model(input, hidden)
        word_index = output.data[0].topk(1)[1][0].item()
        word = config.ix2word[word_index]

        if pre_word in ['。', '<START>']:
            if acrostic_index == given_words_len:
                break
            else:
                word = args.given_words[acrostic_index]
                input = Variable(input.data.new([config.word2ix[word]]).view(1,1))
                acrostic_index += 1

        else:
            input = Variable(input.data.new([config.word2ix[word]]).view(1, 1))
        poem.append(word)
        pre_word = word

    return poem


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='AttLSTM', help="model to be used")
    parser.add_argument('--use_gpu', action='store_true', help="whether use gpu")
    parser.add_argument('--load_model_path', type=str, required=True, help="Path of trained model")
    parser.add_argument('--acrostic', action='store_true', help="whether generate acrostic poem")
    parser.add_argument('--given_words', type=str, default='月落乌啼霜满天', help="given words to generate poem")
    parser.add_argument('--prefix_words', type=str, default='独怜幽草涧边生，上有黄鹂深树鸣。', help="given words to generate poem")

    args = parser.parse_args()

    if args.acrostic:
        poem = generate_with_acrostic(args)
    else:
        poem = generate_with_beginning(args)

    print(''.join(poem))

