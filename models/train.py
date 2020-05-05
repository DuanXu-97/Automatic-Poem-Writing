import sys
sys.path.append('../')
import os
import re
import time
import argparse
import torch as t
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet import meter
from models import configs
from models import network
from utils.dataset import PoemDataset
from utils.visualize import Visualizer


def train(args):
    vis = Visualizer()

    config = getattr(configs, args.model + 'Config')()
    model = getattr(network, args.model)(config).eval()

    dataset = PoemDataset(data_path=config.data_path, config=config)
    dataloader = DataLoader(dataset, config.batch_size,
                            shuffle=True,
                            num_workers=config.num_workers)

    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    loss_meter = meter.AverageValueMeter()

    time_begin = time.clock()

    for epoch in range(config.epoch):

        # train
        model.train()
        loss_meter.reset()

        for _iter, data in enumerate(dataloader):

            data = data.long().transpose(1,0).contiguous()

            if args.use_gpu:
                data = data.cuda()

            optimizer.zero_grad()
            input, target = Variable(data[:-1, :]), Variable(data[1:, :])
            output, _ = model(input)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.data[0])

            if _iter % config.print_freq == 0:
                vis.plot('train_loss', loss_meter.value()[0])
        model.save(path=os.path.join(args.ckpts_dir, 'model_{0}.pth'.format(str(epoch))))

        vis.log("epoch:{epoch}, train_loss:{train_loss}".format(
            epoch=epoch,
            train_loss=loss_meter.value()[0],
        ))
        print("epoch:{epoch}, train_loss:{train_loss}".format(
            epoch=epoch,
            train_loss=loss_meter.value()[0],
        ))

    model.save(path=os.path.join(args.ckpts_dir, 'model.pth'))
    vis.save()
    print("save model successfully")
    time_end = time.clock()
    print('time cost: %.2f' % (time_end - time_begin))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ResNet', help="model to be used")
    parser.add_argument('--pretrain', action='store_true', help="whether use pretrained model")
    parser.add_argument('--use_gpu', action='store_true', help="whether use gpu")
    parser.add_argument('--load_model_path', type=str, default=None, help="Path of pre-trained model")
    parser.add_argument('--ckpts_dir', type=str, default=None, help="Dir to store checkpoints")

    args = parser.parse_args()

    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir)

    train(args)





