import torch as t
from torch import nn
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


class ResidualBlockBasic(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlockBasic, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel))
        self.shortcut_branch = shortcut

    def forward(self, x):
        out = self.main_branch(x)
        residual = x if self.shortcut_branch is None else self.shortcut_branch(x)
        out += residual
        return F.relu(out)


class ResidualBlockBottleneck(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlockBottleneck, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel * 4),
        )
        self.shortcut_branch = shortcut

    def forward(self, x):
        out = self.main_branch(x)
        residual = x if self.shortcut_branch is None else self.shortcut_branch(x)
        out += residual
        return F.relu(out)


class ResNet34(BasicModule):

    def __init__(self, config):
        super(ResNet34, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.present_channel = 64
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, config.num_classes)

    def _make_layer(self, channel, block_num, stride=1):

        if stride != 1 or self.present_channel != channel:
            shortcut = nn.Sequential(
                nn.Conv2d(self.present_channel, channel, 1, stride, bias=False),
                nn.BatchNorm2d(channel)
            )
        else:
            shortcut = None

        layers = []
        layers.append(ResidualBlockBasic(self.present_channel, channel, stride, shortcut))
        self.present_channel = channel
        for i in range(1, block_num):
            layers.append(ResidualBlockBasic(self.present_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = t.flatten(x, 1)

        logits = self.fc(x)
        output = F.softmax(logits, dim=1)

        return logits, output


class ResNet50(BasicModule):
    def __init__(self, config):
        super(ResNet50, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.present_channel = 64
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * 4, config.num_classes)

    def _make_layer(self, channel, block_num, stride=1):

        if stride != 1 or self.present_channel != channel * 4:
            shortcut = nn.Sequential(
                nn.Conv2d(self.present_channel, channel * 4, 1, stride, bias=False),
                nn.BatchNorm2d(channel * 4)
            )
        else:
            shortcut = None

        layers = []
        layers.append(ResidualBlockBottleneck(self.present_channel, channel, stride, shortcut))
        self.present_channel = channel * 4
        for i in range(1, block_num):
            layers.append(ResidualBlockBottleneck(self.present_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = t.flatten(x, 1)

        logits = self.fc(x)
        output = F.softmax(logits, dim=1)

        return logits, output


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth, bn_size, dropout_rate):
        super(DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth, growth,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),

        self.dropout_rate = float(dropout_rate)

    def bn_function(self, inputs):
        concated_features = t.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, features):
        prev_features = features
        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.dropout_rate > 0:
            new_features = F.dropout(new_features, p=self.dropout_rate,
                                     training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth, dropout_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth,
                growth=growth,
                bn_size=bn_size,
                dropout_rate=dropout_rate,
            )
            self.add_module('denselayer{i}'.format(i=i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return t.cat(features, 1)


class DenseNet121(BasicModule):
    def __init__(self, config):
        super(DenseNet121, self).__init__()
        self.config = config

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, config.num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(config.num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = config.num_init_features
        for i, num_layers in enumerate(config.blocks):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=config.bn_size,
                growth=config.growth,
                dropout_rate=config.dropout_rate,
            )
            self.features.add_module('denseblock{i}'.format(i=i + 1), block)
            num_features = num_features + num_layers * config.growth

            if i != len(config.blocks) - 1:
                trans = nn.Sequential(OrderedDict([
                    ('norm', nn.BatchNorm2d(num_features)),
                    ('relu', nn.ReLU(inplace=True)),
                    ('conv', nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False)),
                    ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                ]))
                self.features.add_module('transition{i}'.format(i=i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.fc = nn.Linear(num_features, config.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = t.flatten(x, 1)
        logits = self.fc(x)
        output = F.softmax(logits, dim=1)

        return logits, output
