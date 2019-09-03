import torch.nn as _nn
import torch.nn.functional as _F


# BottleNeck Block
class BottleNeck(_nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = _nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = _nn.BatchNorm2d(planes)
        self.conv2 = _nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = _nn.BatchNorm2d(planes)
        self.conv3 = _nn.Conv2d(planes, self.expansion*planes, kernel_size=1,
                                bias=False)
        self.bn3 = _nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = _nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = _nn.Sequential(
                _nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                           stride=stride, bias=False),
                _nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = _F.relu(self.bn1(self.conv1(x)))
        out = _F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = _F.relu(out)
        return out


# ResNet
class ResNet(_nn.Module):
    def __init__(self, block, num_blocks, num_classes=20):
        super(ResNet, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.conv1 = _nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.bn1 = _nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = _nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return _nn.Sequential(*layers)

    def forward(self, x):
        out = _F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = _F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def __repr__(self):
        d = {'name': self.name,
             'num_classes': self.num_classes,
             'num_blocks': self.num_blocks}
        return str(d)


def resnet50(num_classes):
    x = ResNet(BottleNeck, [3, 4, 6, 3], num_classes)
    x.name = "{}50".format(x.name)
    return x


def init_weights(m):
    if type(m) == _nn.Conv2d or type(m) == _nn.Linear:
        _nn.init.xavier_normal_(m.weight.data)
