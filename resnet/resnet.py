import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torchvision import transforms
import time
import matplotlib.pyplot as plt


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Reference (ResNet Architecture): https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/resnet.py#L124
class SingleBlock(nn.Module):
    """
    This is a basic block that contains two convolutional layers followed by
    a batch normalization layer and a ReLU activation function, where the skip
    connection is added before the second relu.
    ---

    - inplanes: { int } - The number of input channels.
    - planes: { int } - The number of output channels.
    - stride: { int } - The stride of convolutional layers.
    - downsample: { nn.Sequential } - A sequential of convolutional layers that fit the
        identity mapping to the desired output size.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SingleBlock, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        This is the forward pass of the basic block where the input tensor x is passed
        through the first convolutional layer, batch normalization layer, and the ReLU
        activation function. The result is passed through the second convolutional layer,
        batch normalization layer, and the ReLU activation function. The result is then
        added to the identity mapping and passed through the ReLU activation function.
        """
        residual = x

        # Convolve with a 3X3Xplanes kernel
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Convolve with a 3X3Xplanes kernel
        out = self.conv2(out)
        out = self.bn2(out)

        # If the stride is not 1 or the number of input channels is not equal
        # to the number of output channels then we need to fit the identity
        # mapping to the desired output size by applying the downsample.
        if self.downsample is not None:
            residual = self.downsample(x)

        # Add the identity mapping to the output of the second convolutional layer.
        out += residual
        # Apply the ReLU activation function after the addition.
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d( planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # Convolve with a 1X1Xplanes kernel
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Convolve with a 3X3Xplanes kernel
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Convolve with a 1X1Xplanes*expansion kernel
        out = self.conv3(out)
        out = self.bn3(out)

        # If the stride is not 1 or the number of input channels is not equal
        # to the number of output channels then we need to fit the identity
        # mapping to the desired output size by applying the downsample.
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # Apply the ReLU activation function after the addition.
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    This is the ResNet class that is used in ResNet50, ResNet101, and ResNet152.
    """
    def __init__(self, block, layers, stride=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet18(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(SingleBlock, [2, 2, 2, 2], stride=stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=True)
    if num_classes != 1000:
        model.fc = nn.Linear(512 * SingleBlock.expansion, num_classes)
    return model

def resnet34(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(SingleBlock, [3, 4, 6, 3,], stride=stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=True)
    if num_classes != 1000:
        model.fc = nn.Linear(512 * SingleBlock.expansion, num_classes)
    return model


def resnet50(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param pretrained:
        :param stride:
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(Bottleneck, [3, 4, 6, 3], stride=stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50']), strict=True)
    if num_classes != 1000:
        model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model

def resnet101(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param pretrained:
        :param stride:
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = Network(stride=stride, block=Bottleneck, layers=[3, 4, 23, 3],  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=True)
    if num_classes != 1000:
        model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model

def resnet152(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = Network(stride=stride, block=Bottleneck, layers=[3, 8, 36, 3],  **kwargs)
    # model = ResNet(Bottleneck, [3, 8, 36, 3], stride=stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=True)
    if num_classes != 1000:
        model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model


class MultiClassifier(nn.Module):
    """
    This class defines  a multi-classifier head that takes in the output of the backbone
    and classifies the features into either one of the 200 classes of the CUB200 dataset.
    ---
    - inplanes: { int } - The number of input channels.
    - num_classes: { int } - The number of classes in the dataset.
    - fc: { nn.Linear } - A linear layer that takes in the output of the backbone and
        classifies the features into either one of the 200 classes of the CUB200 dataset.
    - dropout: { nn.Dropout } - A dropout layer that randomly sets input elements to 0
        with a probability of 0.5.
    - softmax: { nn.Softmax } - A softmax function that normalizes the output of the
        linear layer.

    """
    def __init__(self, inplanes=2048):
        super(MultiClassifier, self).__init__()
        self.inplanes = inplanes
        self.num_classes = 200
        self.softmax = nn.Softmax(dim=1)
        # self.fconnected = nn.Linear(200, )

    def forward(self, x):
        """
        This function defines the forward pass of the multi-classifier head which takes in
        the output of the backbone and classifies the features into either one of the 200
        classes of the CUB200 dataset.
        """
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Network(nn.Module):
    """
    This class defines a network architecture that consists of a backbone and a multi-classifier
    head. This network architecture is used for training the CUB200 dataset for the purpose of
    fine-grained classification.
    ---
    - backbone: { nn.Module } - A backbone network.
    - classification_head: { MultiClassifier } - A multi-classifier head that takes in the
        output of the backbone and classifies the features into either one of the 200 classes
        of the CUB200 dataset.
    """
    def __init__(self, pretrained=True, cin=None):
        super(Network, self).__init__()

        # self.backbone = resnet18(pretrained=pretrained, num_classes=200)
        # self.backbone = resnet34(pretrained=pretrained, num_classes=200)
        self.backbone = resnet50(pretrained=pretrained, num_classes=200)
        # self.backbone = resnet101(pretrained=pretrained, num_classes=200)
        # self.backbone = resnet152(pretrained=pretrained, num_classes=200)

        self.classification_head = MultiClassifier()


    def forward(self, x):
        """
        This is the forward pass of the network architecture that consists of a backbone and a
        multi-classifier head. This network architecture is used for training the CUB200 dataset
        for the purpose of fine-grained classification.
        """
        feats = self.backbone(x)
        out = nn.Softmax(dim=1)

        return out, feats