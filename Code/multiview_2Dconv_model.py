import torch.nn as nn
import torch
from Code.load_parameter import savecheckpoint, LoadParameter
import pretrainedmodels


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=stride,
                     padding=(0,1,1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1,2,2), padding=1)
        #self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(1,2,2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(1,2,2),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(1,2,2),
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1,1, 1))

        #self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # block,
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        identity=x
        for i in range(3):
            if(i==0):
                x=identity.permute(0,1,4,3,2)
            elif(i==1):
                x=identity.permute(0,1,3,2,4)
            else:
                x=identity
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if (i == 0):
                out1 = x
            elif (i == 1):
                out2 = x
            else:
                out3=x
        x = self.fc1(out1+out2+out3)
        del identity
        return x

def resnet18(num_classes,**kwargs):
    model = ResNet(num_classes,BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class conv_3D(nn.Module):
    def __init__(self):
        super(conv_3D, self).__init__()
        self.conv=nn.Conv3d(3,5,kernel_size=(1,7,7),stride=1,padding=1)
    def forward(self,input):
        return self.conv(input)


class conv_2D(nn.Module):
    def __init__(self):
        super(conv_2D, self).__init__()
        self.conv=nn.Conv2d(3,5,kernel_size=(7,7),stride=1,padding=1,bias=False)
        self.conv2=nn.Conv2d(5,6,9,1,1,bias=False)
    def forward(self,input):
        return self.conv(input)

if __name__ == '__main__':

    from torchvision import models

    resnet = models.resnet18(pretrained=True)
    model=resnet18()
    args1 = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet').state_dict()
    input = torch.ones(10, 3, 16, 224, 224)
    model_state_dict=model.state_dict()
    i=0
    j=0
    for key in args1:
        if 'conv' in key or 'downsample.0' in key:
            args1[key]=args1[key].unsqueeze(2)
            i=i+1
        if key in model_state_dict:
            model_state_dict[key] = args1[key]
            j=j+1

    model.load_state_dict(model_state_dict)
    model.cuda()
    input=input.cuda()
    out=model(input)

    print(1)
