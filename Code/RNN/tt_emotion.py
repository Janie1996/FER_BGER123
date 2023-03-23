"""
提取特征multiview 3D-resnet
特征输入RNN
"""
import torch.nn as nn
import torch
from Code.load_parameter import savecheckpoint, LoadParameter
import pretrainedmodels

from gensim.models import word2vec

sentences = [['An'],['Di'],['Fe'],['Ha'],['Ne'],['Sa'],['Su']]
model = word2vec.Word2Vec(sentences, size=200,min_count=1)  # 默认window=5
#print(model.wv.index2word)
emotion_embed= torch.from_numpy(model.wv.vectors)   # [classes, embedding]
#print(model.wv.vectors)
print(emotion_embed[0][12])


class RNN_feature(nn.Module):
    def __init__(self,in_dim,hidden_dim,classes):
        super(RNN_feature, self).__init__()
        self.lstm=nn.LSTM(in_dim,hidden_dim,1,batch_first=True)
        self.fc=nn.Linear(hidden_dim,classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # 此时可以从out中获得最终输出的状态h
        x = out[:, -1, :]
        # x = h_n[-1, :, :]
        x = self.fc(x)
        return x

class RNN_emotion(nn.Module):
    def __init__(self,in_dim,hidden_dim,classes):
        super(RNN_emotion, self).__init__()
        self.lstm=nn.LSTM(in_dim,hidden_dim,1,batch_first=True)
        self.fc=nn.Linear(hidden_dim,classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # 此时可以从out中获得最终输出的状态h
        x = out[:, -1, :]
        # x = h_n[-1, :, :]
        x = self.fc(x)
        return x




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
        self.conv = nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=(1, 1),stride=(1))
        self.batchnorm = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        x1 = out.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3] * out.shape[4])

        x2 = out.transpose(2, 3)
        x2 = x2.contiguous().view(x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3] * x2.shape[4])

        x3 = out.transpose(2, 4)
        x3 = x3.contiguous().view(x3.shape[0], x3.shape[1], x3.shape[2], x3.shape[3] * x3.shape[4])

        out1 = self.conv(x1)
        out1 = self.batchnorm(out1)
        out1 = self.relu(out1)
        out1 = out1.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3],out.shape[4])

        out2 = self.conv(x2)
        out2 = self.batchnorm(out2)
        out2 = self.relu(out2)
        #out2 = out2.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3],out.shape[4])
        out2 = out2.view(out.shape[0], out.shape[1], out.transpose(2, 3).shape[2],
                         out.transpose(2, 3).shape[3],out.transpose(2, 3).shape[4] )
        out2 = out2.transpose(2, 3)


        out3 = self.conv(x3)
        out3 = self.batchnorm(out3)
        out3 = self.relu(out3)
        #out3 = out3.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3],out.shape[4])    #  需要修改次序
        out3 = out3.view(out.shape[0], out.shape[1], out.transpose(2, 4).shape[2],
                         out.transpose(2, 4).shape[3], out.transpose(2, 4).shape[4])
        out3 = out3.transpose(2, 4)


        out=out1+out2+out3
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, batch,C,num_classes, block, layers,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.batch=batch
        self.C=C
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

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        self.rnn_size=512 * block.expansion
        self.rnn= RNN_feature(self.rnn_size,1024,num_classes)
        # self.rnn1=RNN_emotion(200,512,num_classes)
        self.rnn1 = RNN_emotion(7, 512, num_classes)
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

        batch=int(x.shape[0]/self.C)
        clips = torch.ones([self.C,batch , self.rnn_size])

        for i in range(self.C):
            clips[i] = x[i * batch:(i + 1) *batch, :]
        clips = clips.transpose(0, 1).cuda()

        end = self.rnn(clips)   # 特征RNN结果

        out=self.dropout(x)
        out=self.fc1(out)    # batch*clips,7

        soft=nn.Softmax(1)
        out=soft(out)

        # batch = int(x.shape[0] / self.C)
        # index = torch.max(out.data, 1)[1]
        # clips = torch.ones([self.C, batch, 200])
        #
        #
        # for i in range(self.C):
        #     for j in range(batch):
        #         clips[i][j] = emotion_embed[index[i * batch + j]]
        #
        # clips = clips.transpose(0, 1).cuda()  # batch,length,200
        probs = out.view(self.C,-1,7).permute(1,2,0)  # batch,7,length
        maxpool = nn.AdaptiveAvgPool1d(1)
        probs = maxpool(probs).squeeze(2)  # clips的prob均值化为结果

        clips=out.view(self.C,-1,7).permute(1,0,2)
        out = self.rnn1(clips)    # 情感RNN结果
        del clips

        return probs,end,out,probs+end+out
        #return probs,out,out+end   #均值；情感；特征

def resnet18(batch,C,num_classes,**kwargs):
    model = ResNet(batch,C,num_classes,BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

if __name__ == '__main__':

    # from torchvision import models
    #
    # resnet = models.resnet18(pretrained=True)
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
