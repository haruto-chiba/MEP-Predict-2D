import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1  # 出力チャンネルのスケーリング（Bottleneckブロックでは4）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # shortcut用（次元が違うとき）

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, sigmoid=False):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # Initial layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self.sigmoid = sigmoid
        self.sigmoid_layer = nn.Sigmoid()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.sigmoid:
            x = self.sigmoid_layer(x)
        return x


class SmallInputResNet18(nn.Module):
    def __init__(self, num_classes=10, sigmoid=False):
        super(SmallInputResNet18, self).__init__()
        self.in_channels = 64

        # 🔽 小さい画像用にconv1を軽くする
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 🔽 maxpoolを削除（サイズが小さすぎるので）
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self.sigmoid = sigmoid
        self.sigmoid_layer = nn.Sigmoid()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)  # 削除

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.sigmoid:
            x = self.sigmoid_layer(x)
        return x


class SelfMadeModel(nn.Module):
    def __init__(self, num_classes=10, sigmoid=False):
        super(SelfMadeModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(
            256 * 7 * 5, num_classes
        )  # 入力サイズは入力画像の大きさに合わせて調整

        self.sigmoid = sigmoid
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.sigmoid:
            x = self.sigmoid_layer(x)

        return x


# -------------------------------------#


class block_3D(nn.Module):
    def __init__(
        self,
        first_conv_in_channels,
        first_conv_out_channels,
        identity_conv=None,
        stride=1,
    ):
        """
        残差ブロックを作成するクラス
        Args:
            first_conv_in_channels : 1番目のconv層（1×1）のinput channel数
            first_conv_out_channels : 1番目のconv層（1×1）のoutput channel数
            identity_conv : channel数調整用のconv層
            stride : 3×3conv層におけるstide数。sizeを半分にしたいときは2に設定
        """
        super(block_3D, self).__init__()
        # 1番目のconv層（1×1）
        self.conv1 = nn.Conv3d(
            first_conv_in_channels,
            first_conv_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm3d(first_conv_out_channels)
        # 2番目のconv層（3×3）
        # パターン3の時はsizeを変更できるようにstrideは可変
        self.conv2 = nn.Conv3d(
            first_conv_out_channels,
            first_conv_out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(first_conv_out_channels)

        # 3番目のconv層（1×1）
        # output channelはinput channelの4倍になる
        self.conv3 = nn.Conv3d(
            first_conv_out_channels,
            first_conv_out_channels * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm3d(first_conv_out_channels * 4)
        self.relu = nn.ReLU()

        # identityのchannel数の調整が必要な場合はconv層（1×1）を用意、不要な場合はNone
        self.identity_conv = identity_conv

    def forward(self, x):
        identity = x.clone()  # 入力を保持する

        x = self.conv1(x)  # 1×1の畳み込み
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(
            x
        )  # 3×3の畳み込み（パターン3の時はstrideが2になるため、ここでsizeが半分になる）
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # 1×1の畳み込み
        x = self.bn3(x)

        # 必要な場合はconv層（1×1）を通してidentityのchannel数の調整してから足す
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x


class ResNet50_3D(nn.Module):
    def __init__(self, num_classes, sigmoid):
        super(ResNet50_3D, self).__init__()

        # conv1はアーキテクチャ通りにベタ打ち
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # conv2_xはサイズの変更は不要のため、strideは1
        self.conv2_x = self._make_layer(
            block_3D, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1
        )

        # conv3_x以降はサイズの変更をする必要があるため、strideは2
        self.conv3_x = self._make_layer(
            block_3D,
            4,
            res_block_in_channels=256,
            first_conv_out_channels=128,
            stride=2,
        )
        self.conv4_x = self._make_layer(
            block_3D,
            6,
            res_block_in_channels=512,
            first_conv_out_channels=256,
            stride=2,
        )
        self.conv5_x = self._make_layer(
            block_3D,
            3,
            res_block_in_channels=1024,
            first_conv_out_channels=512,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

        self.sigmoid = sigmoid
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)  # in:(3,224*224)、out:(64,112*112)
        x = self.bn1(x)  # in:(64,112*112)、out:(64,112*112)
        x = self.relu(x)  # in:(64,112*112)、out:(64,112*112)
        x = self.maxpool(x)  # in:(64,112*112)、out:(64,56*56)

        x = self.conv2_x(x)  # in:(64,56*56)  、out:(256,56*56)

        x = self.conv3_x(x)  # in:(256,56*56) 、out:(512,28*28)

        x = self.conv4_x(x)  # in:(512,28*28) 、out:(1024,14*14)

        x = self.conv5_x(x)  # in:(1024,14*14)、out:(2048,7*7)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        if self.sigmoid:
            x = self.sigmoid_layer(x)
        return x

    def _make_layer(
        self,
        block,
        num_res_blocks,
        res_block_in_channels,
        first_conv_out_channels,
        stride,
    ):
        layers = []

        # 1つ目の残差ブロックではchannel調整、及びsize調整が発生する
        # identifyを足す前に1×1のconv層を追加し、サイズ調整が必要な場合はstrideを2に設定
        identity_conv = nn.Conv3d(
            res_block_in_channels,
            first_conv_out_channels * 4,
            kernel_size=1,
            stride=stride,
        )
        layers.append(
            block(res_block_in_channels, first_conv_out_channels, identity_conv, stride)
        )

        # 2つ目以降のinput_channel数は1つ目のoutput_channelの4倍
        in_channels = first_conv_out_channels * 4

        # channel調整、size調整は発生しないため、identity_convはNone、strideは1
        for i in range(num_res_blocks - 1):
            layers.append(
                block(
                    in_channels, first_conv_out_channels, identity_conv=None, stride=1
                )
            )

        return nn.Sequential(*layers)


class ResNet101_3D(nn.Module):
    def __init__(self, num_classes, sigmoid):
        super(ResNet101_3D, self).__init__()

        # conv1はアーキテクチャ通りにベタ打ち
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # conv2_xはサイズの変更は不要のため、strideは1
        self.conv2_x = self._make_layer(
            block_3D, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1
        )

        # conv3_x以降はサイズの変更をする必要があるため、strideは2
        self.conv3_x = self._make_layer(
            block_3D,
            4,
            res_block_in_channels=256,
            first_conv_out_channels=128,
            stride=2,
        )
        self.conv4_x = self._make_layer(
            block_3D,
            23,
            res_block_in_channels=512,
            first_conv_out_channels=256,
            stride=2,
        )
        self.conv5_x = self._make_layer(
            block_3D,
            3,
            res_block_in_channels=1024,
            first_conv_out_channels=512,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

        self.sigmoid = sigmoid
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)  # in:(3,224*224)、out:(64,112*112)
        x = self.bn1(x)  # in:(64,112*112)、out:(64,112*112)
        x = self.relu(x)  # in:(64,112*112)、out:(64,112*112)
        x = self.maxpool(x)  # in:(64,112*112)、out:(64,56*56)

        x = self.conv2_x(x)  # in:(64,56*56)  、out:(256,56*56)

        x = self.conv3_x(x)  # in:(256,56*56) 、out:(512,28*28)

        x = self.conv4_x(x)  # in:(512,28*28) 、out:(1024,14*14)

        x = self.conv5_x(x)  # in:(1024,14*14)、out:(2048,7*7)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        if self.sigmoid:
            x = self.sigmoid_layer(x)

        return x

    def _make_layer(
        self,
        block,
        num_res_blocks,
        res_block_in_channels,
        first_conv_out_channels,
        stride,
    ):
        layers = []

        # 1つ目の残差ブロックではchannel調整、及びsize調整が発生する
        # identifyを足す前に1×1のconv層を追加し、サイズ調整が必要な場合はstrideを2に設定
        identity_conv = nn.Conv3d(
            res_block_in_channels,
            first_conv_out_channels * 4,
            kernel_size=1,
            stride=stride,
        )
        layers.append(
            block(res_block_in_channels, first_conv_out_channels, identity_conv, stride)
        )

        # 2つ目以降のinput_channel数は1つ目のoutput_channelの4倍
        in_channels = first_conv_out_channels * 4

        # channel調整、size調整は発生しないため、identity_convはNone、strideは1
        for i in range(num_res_blocks - 1):
            layers.append(
                block(
                    in_channels, first_conv_out_channels, identity_conv=None, stride=1
                )
            )

        return nn.Sequential(*layers)


class ResNet152_3D(nn.Module):
    def __init__(self, num_classes, sigmoid):
        super(ResNet152_3D, self).__init__()

        # conv1はアーキテクチャ通りにベタ打ち
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # conv2_xはサイズの変更は不要のため、strideは1
        self.conv2_x = self._make_layer(
            block_3D, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1
        )

        # conv3_x以降はサイズの変更をする必要があるため、strideは2
        self.conv3_x = self._make_layer(
            block_3D,
            8,
            res_block_in_channels=256,
            first_conv_out_channels=128,
            stride=2,
        )
        self.conv4_x = self._make_layer(
            block_3D,
            36,
            res_block_in_channels=512,
            first_conv_out_channels=256,
            stride=2,
        )
        self.conv5_x = self._make_layer(
            block_3D,
            3,
            res_block_in_channels=1024,
            first_conv_out_channels=512,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

        self.sigmoid = sigmoid
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)  # in:(3,224*224)、out:(64,112*112)
        x = self.bn1(x)  # in:(64,112*112)、out:(64,112*112)
        x = self.relu(x)  # in:(64,112*112)、out:(64,112*112)
        x = self.maxpool(x)  # in:(64,112*112)、out:(64,56*56)

        x = self.conv2_x(x)  # in:(64,56*56)  、out:(256,56*56)

        x = self.conv3_x(x)  # in:(256,56*56) 、out:(512,28*28)

        x = self.conv4_x(x)  # in:(512,28*28) 、out:(1024,14*14)

        x = self.conv5_x(x)  # in:(1024,14*14)、out:(2048,7*7)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        if self.sigmoid:
            x = self.sigmoid_layer(x)

        return x

    def _make_layer(
        self,
        block,
        num_res_blocks,
        res_block_in_channels,
        first_conv_out_channels,
        stride,
    ):
        layers = []

        # 1つ目の残差ブロックではchannel調整、及びsize調整が発生する
        # identifyを足す前に1×1のconv層を追加し、サイズ調整が必要な場合はstrideを2に設定
        identity_conv = nn.Conv3d(
            res_block_in_channels,
            first_conv_out_channels * 4,
            kernel_size=1,
            stride=stride,
        )
        layers.append(
            block(res_block_in_channels, first_conv_out_channels, identity_conv, stride)
        )

        # 2つ目以降のinput_channel数は1つ目のoutput_channelの4倍
        in_channels = first_conv_out_channels * 4

        # channel調整、size調整は発生しないため、identity_convはNone、strideは1
        for i in range(num_res_blocks - 1):
            layers.append(
                block(
                    in_channels, first_conv_out_channels, identity_conv=None, stride=1
                )
            )

        return nn.Sequential(*layers)


# テスト
if __name__ == "__main__":
    model = ResNet152_3D(num_classes=1)

    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    print("Output shape:", out.shape)
