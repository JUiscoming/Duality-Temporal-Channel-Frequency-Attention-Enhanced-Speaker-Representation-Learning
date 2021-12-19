# resnet34
# SENet
# classifier (AM-Softmax)
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ResBlock, DTCFAttentionBlock


class ResNet_DTCF(nn.Module):
    """
    I refered the Attentive Statistics Pooling (ASP) code at https://github.com/clovaai/voxceleb_trainer/blob/master/models/ResNetSE34L.py
    """
    def __init__(self, num_blocks=[3,4,6,3], num_filters=[32, 32, 64, 128, 256], emb_dim=512, n_mels=80):
        super(ResNet_DTCF, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer_1 = self._make_layer(ResBlock, num_blocks[0], num_filters[0], num_filters[1], 1)
        self.layer_2 = self._make_layer(ResBlock, num_blocks[1], num_filters[1], num_filters[2], 2)
        self.layer_3 = self._make_layer(ResBlock, num_blocks[2], num_filters[2], num_filters[3], 2)
        self.layer_4 = self._make_layer(ResBlock, num_blocks[3], num_filters[3], num_filters[4], 2)

        self.DTCF_1 = DTCFAttentionBlock(num_filters[1], 8)
        self.DTCF_2 = DTCFAttentionBlock(num_filters[2], 8)
        self.DTCF_3 = DTCFAttentionBlock(num_filters[3], 8)
        self.DTCF_4 = DTCFAttentionBlock(num_filters[4], 8)

        self.instance_norm = nn.InstanceNorm1d(n_mels)

        self.sap_linear = nn.Linear(num_filters[-1], num_filters[-1])

        out_L = int(n_mels/8)
        self.attention = nn.Sequential(
                    nn.Conv1d(num_filters[-1]*out_L, 128, kernel_size=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Conv1d(128, num_filters[-1]*out_L, kernel_size=1),
                    nn.Softmax(dim=2),
                    )
        # We use ASP as the pooling layer with the intermediate channel dimension of 128.

        self.fc = nn.Linear(num_filters[-1]*out_L*2, emb_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, n_blocks, input_channels, output_channels, stride=1):
        layers = []
        layers.append(block(input_channels, output_channels, stride))
        for i in range(1, n_blocks):
            layers.append(block(output_channels, output_channels, 1))

        return nn.Sequential(*layers)


    def forward(self, x):
        with torch.no_grad():
            # Mean and variance normalisation (MVN) is performed by applying instance normalisation to the network input.
            # https://arxiv.org/pdf/2003.11982.pdf
            x = self.instance_norm(x.squeeze(1)).unsqueeze(1).detach() # (B, F, T) -> (B, 1, F, T)

        x = self.conv1(x) # (B, 1, F, L) - > (B, 32, F, L)
        x = self.bn1(x)
        x = self.relu(x)

        # ResBlocks and DTCF
        x = self.layer_1(x)  # (B, 32, F, L) -> (B, 32, F, L)
        x = self.DTCF_1(x)
        x = self.layer_2(x) # (B, 32, F, L) -> (B, 64, F/2, L/2)
        x = self.DTCF_2(x)
        x = self.layer_3(x) # (B, 64, F/2, L/2) -> (B, 128, F/4, L/4)
        x = self.DTCF_3(x)
        x = self.layer_4(x) # (B, 128, F/4, L/4) -> (B, 256, F/8, L/8)
        x = self.DTCF_4(x)
        x = x.reshape(x.size()[0],-1,x.size()[-1]) # (B, 256, F/8, L/8) -> (B, 256*F/8, L/8)

        # SAP
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu,sg),1)

        # Classifier
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    num_filters = [32, 32, 64, 128, 256]
    x = torch.randn((2, 80, 80))
    
    model = ResNet_DTCF()
    print(model(x).shape)

    # n = 1+(32000-160)//400 == 80
    # print(n)