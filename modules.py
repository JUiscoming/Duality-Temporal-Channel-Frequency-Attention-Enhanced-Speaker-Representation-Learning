import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, groups: int=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.short_cut = nn.Sequential() # shortcut connection = skip connection = identity term
        if stride != 1: # if spatial shape (feature map size) is reduced, skip connection is also downsampled.
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x): 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.short_cut(x) # skip connection
        out = self.relu(out)

        return out


class DTCFAttentionBlock(nn.Module):
    """
    SENet:
        from the global average pooling (GAP), get inter-channel correlation and calculate channel attention score.
    - input: 3D speech feature map. shape=(Channel x Frequency x Time)
    - output: channel attention coefficient. shape=(Channel)
    
    Duality Temporal-Channel-Frequency (DTCF) Attention Block:
        SENet attention averages the channel-frequency-time feature map R^(CxFxT) -> R^(C)
        , which may lose the discriminative speaker information in temporal and frequency domain.
        SE attention neglects the exploration in global context relationships on time and frequency dimensions.
        To alleviate these issues, encode global TF information into the channel-wise feature


    """
    def __init__(self, C, reduction_factor=8):
        # C: channel_dim, F: frequency_dim, T: time_dim
        super(DTCFAttentionBlock, self).__init__()
        self.freq_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, None)) # (B, C, F, T) -> (B, C, 1, T)
        self.time_avg_pool = nn.AdaptiveAvgPool2d(output_size=(None, 1)) # (B, C, F, T) -> (B, C, F, 1)

        self.conv_bottleneck = nn.Conv2d(C, C//reduction_factor, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C//reduction_factor)
        self.relu = nn.ReLU(inplace=True)

        self.conv_time_attn = nn.Conv2d(C//reduction_factor, C, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_freq_attn = nn.Conv2d(C//reduction_factor, C, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        F, T = x.shape[2:]
        # 1. 2D-pooling to extract global temporal and frequency information
        global_time_info = self.freq_avg_pool(x).squeeze(2) # (B, C, F, T) -> (B, C, T)
        global_freq_info = self.time_avg_pool(x).squeeze(3) # (B, C, F, T) -> (B, C, F)

        # 2. Convolution layer as bottleneck
        z = torch.cat((global_time_info, global_freq_info), dim=-1).unsqueeze(-1)
        z = self.conv_bottleneck(z)
        z = self.bn(z)
        z = self.relu(z)

        # 3. calculate temporal and frequency attention masking coef.
        attn_t, attn_f = torch.split(z, [T, F], dim=2)
        attn_t = self.sigmoid(self.conv_time_attn(attn_t).permute(0, 1, 3, 2).contiguous()) # (B, C, T, 1)
        attn_f = self.sigmoid(self.conv_freq_attn(attn_f)) # (B, C, F, 1)
        attn = attn_t * attn_f

        # 4. apply TF attention
        return x * attn


if __name__ == '__main__':
    # l = 32000
    # w = 400
    # s = 160

    # n, mod = 1+(l-w)//s, (l-w)%s
    # print(n, mod)
    y = torch.tensor([[[[1,1,1,1],
                    [2,2,2,2],
                    [3,3,3,3],
                    [4,4,4,4]]]]).type(torch.float32)

    x = torch.randn((2, 64, 8, 8))
    m = DTCFAttentionBlock(64, 8)

    m(x)
    # m = nn.AdaptiveAvgPool2d((None, 1))
    # print(m(x).shape)