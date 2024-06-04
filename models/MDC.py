from models.Transformer import TransformerModel
from einops import rearrange
from common import *

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

class MRCB(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(MRCB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=(1, 1, 1)),
            BatchNorm(in_channels // 4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            BatchNorm(in_channels // 4),
            nn.ReLU(),
        )
        self.inp = inp

        self.deconv1 = nn.Sequential(
            nn.Conv3d(in_channels // 4, in_channels // 8, (1, 3, 1), padding=(0, 1, 0)),
            BatchNorm(in_channels // 8),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.Conv3d(in_channels // 4, in_channels // 8, (3, 1, 1), padding=(1, 0, 0)),
            BatchNorm(in_channels // 8),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.Conv3d(in_channels // 4, in_channels // 8, (1, 1, 3), padding=(0, 0, 1)),
            BatchNorm(in_channels // 8),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.Conv3d(n_filters, in_channels//8,(3,1,1),padding=(1,0,0)),
            BatchNorm(in_channels//8),
            nn.ReLU(),
        )

    def forward(self, x, inp=False):
        x_4d = x
        x_5d = torch.unsqueeze(x_4d, dim=2).expand(-1, -1, 1, -1, -1)
        x = self.conv1(x_5d)
        m = self.conv2(x_5d)

        x1 = self.deconv1(m)
        x2 = self.deconv2(m)
        x3 = self.deconv3(m)

        x4 = self.inv_h_transform(self.deconv1(self.h_transform(x)))

        x5 = self.inv_v_transform(self.deconv3(self.v_transform(x)))

        x6 = self.d_transform(x)
        x6 = self.deconv4(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6), 1)
        x = rearrange(x, 'b c d h w ->b (c d) h w')
        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], shape[2], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[3], 2 * shape[4] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], shape[2], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 2, 4, 3)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], shape[2], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[3], 2 * shape[4] - 1)
        return x.permute(0, 1, 2, 4, 3)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 2, 4, 3)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], shape[2], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 2, 4, 3)

    def d_transform(self, x):
        x = x.permute(0, 4, 2, 3, 1)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], shape[2], -1)[..., :]
        x = x.reshape(shape[0], shape[1], shape[2], shape[3], 2 * shape[4])
        return x.permute(0, 4, 2, 3, 1)

class PFMM(nn.Module):
    def __init__(self, in_channels):
        super(PFMM, self).__init__()
        self.n_bands = in_channels

        self.conv1 = nn.Conv2d(in_channels=self.n_bands, out_channels=self.n_bands//2, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(self.n_bands//2)
        self.conv1_2 = nn.Conv2d(in_channels=self.n_bands//2, out_channels=self.n_bands//2, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(self.n_bands//2)

        self.conv2 = nn.Conv2d(in_channels=self.n_bands//2, out_channels=self.n_bands, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.n_bands)
        self.conv2_2 = nn.Conv2d(in_channels=self.n_bands, out_channels=self.n_bands, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(self.n_bands)

        self.conv3 = nn.Conv2d(in_channels=self.n_bands, out_channels=self.n_bands * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.n_bands * 2)
        self.conv3_2 = nn.Conv2d(in_channels=self.n_bands * 2, out_channels=self.n_bands * 2, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(self.n_bands * 2)

        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, x):

        out1 = self.LeakyReLU(self.bn1(self.conv1(x)))
        out1 = self.bn1_2(self.conv1_2(out1))

        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        out2 = self.LeakyReLU(self.bn2(self.conv2(out1_mp)))
        out2 = self.bn2_2(self.conv2_2(out2))

        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        out3 = self.LeakyReLU(self.bn3(self.conv3(out2_mp)))
        out3 = self.bn3_2(self.conv3_2(out3))
        return out1, out2, out3


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class SFE(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats * 2, stride=2)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats * 2, out_channels=n_feats * 2,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats * 2, n_feats * 2, stride=1)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x

#################################################################
class MDC(nn.Module):
    def __init__(self,
                 arch,
                 scale_ratio,
                 n_select_bands,
                 n_bands,
                 dataset
                 ):
        super(MDC, self).__init__()

        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))

        self.num_res_blocks = [16, 4, 4, 4, 4]
        self.res_scale = 1

        self.ca = ChannelAttention(n_bands)
        self.sa = SpatialAttention()

        self.D1 = nn.Sequential(
            nn.Conv2d(n_select_bands, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        if dataset == 'Pavia':
            u1_channel = n_bands + 48
        elif dataset == 'PaviaU':
            u1_channel = n_bands + 47
        elif dataset == 'Washington':
            u1_channel = n_bands + 91
        elif dataset == 'Houston_HSI':
            u1_channel = n_bands + 72
        elif dataset == 'Salinas_corrected':
            u1_channel = n_bands + 102
        self.U1 = nn.Sequential(
            nn.Conv2d(u1_channel, n_bands * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands * 2, n_bands * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        if dataset == 'Pavia':
            u2_channel = n_bands - 30
        elif dataset == 'PaviaU':
            u2_channel = n_bands - 31
        elif dataset == 'Washington':
            u2_channel = n_bands - 53
        elif dataset == 'Houston_HSI':
            u2_channel = n_bands - 36
        elif dataset == 'Salinas_corrected':
            u2_channel = n_bands - 54

        self.U2 = nn.Sequential(
            nn.Conv2d( u2_channel, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(n_bands * 2 + 5, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.PFMM_HSI = PFMM(in_channels=n_bands)
        self.PFMM_MSI = PFMM(in_channels=n_bands)

        self.transformer1 = TransformerModel(
            map_size= 16,
            M_channel = n_bands * 2,
            dim=128,
            heads=4,
            mlp_dim=n_bands,
            attn_dropout_rate=0.1,
        )
        self.transformer2 = TransformerModel(
            map_size = 32,
            M_channel=n_bands,
            dim=64,
            heads=4,
            mlp_dim=n_bands,
            attn_dropout_rate=0.1,
        )
        self.transformer3 = TransformerModel(
            map_size=64,
            M_channel=n_bands //2,
            dim=32,
            heads=4,
            mlp_dim=n_bands,
            attn_dropout_rate=0.1,
        )

        self.SFE = SFE(self.n_bands,self.num_res_blocks[0],self.n_bands,self.res_scale)

        self.conv11_headSUM = nn.Sequential(
            nn.Conv2d(self.n_bands * 2, self.n_bands, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.n_bands, self.n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands, self.n_bands * 2, kernel_size=1, stride=1, padding=0),
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(self.n_bands * 2, self.n_bands, kernel_size=3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands, self.n_bands, kernel_size=3, stride= 1, padding= 1),
        )

        #########stage2##############
        self.conv22_headSUM = nn.Sequential(
            nn.Conv2d(self.n_bands, self.n_bands // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.n_bands // 2, self.n_bands //2 , kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands // 2, self.n_bands, kernel_size=1, stride=1, padding=0),
        )

        self.conv23 = nn.Sequential(
            nn.Conv2d(self.n_bands, self.n_bands // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands // 2, self.n_bands // 2, kernel_size=3, stride=1, padding=1),
        )
        #########stage3##############
        self.conv33_headSUM = nn.Sequential(
            nn.Conv2d((self.n_bands // 2) * 3, self.n_bands // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.n_bands // 4, self.n_bands // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands // 4, self.n_bands // 2, kernel_size=1, stride=1, padding=0),
        )

        self.conv34 = nn.Sequential(
            nn.Conv2d(self.n_bands//2, self.n_bands // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands // 4, self.n_bands // 4, kernel_size=3, stride=1, padding=1),
        )

        self.conv44_headSUM = nn.Sequential(
            nn.Conv2d((self.n_bands // 4) * 3, self.n_bands // 8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.n_bands // 8, self.n_bands // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands // 8, self.n_bands // 4, kernel_size=1, stride=1, padding=0),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d((self.n_bands * 2) + (self.n_bands) + (self.n_bands // 2), self.n_bands, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands, self.n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        if dataset == 'PaviaU' or dataset ==  'Washington':
            u3_channel = n_bands - 1
        elif dataset == 'Houston_HSI' or dataset == 'Pavia' or dataset =='Salinas_corrected':
            u3_channel = n_bands
        self.MRCB1 = MRCB(n_bands * 2, u3_channel, nn.BatchNorm3d)

        if dataset == 'PaviaU' or dataset == 'Washington' or dataset =='Pavia':
            u4_channel = n_bands //2 - 1
        elif dataset == 'Houston_HSI' or dataset =='Salinas_corrected':
            u4_channel = n_bands // 2
        self.MRCB2 = MRCB(n_bands, u4_channel, nn.BatchNorm3d)

        self.conv_spat = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv_spec = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x_lr, x_hr):

        if self.arch == 'MDC':

            x_MSI = self.D1(x_hr)
            x_MSI = x_MSI * self.ca(x_MSI)
            x_MSI = x_MSI * self.sa(x_MSI)

            x_MSI_UD = F.interpolate(x_MSI, scale_factor=2, mode='bilinear')
            x_MSI_UD = F.interpolate(x_MSI_UD, scale_factor=1 / 2, mode='bilinear')

            x_HSI = F.interpolate(x_lr, scale_factor=2, mode='bilinear')
            x_HSI = x_HSI * self.sa(x_HSI)

            x_m1, x_m2, x_m3 = self.PFMM_MSI(x_MSI)
            x_mu1, x_mu2, x_mu3 = self.PFMM_MSI(x_MSI_UD)
            x_l1, x_l2, x_l3 = self.PFMM_HSI(x_HSI)

            d = F.interpolate(x_lr, scale_factor=4, mode='bilinear')
            d = d * self.ca(d)
            d = d * self.sa(d)

            # ###########H/8,W/8,2C################
            transformer_results = self.transformer1(x_m3, x_mu3, x_l3)
            e = transformer_results['z']
            x = self.SFE(x_lr)
            a = x
            a_res = a
            a_res = a_res + e
            a_res = self.conv11_headSUM(a_res)
            a = a_res + a
            a =self.MRCB1(a)
            a = self.U1(a)

            # ###########H/4,W/4,C################
            transformer_results1 = self.transformer2(x_m2, x_mu2, x_lr)
            g = transformer_results1['z']
            b = self.conv12(a)
            b = F.interpolate(b, scale_factor=2, mode='bilinear')
            b_res = b
            b_res = b_res + g
            b_res = self.conv22_headSUM(b_res)
            b = b + b_res
            b = self.MRCB2(b)
            b = self.U2(b)

            # ###########H/2,W/2,C################
            transformer_results2 = self.transformer3(x_m1, x_mu1, x_l1)
            h = transformer_results2['z']
            c = self.conv23(b)
            c = F.interpolate(c, scale_factor=2, mode='bilinear')
            c_res = c
            c_res = c_res + h
            c_res = torch.cat((c_res,x_m1,x_l1),1)
            c_res = self.conv33_headSUM(c_res)
            c = c + c_res

            a_up = F.interpolate(a, scale_factor=8, mode='bilinear')
            b_up = F.interpolate(b, scale_factor=4, mode='bilinear')
            c_up = F.interpolate(c, scale_factor=2, mode='bilinear')
            x_fusion = torch.cat((a_up, b_up, c_up),dim=1)

            x_fusion = self.final_conv(x_fusion)
            x = torch.cat((x_fusion, x_hr), 1)
            x = torch.cat((x, d), 1)
            x = self.conv3(x)
            x_spat = x + self.conv_spat(x)
            x_spec = x_spat + self.conv_spec(x_spat)

            x = x_spec
        return x
