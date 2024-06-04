import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepCrossAttention(nn.Module):
    def __init__(
            self, dim, heads, in_pixels = 1, qkv_bias=False, qk_scale=None, dropout_rate=0.0,
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.dim = dim
        self.in_pixels = in_pixels
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.out = {}
        self.attention = ScaledDotProductAttention(temperature=in_pixels ** 0.5)

    def forward(self, input):

        a = x = input['x']
        b = y = input['y']
        c = z = input['z']

        B, N, C = x.shape
        qkv1 = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        qkv2 = (
            self.qkv(y)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        qkv3 = (
            self.qkv(z)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv1[0],
            qkv1[1],
            qkv1[2],
        )
        q1, k1, v1 = (
            qkv2[0],
            qkv2[1],
            qkv2[2],
        )  # ma
        q2, k2, v2 = (
            qkv3[0],
            qkv3[1],
            qkv3[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        x1 = (attn @ v1).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        x1 = x1 + a

        x2 = (attn @ v2).transpose(1, 2).reshape(B, N, C)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)
        x2 = x2 + a

        x = x1 + x2

        y1 = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        y1 = self.proj(y1)
        y1 = self.proj_drop(y1)
        y1 = y1 + b

        y2 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        y2 = self.proj(y2)
        y2 = self.proj_drop(y2)
        y2 = y2 + b

        y = y1 + y2

        z1 = (attn2 @ v).transpose(1, 2).reshape(B, N, C)
        z1 = self.proj(z1)
        z1 = self.proj_drop(z1)
        z1 = z1 + c

        z2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        z2 = self.proj(z2)
        z2 = self.proj_drop(z2)
        z2 = z2 + c

        z = z1 + z2

        self.out['x'], self.out['y'] = x, y
        self.out['z'] = z
        return self.out

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)
        return output

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings

class TransformerModel(nn.Module):
    def __init__(
            self,
            map_size,
            M_channel,
            dim,
            heads,
            mlp_dim,
            attn_dropout_rate=0.1,
    ):
        super().__init__()

        self.attention = DeepCrossAttention(dim, heads, dropout_rate=attn_dropout_rate)
        self.input = {}
        self.output = {}
        self.map_size = map_size
        self.linear_encoding = nn.Linear(M_channel, dim)
        self.linear_encoding_same = nn.Linear(dim, dim)
        self.linear_encoding_de = nn.Linear(dim, M_channel)
        self.position_encoding = LearnedPositionalEncoding(M_channel, dim, map_size * map_size)

    def forward(self, x, y, z):
        # [batch_size, channels，width, height]-》[batch_size, width, height, channels]
        x_ = x.permute(0, 2, 3, 1).contiguous()
        y_ = y.permute(0, 2, 3, 1).contiguous()
        z_ = z.permute(0, 2, 3, 1).contiguous()
        x = x_.view(x_.size(0), x_.size(2)*x_.size(1), -1)
        y = y_.view(y_.size(0), y_.size(2)*y_.size(1), -1)
        z = z_.view(z_.size(0), z_.size(2) * z_.size(1), -1)

        self.input['x'] = self.position_encoding(self.linear_encoding(x))
        self.input['y'] = self.position_encoding(self.linear_encoding(y))
        self.input['z'] = self.position_encoding(self.linear_encoding(z))

        results = self.attention(self.input)
        x, y, z = results['x'], results['y'], results['z']
        x = self.linear_encoding_de(x).permute(0, 2, 1).contiguous()
        self.output['x'] = x.view(x.size(0), x.size(1), self.map_size, self.map_size)
        y = self.linear_encoding_de(y).permute(0, 2, 1).contiguous()
        self.output['y'] = y.view(y.size(0), y.size(1), self.map_size, self.map_size)
        z = self.linear_encoding_de(z).permute(0, 2, 1).contiguous()
        self.output['z'] = z.view(x.size(0), x.size(1), self.map_size, self.map_size)
        self.output['z'] = self.output['x'] + self.output['y'] + self.output['z']
        return self.output