import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
import torchvision

class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,
                                   padding=padding, groups=in_chans, bias=bias)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)

        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DeformableProjEmbed(nn.Module):
    """ feature map to Projected Embedding
    """
    def __init__(self, in_chans=512, emb_chans=128):
        super().__init__()
        self.kernel_size = kernel_size = 3
        self.stride = stride = 1
        self.padding = padding = 1
        self.proj = nn.Conv2d(in_chans, emb_chans, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_chans, 1 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.norm = nn.BatchNorm2d(emb_chans)
        self.act = nn.GELU()

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x

    def forward(self, x):
        x = self.deform_proj(x)
        x = self.act(self.norm(x))
        return x


class DeformableMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DeformableMLP, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.offset_modulator_conv = DWConv2d(in_channels, 3 * in_channels)

        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        B, C, H, W = input.size()
        offset_modulator = self.offset_modulator_conv(input)
        offset_y, offset_x, modulator = torch.chunk(offset_modulator, 3, dim=1)
        modulator = 2. * torch.sigmoid(modulator)
        offset = torch.cat((offset_y, offset_x), dim=1)
        max_offset = max(H, W) // 4
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(input=input,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation
                                          )

        x = self.act(self.norm(x))
        return x

    def extra_repr(self) -> str:
        # s = self.__class__.__name__ + '('
        s = ''
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        # s += ')'
        return s.format(**self.__dict__)


class DeformableMLPBlock(nn.Module):
    def __init__(self, in_chans=512, emb_chans=64):
        super().__init__()
        # spatial deformable proj
        self.sdp = DeformableProjEmbed(in_chans=in_chans, emb_chans=emb_chans)
        self.h_mlp = DeformableMLP(emb_chans, emb_chans, (1, 3), 1, 0)
        self.w_mlp = DeformableMLP(emb_chans, emb_chans, (3, 1), 1, 0)
        self.c_mlp = nn.Linear(emb_chans, emb_chans)
        self.proj = nn.Linear(emb_chans, emb_chans)

    def forward(self, x):
        x = self.sdp(x)
        # B, C, H, W = x.shape
        h = self.h_mlp(x).permute(0, 2, 3, 1)
        w = self.w_mlp(x).permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 1)
        x = x + h + w
        c = self.c_mlp(x)
        x = x + c
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d=256):
        super(Affinity, self).__init__()
        self.d = d

        self.fc_M = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)

        )

        # self.project_sr = nn.Linear(256, 256,bias=False)
        # self.project_tg = nn.Linear(256, 256,bias=False)
        self.project_sr = nn.Linear(256, 256,bias=False)
        self.project_tg = nn.Linear(256, 256,bias=False)
        self.reset_parameters()


    def reset_parameters(self):

        for i in self.fc_M:
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, std=0.01)
                nn.init.constant_(i.bias, 0)


        nn.init.normal_(self.project_sr.weight, std=0.01)
        nn.init.normal_(self.project_tg.weight, std=0.01)

        # The common GM design doesn;t work!!
        # stdv = 1. / math.sqrt(self.d)
        # self.A.data.uniform_(-stdv, stdv)
        # self.A.data += torch.eye(self.d).cuda()
        # nn.init.normal_(self.project_2.weight, std=0.01)
        # nn.init.normal_(self.project2.weight, std=0.01)
        # nn.init.constant_(i.bias, 0)
    def forward(self, X, Y):

        X = self.project_sr(X)
        Y = self.project_tg(Y)

        N1, C = X.size()
        N2, C = Y.size()

        X_k = X.unsqueeze(1).expand(N1, N2, C)
        Y_k = Y.unsqueeze(0).expand(N1, N2, C)
        M = torch.cat([X_k, Y_k], dim=-1)
        M = self.fc_M(M).squeeze()

        # The common GM design doesn;t work!!

        # M = self.affinity_pred(M[None,]).squeeze()
        # M_r = self.fc_M(M_r).squeeze()
        # M = torch.matmul(X, (self.A + self.A.transpose(0, 1).contiguous()) / 2)
        # M = torch.matmul(M, Y.transpose(0, 1).contiguous())


        return M

class dot_attention(nn.Module):

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, dropout=0.0, version='v2'):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.version  = version

    def forward(self, key, value, query, attn_mask=None):

        if self.version == 'v2':
            B =1
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            query = query.unsqueeze(1)
            residual = query
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(key.size(0), B * num_heads, dim_per_head).transpose(0,1)
            value = value.view(value.size(0), B * num_heads, dim_per_head).transpose(0,1)
            query = query.view(query.size(0), B * num_heads, dim_per_head).transpose(0,1)

            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            # (query, key, value, scale, attn_mask)
            context = context.transpose(0, 1).contiguous().view(query.size(1), B, dim_per_head * num_heads)
            output = self.linear_final(context)
            # dropout
            output = self.dropout(output)
            output = self.layer_norm(residual + output)
            # output = residual + output

        elif self.version == 'v1': # some difference about the place of torch.view fuction
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            query = query.unsqueeze(0)
            residual = query
            B, L, C = key.size()
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            batch_size = key.size(0)

            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)

            if attn_mask:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)
            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            context = context.view(batch_size, -1, dim_per_head * num_heads)
            output = self.linear_final(context)
            output = self.dropout(output)
            output = self.layer_norm(residual + output)

        return output.squeeze(), attention.squeeze()