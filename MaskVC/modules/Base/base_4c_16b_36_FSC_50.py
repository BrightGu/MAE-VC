import numpy as np
import torch
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Dropout, Conv1d, MultiheadAttention
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)

def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def pad_layer_2d(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size
    if kernel_size[0] % 2 == 0:
        pad_lr = [kernel_size[0]//2, kernel_size[0]//2 - 1]
    else:
        pad_lr = [kernel_size[0]//2, kernel_size[0]//2]
    if kernel_size[1] % 2 == 0:
        pad_ud = [kernel_size[1]//2, kernel_size[1]//2 - 1]
    else:
        pad_ud = [kernel_size[1]//2, kernel_size[1]//2]
    pad = tuple(pad_lr + pad_ud)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def pixel_shuffle_1d(inp, scale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out

def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x_up

def flatten(x):
    out = x.contiguous().view(x.size(0), x.size(1) * x.size(2))
    return out

def concat_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, c_channels]
    cond = cond.unsqueeze(dim=2)
    cond = cond.expand(*cond.size()[:-1], x.size(-1))
    out = torch.cat([x, cond], dim=1)
    return out

def append_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, x_channels * 2]
    p = cond.size(1) // 2
    mean, std = cond[:, :p], cond[:, p:]
    out = x * std.unsqueeze(dim=2) + mean.unsqueeze(dim=2)
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

class Conv1D_Norm_Act(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, act_fn=None):
        super(Conv1D_Norm_Act, self).__init__()
        self.act_fn = act_fn
        self.conv_block = nn.ModuleList()
        # self.conv_block.append(nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding,padding_mode="replicate"))
        self.conv_block.add_module("conv0", weight_norm(nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                                                                  stride=stride, padding=padding)))
        # self.conv_block.add_module("conv1",nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding))
        self.conv_block.apply(init_weights)
    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        if self.act_fn is not None:
            x = F.relu(x)
        return x

    def remove_weight_norm(self):
        for l in self.conv_block:
            remove_weight_norm(l)

class Block_Unit(nn.Module):
    def __init__(self, c_in, c_out, act_fn=None, norm="", dropout=0.2):
        super(Block_Unit, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.conv_block1 = Conv1D_Norm_Act(c_in, c_out, 3, 1, 1, act_fn)
        self.conv_block2 = Conv1D_Norm_Act(c_out, c_out, 3, 1, 1, act_fn)
        self.dropout = Dropout(dropout)
        self.adjust_dim_layer = weight_norm(nn.Conv1d(c_in, c_out, kernel_size=1, stride=1, padding=0))
        self.adjust_dim_layer.apply(init_weights)
        self.norm = None
        if norm == "IN":
            self.norm = nn.InstanceNorm1d(c_out)

    def forward(self, x):
        out1 = self.conv_block2(F.relu(self.conv_block1(x)))
        out1 = self.dropout(out1)
        if self.c_in != self.c_out:
            x = self.adjust_dim_layer(x)
        x = out1 + x
        if self.norm:
            x = self.norm(x)
        return x

    def remove_weight_norm(self):
        self.conv_block1.remove_weight_norm()
        self.conv_block2.remove_weight_norm()
        remove_weight_norm(self.adjust_dim_layer)

class Smoother(nn.Module):
    """Convolutional Transformer Encoder Layer"""
    def __init__(self, d_model: int, nhead: int, d_hid: int, dropout=0.1):
        super(Smoother, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.conv0 = weight_norm(Conv1d(d_model, d_hid, 3, padding=1))
        self.conv1 = weight_norm(Conv1d(d_hid, d_model, 3, padding=1))
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.conv0.apply(init_weights)
        self.conv1.apply(init_weights)

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # multi-head self attention
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        # add & norm
        src = src + self.dropout1(src2)  # len,B,dim
        # src = self.act(src)
        src2 = src.transpose(0, 1).transpose(1, 2)  # B,dim,len
        src2 = self.conv1(F.relu(self.conv0(src2)))  # B,dim, len
        # add & norm
        src2 = src2.transpose(1, 0).transpose(2, 0)
        src = src + self.dropout2(src2)  # len,B,dim
        return src

    def remove_weight_norm(self):
        remove_weight_norm(self.conv0)
        remove_weight_norm(self.conv1)


class Common(nn.Module):
    def __init__(self, c_in, c_hid, c_out):
        super(Common, self).__init__()
        self.pre_block = nn.Sequential(
            Block_Unit(c_in, c_hid),
            Block_Unit(c_hid, c_hid)
        )
        self.smoothers = nn.Sequential(
            Smoother(c_hid, 8, 512),
            # Smoother(c_hid, 8, 512),
        )  # len,B,dim
        self.post_block = nn.Sequential(
            Block_Unit(c_hid, c_hid),
            Block_Unit(c_hid, c_out)
        )

    # B len dim
    def forward(self, x1):
        x1 = x1.transpose(1, 2)
        x1 = self.pre_block(x1)  # B dim len
        x1 = x1.transpose(1, 2).transpose(1, 0)  # len b dim
        x1 = self.smoothers(x1)
        x1 = x1.transpose(1, 2).transpose(2, 0)  # b dim len
        x1 = self.post_block(x1)  # b dim len
        x1 = x1.transpose(1, 2)
        return x1

    def remove_weight_norm(self):
        for l in self.pre_block:
            l.remove_weight_norm()
        for l in self.post_block:
            l.remove_weight_norm()
        for l in self.smoothers:
            l.remove_weight_norm()

class ModuUnit(nn.Module):
    def __init__(self, dim=64,eps = 1e-5):
        super(ModuUnit, self).__init__()
        self.eps = eps
        self.dnorm = nn.InstanceNorm1d(dim) # b dim len

    def forward(self, x, spk_emb): # B len dim
        spk_emb = spk_emb.transpose(1,2) # B 128 1
        y_dmn = spk_emb[:,:spk_emb.shape[1]//2,:]
        y_dsd = spk_emb[:,spk_emb.shape[1]//2:,:]
        x = x.transpose(1,2) # B dim len
        x = self.dnorm(x)*y_dsd+y_dmn # B dim len
        x = x.transpose(1,2) # B len dim
        return x

class Handler(nn.Module):
    def __init__(self, c_in, c_hid, c_out,nhead = 2):
        super(Handler, self).__init__()
        self.pre_block = nn.Sequential(
            Block_Unit(c_in, c_hid),
            # Block_Unit(c_hid, c_hid),
        )
        self.smoothers = nn.Sequential(
            # Smoother(c_hid, 4, 512),
            Smoother(c_hid, nhead, 256)
        )  # len,B,dim

        self.post_block = nn.Sequential(
            Block_Unit(c_hid, c_out,nn.Sigmoid()),
            # Block_Unit(c_out, c_out),
        )

    # B len dim
    def forward(self, x1):
        x1 = x1.transpose(1, 2)
        x1 = self.pre_block(x1)  # B dim len
        x1 = x1.transpose(1, 2).transpose(1, 0)  # len b dim
        x1 = self.smoothers(x1)
        x1 = x1.transpose(1, 2).transpose(2, 0)  # b dim len
        x1 = self.post_block(x1)  # b dim len
        out = x1.transpose(1, 2) # b len dim
        return out

    def remove_weight_norm(self):
        for l in self.pre_block:
            l.remove_weight_norm()
        for l in self.post_block:
            l.remove_weight_norm()
        for l in self.smoothers:
            l.remove_weight_norm()

class ModuBlock_mask(nn.Module):
    def __init__(self,c_in,c_hid,c_out,nhead = 1):
        super(ModuBlock_mask, self).__init__()

        self.handler = Handler(c_in, c_hid, c_out,nhead)
        self.sim_block = SimilarityPredict(c_out,c_hid,nhead)
        self.speak_line = nn.Linear(256,c_out*2)
        self.modu = ModuUnit(c_out)

        self.mean_line1 = nn.Linear(c_out, c_out)
        self.mean_line2 = nn.Linear(c_out, 1)

        self.std_line1 = nn.Linear(c_out, c_out)
        self.std_line2 = nn.Linear(c_out, 1)

        self.Lnorm = nn.InstanceNorm1d(128)
    def forward(self, x,spk_emb,mask_rate):
        lat_spk_emb = self.speak_line(spk_emb)
        lat_x = self.handler(x)
        Lmean = self.mean_line2(torch.tanh(self.mean_line1(lat_x)))
        Lstd = self.std_line2(torch.tanh(self.std_line1(lat_x)))
        mask_x = self.sim_block(lat_x,mask_rate)
        out = self.modu(mask_x,lat_spk_emb)
        out = self.Lnorm(out) * Lstd + Lmean
        return out

    def remove_weight_norm(self):
        self.handler.remove_weight_norm()
        self.sim_block.remove_weight_norm()


class Generator(nn.Module):
    def __init__(self, c_in, c_hid, c_out):
        super(Generator, self).__init__()
        self.pre_block = nn.Sequential(
            Block_Unit(c_in, c_hid),
            Block_Unit(c_hid, c_hid, nn.ReLU())
        )
        self.smoothers = nn.Sequential(
            Smoother(c_hid, 8, 512),
        )  # len,B,dim

        self.post_block = nn.Sequential(
            Block_Unit(c_hid, c_hid,nn.ReLU()),
            Block_Unit(c_hid, c_out),
            Block_Unit(c_out, c_out,nn.ReLU())
        )
    # B len dim
    def forward(self, x1):
        x1 = x1.transpose(1, 2)
        x1 = self.pre_block(x1)  # B dim len
        x1 = x1.transpose(1, 2).transpose(1, 0)  # len b dim
        x1 = self.smoothers(x1)
        x1 = x1.transpose(1, 2).transpose(2, 0)  # b dim len
        x1 = self.post_block(x1)  # b dim len
        x1 = x1.transpose(1, 2)
        return x1

    def remove_weight_norm(self):
        for l in self.pre_block:
            l.remove_weight_norm()
        for l in self.post_block:
            l.remove_weight_norm()
        for l in self.smoothers:
            l.remove_weight_norm()


class Quantify(torch.autograd.Function):
    @staticmethod
    def forward(self, mask_map,mask_rate):
        self.save_for_backward(mask_map)
        mask_map_detach = mask_map.clone().detach().cpu()
        k = torch.tensor(np.percentile(mask_map_detach, mask_rate, axis=1)).cuda()
        # mask_map_detach = mask_map
        # k = torch.tensor(np.percentile(mask_map_detach, mask_rate,axis=1))
        k = k.unsqueeze(2)
        q_mask_map = torch.where(mask_map > k, 1, 0) # B len 1
        return q_mask_map

    @staticmethod
    def backward(self, grad_output):
        mask_map = self.saved_tensors
        grad = grad_output.clone()*mask_map
        return grad

class Sim_Block(nn.Module):
    def __init__(self, c_in, c_hid, c_out,nhead=1):
        super(Sim_Block, self).__init__()
        self.pre_block = nn.Sequential(
            Block_Unit(c_in, c_hid),
            Block_Unit(c_hid, c_hid),
        )
        self.smoothers = nn.Sequential(
            Smoother(c_hid, nhead, 512),
        )  # len,B,dim
        self.post_block = nn.Sequential(
            Block_Unit(c_hid, c_hid),
            Block_Unit(c_hid, c_out),
        )

    # B len dim
    def forward(self, x1):
        x1 = x1.transpose(1, 2)
        x1 = self.pre_block(x1)  # B dim len
        x1 = x1.transpose(1, 2).transpose(1, 0)  # len b dim
        x1 = self.smoothers(x1)
        x1 = x1.transpose(1, 2).transpose(2, 0)  # b dim len
        x1 = self.post_block(x1)  # b dim len
        x1 = x1.transpose(1, 2)
        return x1

    def remove_weight_norm(self):
        for l in self.pre_block:
            l.remove_weight_norm()
        for l in self.post_block:
            l.remove_weight_norm()
        for l in self.smoothers:
            l.remove_weight_norm()

class SimilarityPredict(nn.Module):
    def __init__(self, c_in,c_hid,nhead=1):
        super(SimilarityPredict, self).__init__()
        # c_in, c_hid, c_out, nhead = 1
        self.pre_sim_block = Sim_Block(c_in,c_hid,1,nhead)
        self.quantify = Quantify.apply

    # B len dim to B len 1
    def forward(self, x,mask_rate):
        # x = x.transpose(1, 2)  # B dim len
        x_shift = torch.cat([x[:,-2:-1,:],x[:,0:-1:,]],dim = 1) # B len dim
        x_hid = self.pre_sim_block(x)
        x_shift_hid = self.pre_sim_block(x_shift) # B len 1
        mask_pre = x_hid - x_shift_hid
        mask_pre = torch.sigmoid(torch.abs(mask_pre)) # B len 1
        mask = self.quantify(mask_pre,mask_rate)
        out = x*mask
        return out

    def remove_weight_norm(self):
        self.pre_sim_block.remove_weight_norm()



class SpeakerEncoder(nn.Module):
    def __init__(self, c_in=80, c_h=128, c_out=256, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=80,
            n_conv_blocks=6, n_dense_blocks=6,
            subsample=[1, 2, 2, 2, 2, 2], act='relu', dropout_rate=0):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = weight_norm(nn.Conv1d(in_channels, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([weight_norm(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([weight_norm(nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub))
            for sub, _ in zip(subsample, range(n_conv_blocks))])

        self.pooling_layer = nn.AdaptiveAvgPool1d(1)

        self.dense_layers = nn.ModuleList([weight_norm(nn.Conv1d(c_h, c_h, kernel_size=3)) for _ \
                in range(n_dense_blocks)])
        self.output_layer = weight_norm(nn.Conv1d(c_h, c_out, kernel_size=3))

        self.dropout_layer = nn.Dropout(p=dropout_rate)

        self.mean_layer = nn.Linear(c_out, c_out // 2)
        self.std_layer = nn.Linear(c_out, c_out // 2)

        self.in_conv_layer.apply(init_weights)
        self.first_conv_layers.apply(init_weights)
        self.second_conv_layers.apply(init_weights)
        self.dense_layers.apply(init_weights)
        self.output_layer.apply(init_weights)

    def conv_blocks(self, inp):
        out = inp
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        # dense layers
        for l in range(self.n_dense_blocks):
            y = pad_layer(out, self.dense_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x):
        x = x.transpose(1, 2)
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer) # 5 128 len
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)# dim 不变，把len从346变为44.  B dim len
        # avg pooling
        out = self.dense_blocks(out) # B dim len
        out = pad_layer(out, self.output_layer)
        out = self.pooling_layer(out)
        out = out.transpose(1, 2)
        mean = self.mean_layer(out)# B 1 64
        std = self.std_layer(out)# B 1 64
        out = torch.cat([mean,std],dim=2)
        return out  # 5 128

    def remove_weight_norm(self):
        remove_weight_norm(self.in_conv_layer)
        remove_weight_norm(self.output_layer)
        for l in self.first_conv_layers:
            remove_weight_norm(l)
        for l in self.second_conv_layers:
            remove_weight_norm(l)
        for l in self.dense_layers:
            remove_weight_norm(l)

class MagicModel(nn.Module):
    def __init__(self, d_model=256):
        super(MagicModel, self).__init__()
        self.common = Common(80,256,80)
        self.spk_en = SpeakerEncoder()
        self.modu1_mask = ModuBlock_mask(80, 36, 16,nhead =6)
        self.modu2_mask = ModuBlock_mask(16, 36, 16,nhead =6)
        self.modu3_mask = ModuBlock_mask(16, 36, 16,nhead =6)
        self.modu4_mask = ModuBlock_mask(16, 36, 80,nhead =6)
        self.ge = Generator(80,256,80)

    def forward(self, x1,x2):
        x1 = self.common(x1)
        x2 = self.common(x2)
        spk_emb = self.spk_en(x2) # B 128 1

        lat1 = self.modu1_mask(x1, spk_emb, 50)
        lat2 = self.modu2_mask(lat1, spk_emb, 50)
        lat3 = self.modu3_mask(lat2, spk_emb, 50)
        lat4 = self.modu4_mask(lat3, spk_emb, 50)
        out = self.ge(lat4)
        return out

    def infer(self, x1,x2):
        x1 = self.common(x1)
        x2 = self.common(x2)
        spk_emb = self.spk_en(x2) # B 128 1

        lat1 = self.modu1_mask(x1,spk_emb,50)
        lat2 = self.modu2_mask(lat1,spk_emb,50)
        lat3 = self.modu3_mask(lat2,spk_emb,50)
        lat4 = self.modu4_mask(lat3,spk_emb,50)
        out = self.ge(lat4)
        return out

    def remove_weight_norm(self):
        self.common.remove_weight_norm()
        self.spk_en.remove_weight_norm()
        self.modu1_mask.remove_weight_norm()
        self.modu2_mask.remove_weight_norm()
        self.modu3_mask.remove_weight_norm()
        self.modu4_mask.remove_weight_norm()
        self.ge.remove_weight_norm()


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if
                  "auxiliary" not in name) / 1e6


if __name__ == '__main__':
    x1 = torch.randn([5, 127, 80])
    x2 = torch.randn([5, 257, 80])
    # word_emb = torch.randn([5, 137, 80])
    magic = MagicModel()
    magic.remove_weight_norm()
    print("model params:", count_parameters_in_M(magic))
    print("ge params:", count_parameters_in_M(magic.ge))
    print("common params:", count_parameters_in_M(magic.common))
    print("spk_en params:", count_parameters_in_M(magic.spk_en))


    out = magic(x1,x2)
    print("out:", out.shape)









