"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 默认为 ViT-B
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 224/16=14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 14*14

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)   # 3，768，16，16
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # print("x1", x.shape)                          # [B, 10, 27, 27]

        # ViT 模型的传入图片的大小是固定的
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW], 从第二个位置展平
        # transpose: [B, C, HW] -> [B, HW, C]
        # 调换位置，变成  [batch_size, token，dimension]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # print("x2", x.shape)                         # [B, 36, 64]
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5         # 指 根号下维度分支一
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 可以用三个全连接层分别得到 qkv
        self.attn_drop = nn.Dropout(attn_drop_ratio)      # 
        self.proj = nn.Linear(dim, dim)                   # 对 concat 拼接的head 传入全连接层
        self.proj_drop = nn.Dropout(proj_drop_ratio)      # 

    def forward(self, x):
        # print("x3", x.shape)                              # [64, 37, 64]
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 为了后面进行计算，不太好理解，可以好好看看
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print("qkv", qkv.shape)                           # [3, 64, 4, 37, 16]
        # 通过切片的方式，得到 q，k，v
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print("q", q.shape)                               # [64, 4, 37, 16]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        # 多维数据的矩阵乘法，只对最后两位进行操作
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print("attn1", attn.shape)                        # [64, 4, 37, 37]
        # 对每行进行softmax 处理
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print("attn2", attn.shape)                        # [64, 4, 37, 37]

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("x4", x.shape)                              # [64, 37, 64]
        x = self.proj(x)
        # print("x5", x.shape)                              # [64, 37, 64]
        x = self.proj_drop(x)
        # print("x6", x.shape)                              # [64, 37, 64]
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,      # mlp 升维的倍数
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # print("x7", x.shape)                                  # [64, 37, 64]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # print("x8", x.shape)      #11 [64, 5, 64],            # [64, 37, 64]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x9", x.shape)                                  # [64, 37, 64]
        # x = x + self.drop_path(self.gcn(self.norm2(x)))
        return x

# 注意这里没有使用 distilled
class VisionTransformer_wo_token(nn.Module):
    def __init__(self, img_size=29, patch_size=4, in_c=5, num_classes=16,
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):

        super(VisionTransformer_wo_token, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.num_tokens = 2 if distilled else 1

        # partial
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 制作patch
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # class_token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        # 位置编码， 1D 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 构建等差序列，构建 drop_path
        # analysis  默认为0， 可以试试0.1， 0.2
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        # 堆叠 Encoder Block
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None

    def forward_features(self, x):
        # print("forward features shape", x.shape)               #  [128, 1, 10, 25, 25]
        x = torch.squeeze(x)
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        # print("x10", x.shape)                                  # [B,36,64]
        # [1, 1, 768] -> [B, 1, 768]

        x = self.pos_drop(x + self.pos_embed)
        # print("x11", x.shape)                                  # [B, 37, 64]
        x = self.blocks(x)
        # print("x12", x.shape)                                  # [B, 37, 64]
        x = self.norm(x)
        # print("x13", x.shape)                                  # [B, 37, 64]


        x = self.avgpool(x.transpose(1, 2)) 
        # print("self.avgpool ", x.shape)                        # [B, 64, 1]

        x = torch.flatten(x, 1)
        # print("flatten ", x.shape)                             # [B, 64]
        return x


    def forward(self, x):
        # print("input shape", x.shape)
        x = self.forward_features(x)
        # print("x_forward", x.shape)                     # 64, 64
        ##这里不执行###########################
        if self.head_dist is not None:                  # default self.head_dist = False
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # 直接走到这#######################
            x = self.head(x)
            # print("x_head", x.shape)                              # [B, 17])
        return x


if __name__ == "__main__":
    temp = torch.rand((128, 32, 27, 27))

    net = VisionTransformer_wo_token(img_size=27, patch_size=4, in_c=32, num_classes=17,
                embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0)
    # print(net)
    feature = net(temp)
    print(feature.shape)
