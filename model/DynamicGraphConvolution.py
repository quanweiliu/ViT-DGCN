import torch
import torch.nn as nn

class DynamicGraphConvolution(nn.Module):
    # 这是正确的，需要记住
    # [B, vector_length, class_num] 分别为 batch_size, 向量长度， 向量维度
    # 一维卷积（in_channels, out_channels）在最后一维上进行计算， 随意按照上面的逻辑，需要先用permute 函数反转 1，2 维
    # 得到[B, class_num, vector_length], 然后，得到
    # batch_size * out_channels * vector_length
    # 如果不反转，就直接计算维度上的信息了, 得到
    # batch_size * out_channels * class_num
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()
        self.num_nodes = num_nodes                               # num_nodes = class_num
        

        # 使用这个得到邻接矩阵 A, 好好分析一下为什么邻接矩阵可以这样表示？
        # 发现：这个邻接矩阵不是 n*n 的，而是 n*c 的

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        # 前向传播的全连接层，也就是 W
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        """
            - 4, 16, 1024 * 1024, 1024
            - 所谓静态图就是全连接网络
        """
        # print("num_nodes", self.num_nodes)                      # class_num
        # print("x6 ", x.shape)                                   # ([B, 1024, class_num])
        # 首先和邻接矩阵相乘
        x = self.static_adj(x.transpose(1, 2))                    #  B * class_num * class_num @ class_num * 1024
        # print("static adj", x.shape)                            # ([B, class_num, 1024])
        # 然后和权重相乘
        x = self.static_weight(x.transpose(1, 2))                 # 1024 * 1024 @ 1024 * class_num
        # print("static weight", x.shape)                         # ([B, 1024, class_num])
        return x

    # 这里值得好好研究一下，惊喜！！
    def forward_construct_dynamic_graph(self, x):
        # print("x7", x.shape)                                      # ([B, 1024, class_num])
        ### Model global representations ###
        x_glb = self.gap(x)
        # print("gap", x_glb.shape)                                 # ([B, 1024, 1])
        x_glb = self.conv_global(x_glb)                           # B * 1024 * 1024 @ 1024 * 1    
        # print("conv_global", x_glb.shape)                         # ([B, 1024, 1])
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        # print("expand ", x_glb.shape)                             # ([B, 1024, class_num])
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        # print("x8", x.shape)                                      # ([B, 2048, class_num])
        dynamic_adj = self.conv_create_co_mat(x)                  # B * class_num * 2048 @ 2048 * class_num
        # print("dynamic_adj1", dynamic_adj.shape)                  # ([B, class_num, class_num])
        dynamic_adj = torch.sigmoid(dynamic_adj)                  
        # print("dynamic_adj2", dynamic_adj.shape)                  # ([B, class_num, class_num])
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        # print("x", x.shape, "dynamic_adj", dynamic_adj.shape)   # ([B, 1024, class_num]) ([4, class_num, class_num])
        x = torch.matmul(x, dynamic_adj)
        # print("x9", x.shape)                                      # ([B, 1024, class_num])
        x = self.relu(x)
        x = self.dynamic_weight(x)
        # print("x10", x.shape)                                     # ([B, 1024, class_num])

        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        - 动态图是加入注意力机制的全连接网络
        """
        # print("DynamicGraphConvolution_input", x.shape)            # ([4, 1024, 16])
        out_static = self.forward_static_gcn(x)                      # 
        # print('static output', out_static.shape)                   # ([4, 1024, 16])
        x = x + out_static  # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        # print("dynamic_adj" , dynamic_adj.shape)                   # ([4, 16, 16])
        x = self.forward_dynamic_gcn(x, dynamic_adj)          
        # print('dynamic output', x.shape)                           # ([4, 1024, 16])
        return x








