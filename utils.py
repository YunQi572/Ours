import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_dense_adj, get_laplacian

#设置随机种子
def set_seed(seed):
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

#生成器类
class Generator(nn.Module):

    def __init__(self, noise_dim, input_dim, output_dim, dropout):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(output_dim, output_dim)

        hid_layers = []
        dims = [noise_dim + output_dim, 64, 128, 256]
        for i in range(len(dims) - 1):
            d_in = dims[i]
            d_out = dims[i + 1]
            hid_layers.append(nn.Linear(d_in, d_out))
            #与Ghost不同的地方(添加了一个批归一化层)
            hid_layers.append(nn.BatchNorm1d(d_out))  # 添加BN层
            
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p = dropout, inplace = False))
        self.hid_layers = nn.Sequential(* hid_layers)
        self.nodes_layer = nn.Linear(256, input_dim)
    
    def forward(self, z, c):
        #标签嵌入 
        label_emb = self.emb_layer.forward(c)    
        #拼接噪声和标签嵌入
        z_c = torch.cat((label_emb, z), dim = -1)
        #通过隐藏层
        hid = self.hid_layers(z_c)
        #生成最终节点特征
        node_logits = self.nodes_layer(hid)
        return node_logits

# 链接预测器类，基于两层MLP架构
class LinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super(LinkPredictor, self).__init__()
        # 第一层：将两个节点特征连接后映射到隐藏维度
        self.layer1 = nn.Linear(2 * input_dim, hidden_dim)  # 2倍输入维度，因为我们连接两个节点的特征
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        # 第二层(输出)：将隐藏表示映射到单一输出，表示两节点之间存在连接的概率
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_i, x_j):
        # 将两个节点的特征连接起来
        x = torch.cat([x_i, x_j], dim=-1)
        # 通过第一层
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        # 通过第二层，得到预测的边存在概率
        x = self.layer2(x)
        # 应用sigmoid函数，得到0~1之间的概率
        x = self.sigmoid(x)
        return x
    
    def predict_links(self, node_features, threshold=0.5):
        """
        基于节点特征预测图的邻接矩阵
        
        参数:
        - node_features: 节点特征矩阵，形状为[num_nodes, feature_dim]
        - threshold: 预测边存在的概率阈值
        
        返回:
        - adj_matrix: 预测的邻接矩阵，形状为[num_nodes, num_nodes]
        """
        num_nodes = node_features.shape[0]
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=node_features.device)
        
        # 对每一对节点进行预测
        for i in range(num_nodes):
            # 获取当前节点特征并扩展维度以便广播
            x_i = node_features[i].unsqueeze(0).expand(num_nodes, -1)
            # 获取所有其他节点的特征
            x_j = node_features
            # 预测链接概率
            probs = self.forward(x_i, x_j).squeeze()
            # 根据阈值确定边的存在性
            edges = (probs > threshold).float()
            # 更新邻接矩阵
            adj_matrix[i] = edges
            
        return adj_matrix
    
#根据生成器生成的节点特征和相似度进行图的构建
def construct_graph(node_logits, link_predictor=None, threshold=0.5):
    """
    使用生成的节点特征和链接预测器构建图
    
    参数:
    node_logits: 生成的节点伪特征矩阵
    link_predictor: 链接预测器模型，如果为None则使用余弦相似度
    threshold: 确定边存在的概率阈值
    
    返回:
    adjacency_matrix: 构建的邻接矩阵
    """
    # 如果提供了链接预测器，使用它来预测边
    if link_predictor is not None:
        adjacency_matrix = link_predictor.predict_links(node_logits, threshold)
    else:
        # 使用余弦相似度计算节点间相似度
        node_features_norm = torch.nn.functional.normalize(node_logits, p=2, dim=1)
        adj_logits = torch.mm(node_features_norm, node_features_norm.t())
        # 根据阈值确定边的存在性
        adjacency_matrix = (adj_logits > threshold).float()
    
    # 移除自环（对角线元素置零）
    adjacency_matrix.fill_diagonal_(0)
    
    return adjacency_matrix

#深度特征正则化的前向钩子类
class DeepInversionHook:
    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # nch = input[0].shape[1]     #输入特征的通道数
        # mean = input[0].mean([0, 2, 3])     # 计算当前批次特征图的均值（沿批次、高度和宽度维度）
        # var = input[0].permute(1, 0, 2, 3).contigous().view([nch, -1]).var(1, unbiased = False)     #计算每个通道的方差

        # input[0] shape: [batch_size, features]
        mean = input[0].mean(0)
        var = input[0].var(0, unbiased = False)

        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else: #动量平滑后的均值和方差
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data) 

    def remove(self):
        self.hook.remove()   
 
#低频蒸馏损失函数，传入的输出应该是经过softmax层后
def edge_distribution_low(edge_idx, student_out, teacher_out):
    edge_idx = edge_idx.detach()
    src_nodes = edge_idx[0]         #起点集合
    dst_nodes = edge_idx[1]         #终点集合
    # criterion = nn.KLDivLoss(reduction = "batchmean", log_target = True)
    criterion = nn.KLDivLoss(reduction = "none", log_target = False)
    print(f"edge_distribution_low edge_idx : f{edge_idx}")
    # 计算所有边的损失
    # total_loss = 0
    # total_loss = torch.tensor(0.0, device=student_out.device)
    losses = []
    unique_src = torch.unique(src_nodes)
    edge_num = 0        #边的个数
    for src in unique_src:
        # 找到当前节点的所有邻接边的终点
        neighbors_mask = (src_nodes == src)
        dst = dst_nodes[neighbors_mask]
        
        if len(dst) > 0:
            # 计算当前节点与其所有邻接点之间的损失
            src_out = teacher_out[src]
            for dst_node in dst:
                dst_out = student_out[dst_node]     
                # total_loss += criterion(dst_out, src_out).sum()
                loss = criterion(dst_out, src_out).sum()
                losses.append(loss)
            edge_num += len(dst)

    # # 如果没有边，返回零损失
    # if len(unique_src) == 0:
    #     return torch.tensor(0.0, device=student_out.device)
    
    if len(losses) == 0: 
        return torch.tensor(0.0, device=student_out.device)
    total_loss = torch.stack(losses).sum() / edge_num
    # 返回平均损失
    # return total_loss / edge_num
    return total_loss

#Average Accuary 平均准确率
def AA(M_acc, T = None):        
    """
    M_acc[i, j] 第i个任务训练完成后, 在第 j 个任务上的准确率
    """
    if T is None:
        T = M_acc.size(0)
    ret = 0
    for i in range(0, T):
        ret += M_acc[T - 1, i]       #训练了T个任务，最后一个任务的编号是 T - 1
    ret /= T
    return ret

#Average Forgetting 平均遗忘率
def AF(M_acc, T = None):
    if T is None:
        T = M_acc.size(0)
    if T == 1:                  #第一个任务
        return -1
    ret = 0
    for i in range(0, T - 1):
        forgetting = M_acc[i, i] - M_acc[T - 1, i]
        ret += forgetting
    ret /= T - 1
    return ret

#计算图的拉普拉斯能量分布（LED）
def compute_led(graph_data):
    """
    参数:
    graph_data: 图数据对象 Data(x, edge_index, y)
    返回:
    energy_distribution: 拉普拉斯能量分布 [N,]
    """
    nodes_feature = graph_data.x  # [N, d] 节点特征矩阵
    edge_index = graph_data.edge_index
    num_nodes = nodes_feature.shape[0]
        
    # 计算标准化拉普拉斯矩阵 L = I - D^(-1/2) A D^(-1/2)
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index, 
        num_nodes=num_nodes, 
        normalization='sym'  # 对称标准化
    )
        
    # 转换为稠密矩阵
    L = to_dense_adj(edge_index_laplacian, edge_attr=edge_weight_laplacian, max_num_nodes=num_nodes)[0]
    # 特征值分解，获取特征向量矩阵 U
    eigenvalues, eigenvectors = torch.linalg.eigh(L)  # U: [N, N]
    U = eigenvectors  # 特征向量矩阵
    # 计算图傅里叶变换 \hat{X} = U^T X
    X_hat = torch.matmul(U.T, nodes_feature)  # [N, d] 傅里叶变换后的特征
    # 计算每个频率分量的能量（所有特征维度的平方和）
    energy_per_freq = torch.sum(X_hat ** 2, dim=1)  # [N,] 每个频率的能量
    # 计算总能量
    total_energy = torch.sum(energy_per_freq)
        
    # 计算能量分布（归一化）  \bar{x}_n = \frac{\hat{x}_n^2}{\sum_{i=1}^N \hat{x}_i^2}
    if total_energy > 0:
        energy_distribution = energy_per_freq / total_energy  # [N,] 归一化的能量分布
    else:
        energy_distribution = torch.zeros_like(energy_per_freq)
    
    return energy_distribution
    

#高斯核函数
def gaussian_kernel1(x):        #np
    return (1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2))

def gaussian_kernel(x):
    return (1 / torch.sqrt(torch.tensor(2 * np.pi, device=x.device))) * torch.exp(-0.5 * x ** 2)


#核密度估计函数
def KDE1(x, y, bandwidth, kernel_func):  #np
    n = len(y)
    kernel_values = kernel_func((x - y) / bandwidth)                #计算核函数的值
    density_estimation = np.sum(kernel_values) / (n * bandwidth)    #计算概率密度估计值
    return density_estimation

def KDE(x, y, bandwidth, kernel_func):
    n = y.shape[0]
    kernel_values = kernel_func((x - y) / bandwidth)
    density_estimation = torch.sum(kernel_values) / (n * bandwidth)
    return density_estimation


def apply_kde_to_energy_dist1(energy_dist, bandwidth, eval_nums, device):   #np
    """
    使用KDE将能量分布转换为概率分布
    
    参数:
    energy_dist: 能量分布 [N,]
    bandwidth: KDE带宽参数
    eval_nums: 评估点的数量，决定输出概率分布的维度
    device: 计算设备
    
    返回:
    prob_dist: 概率分布 [eval_nums,]
    """
    # 确保输入为tensor且在正确设备上
    if isinstance(energy_dist, torch.Tensor):
        energy_dist = energy_dist.detach().cpu().numpy()
    
    # 归一化能量分布为概率权重(多余)
    # energy_dist = energy_dist / torch.sum(energy_dist)
    
    # n = len(energy_dist)  # 原始能量分布的长度
    
    # 创建评估点网格（频率域的归一化位置），数量由eval_nums决定
    x_eval = torch.linspace(0, 1, eval_nums, device=device).cpu().numpy()
    # prob_dist = torch.zeros(eval_nums, device=device)
    
    # # 数据点位置（频率索引归一化）
    # y_data = torch.linspace(0, 1, n, device=device).cpu().numpy()
    # weights = energy_dist.cpu().numpy()
    
    # # 对每个评估点使用KDE计算概率密度
    # for i in range(eval_nums):
    #     x_eval_point = x_eval[i].item()
        
    #     # 使用utils中的KDE函数计算加权概率密度
    #     density = 0.0
    #     # for j in range(n):
    #     #     density += KDE(x_eval_point, y_data[j], bandwidth, gaussian_kernel) * weights[j]
    #     prob_dist[i] = density
    

    # 归一化为概率分布
    # prob_dist = prob_dist / torch.sum(prob_dist)
    
    prob_dist = np.array([KDE(xi, energy_dist, bandwidth, gaussian_kernel) for xi in x_eval])
    
    return prob_dist

def apply_kde_to_energy_dist(energy_dist, bandwidth, eval_nums, device):
    # energy_dist: [N,] torch tensor
    x_eval = torch.linspace(0, 1, eval_nums, device=device)
    prob_dist = torch.stack([KDE(xi, energy_dist, bandwidth, gaussian_kernel) for xi in x_eval])
    prob_dist = prob_dist / torch.sum(prob_dist)
    return prob_dist


def compute_js_divergence_from_prob_dist1(prob_dist_1, prob_dist_2, device):
    """
    直接计算两个概率分布之间的Jensen-Shannon散度
    （假设输入已经是通过KDE处理的概率分布）
    
    参数:
    prob_dist_1: 第一个概率分布 [N1,]
    prob_dist_2: 第二个概率分布 [N2,]
    
    返回:
    js_divergence: Jensen-Shannon散度值
    """
    # 确保输入为tensor且在同一设备上
    # prob_dist_1 = prob_dist_1.to(device)
    # prob_dist_2 = prob_dist_2.to(device)
    prob_dist_1 = torch.tensor(prob_dist_1, dtype=torch.float32).to(device)
    prob_dist_2 = torch.tensor(prob_dist_2, dtype=torch.float32).to(device)

    '''
        # 处理不同长度的分布：填充或截断到相同长度
        max_len = max(len(prob_dist_1), len(prob_dist_2))
        
        if len(prob_dist_1) < max_len:
            # 用零填充较短的分布
            padding = torch.zeros(max_len - len(prob_dist_1), device=device)
            prob_dist_1 = torch.cat([prob_dist_1, padding])
        elif len(prob_dist_1) > max_len:
            prob_dist_1 = prob_dist_1[:max_len]
            
        if len(prob_dist_2) < max_len:
            padding = torch.zeros(max_len - len(prob_dist_2), device=device)
            prob_dist_2 = torch.cat([prob_dist_2, padding])
        elif len(prob_dist_2) > max_len:
            prob_dist_2 = prob_dist_2[:max_len]
        
        # 重新归一化
        prob_dist_1 = prob_dist_1 / torch.sum(prob_dist_1)
        prob_dist_2 = prob_dist_2 / torch.sum(prob_dist_2)
    '''

    # 计算Jensen-Shannon散度
    # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), 其中 M = 0.5(P+Q)
    
    # 计算混合分布 M = 0.5(P + Q)
    M = 0.5 * (prob_dist_1 + prob_dist_2)
    
    # 避免log(0)的情况，添加小的epsilon
    epsilon = 1e-10
    prob_dist_1 = prob_dist_1 + epsilon
    prob_dist_2 = prob_dist_2 + epsilon
    M = M + epsilon
    
    # 计算KL散度: KL(P||M) = sum(P * log(P/M))
    kl_1_m = torch.sum(prob_dist_1 * torch.log(prob_dist_1 / M))
    kl_2_m = torch.sum(prob_dist_2 * torch.log(prob_dist_2 / M))
    
    # Jensen-Shannon散度
    js_divergence = 0.5 * kl_1_m + 0.5 * kl_2_m
    
    return js_divergence.item()


def compute_js_divergence_from_prob_dist(prob_dist_1, prob_dist_2):
    M = 0.5 * (prob_dist_1 + prob_dist_2)
    epsilon = 1e-10
    prob_dist_1 = prob_dist_1 + epsilon
    prob_dist_2 = prob_dist_2 + epsilon
    M = M + epsilon
    kl_1_m = torch.sum(prob_dist_1 * torch.log(prob_dist_1 / M))
    kl_2_m = torch.sum(prob_dist_2 * torch.log(prob_dist_2 / M))
    js_divergence = 0.5 * kl_1_m + 0.5 * kl_2_m
    return js_divergence


#得到生成图与客户端图之间的SC值
def get_SC1(synthetic_data, clients_nodes_num, clients_graph_energy, device, h):    #np
    """
    计算生成图与客户端图之间的SC值（Spectral Consistency）
    
    参数:
    synthetic_data: 生成图，Data(x, y, edge_index)
    clients_nodes_num: 客户端图的节点数量列表 [N1, N2, ..., Nc]
    clients_graph_energy: 客户端的拉普拉斯能量分布集合 [[energy_dist_1], [energy_dist_2], ...]
    
    返回:
    weighted_sc: 按节点数量加权平均的SC值
    """
    # 1. 计算生成图的拉普拉斯能量分布
    synthetic_energy_dist = compute_led(synthetic_data)
    
    # 2. 设置核密度估计的带宽参数h
    # h = getattr(args, 'bandwidth', 0.1)  # 默认带宽为0.1
    
    # 3. 计算与每个客户端图的SC值
    sc_values = []
    total_nodes = sum(clients_nodes_num)
    
    # 使用KDE将拉普拉斯能量分布转换为概率分布
    # 获取所有客户端子图节点数量的最大值
    max_client_nodes = max(clients_nodes_num)

    # 获取生成图 synthetic_data 的节点数量
    synthetic_nodes_num = synthetic_data.x.shape[0]

    # 获取两者中的最大值
    max_nodes_num = max(max_client_nodes, synthetic_nodes_num)
    
    # 3.1 为生成图的能量分布生成概率分布
    synthetic_prob_dist = apply_kde_to_energy_dist(synthetic_energy_dist, h, max_nodes_num, device)

    for i, client_energy_dist in enumerate(clients_graph_energy):
        # 3.2 为当前客户端图的能量分布生成概率分布
        client_prob_dist = apply_kde_to_energy_dist(client_energy_dist, h, max_nodes_num, device)
        
        # 3.3 计算Jensen-Shannon散度
        js_divergence = compute_js_divergence_from_prob_dist(
            synthetic_prob_dist, client_prob_dist, device
        )
        
        # SC = JS散度（值越小表示越相似）
        sc_value = js_divergence
        sc_values.append(sc_value)
    
    # 4. 按客户端图的节点数量进行加权平均
    weighted_sc = 0.0
    for i, sc_value in enumerate(sc_values):
        weight = clients_nodes_num[i] / total_nodes
        weighted_sc += weight * sc_value
    
    return weighted_sc

def get_SC(synthetic_data, clients_nodes_num, clients_graph_energy, device, h):
    synthetic_energy_dist = compute_led(synthetic_data)  # torch tensor
    max_client_nodes = max(clients_nodes_num)
    synthetic_nodes_num = synthetic_data.x.shape[0]
    max_nodes_num = max(max_client_nodes, synthetic_nodes_num)
    synthetic_prob_dist = apply_kde_to_energy_dist(synthetic_energy_dist, h, max_nodes_num, device)

    sc_values = []
    total_nodes = sum(clients_nodes_num)
    for i, client_energy_dist in enumerate(clients_graph_energy):
        client_prob_dist = apply_kde_to_energy_dist(client_energy_dist, h, max_nodes_num, device)
        js_divergence = compute_js_divergence_from_prob_dist(synthetic_prob_dist, client_prob_dist)
        sc_values.append(js_divergence)
    weighted_sc = torch.tensor(0.0, device=device)
    for i, sc_value in enumerate(sc_values):
        weight = clients_nodes_num[i] / total_nodes
        weighted_sc += weight * sc_value
    return weighted_sc  # 现在是 torch tensor，可参与反向传播


#高频
def get_Shigh(synthetic_data):
    """
    计算图的高频分量 S_high
    
    参数:
    synthetic_data: 图数据对象 Data(x, edge_index, y)
    
    返回:
    S_high: 高频分量值
    """
    # 获取节点特征和边索引
    node_features = synthetic_data.x  # [N, d] 节点特征矩阵
    edge_index = synthetic_data.edge_index
    num_nodes = node_features.shape[0]
    feature_dim = node_features.shape[1]
    
    # 计算拉普拉斯矩阵 L = D - A (组合拉普拉斯矩阵)
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index, 
        num_nodes=num_nodes, 
        normalization=None  # 使用组合拉普拉斯矩阵 L = D - A
    )
    
    # 转换为稠密矩阵
    L = to_dense_adj(edge_index_laplacian, edge_attr=edge_weight_laplacian, max_num_nodes=num_nodes)[0]
    
    # 随机选取每个节点相同的10%的特征平均作为节点的特征值x_i
    num_selected_features = max(1, int(0.1 * feature_dim))  # 至少选择1个特征
    
    # 为了保证每个节点选择相同的特征维度，我们随机选择特征索引
    selected_indices = torch.randperm(feature_dim)[:num_selected_features]
    
    # 提取选定特征并计算每个节点的平均值
    selected_features = node_features[:, selected_indices]  # [N, num_selected_features]
    x = selected_features.mean(dim=1)  # [N,] 每个节点的特征值
    
    # 计算 S_high = x^T L x / x^T x
    xTLx = torch.matmul(torch.matmul(x.unsqueeze(0), L), x.unsqueeze(1)).squeeze()  # x^T L x
    xTx = torch.dot(x, x)  # x^T x
    
    # 避免除零
    if xTx == 0:
        S_high = torch.tensor(0.0, device=node_features.device)
    else:
        S_high = xTLx / xTx
    
    return S_high