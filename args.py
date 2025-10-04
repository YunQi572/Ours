import argparse
import os
import torch

parser = argparse.ArgumentParser(description='参数设置')

# 数据集路径设置
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

parser.add_argument("--dataset_name", type=str, default="cora", help='数据集名称')
parser.add_argument("--dataset_dir", type=str, default=root_dir, help='数据集路径')

# 基本参数
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--device_id', type=int, default=0, help='GPU ID')

# 联邦学习参数
parser.add_argument('--clients_num', type=int, default=3, help='客户端数量')
parser.add_argument('--rounds', type=int, default=20, help='联邦学习轮数')
parser.add_argument('--local_epochs', type=int, default=2000, help='本地训练轮数')

# 数据分割参数
parser.add_argument('--per_task_class_num', type=int, default=2, help='每个任务的类别数')
parser.add_argument('--train_prop', type=float, default=0.6, help='训练集比例')
parser.add_argument('--valid_prop', type=float, default=0.2, help='验证集比例') 
parser.add_argument('--test_prop', type=float, default=0.2, help='测试集比例')
parser.add_argument('--shuffle_flag', type=bool, default=False, help='是否打乱数据')

# 模型参数
parser.add_argument('--model', type=str, default='GCN', help='模型: GCN, GAT, etc.')
parser.add_argument('--input_dim', type=int, default=1433, help='输入特征维度')
parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
parser.add_argument('--output_dim', type=int, default=7, help='输出维度')
parser.add_argument('--num_layers', type=int, default=2, help='模型层数')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')

# 优化器参数
parser.add_argument('--lr', type=float, default=0.01, help='学习率')
parser.add_argument('--lr_g', type=float, default=0.01, help='生成器学习率')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')

# 生成器参数
parser.add_argument('--noise_dim', type=int, default=100, help='噪声维度')
parser.add_argument('--bn_momentum', type=float, default=0.1, help='批归一化动量参数')
parser.add_argument('--gen_epochs', type=int, default=2000, help='生成器训练轮数')
parser.add_argument('--num_samples', type=int, default=1000, help='每轮生成的样本数')
parser.add_argument('--bn_weight', type=float, default=1.0, help='BN损失权重')
parser.add_argument('--sc_weight', type=float, default=1.0, help='SC权重')
parser.add_argument('--div_weight', type=float, default=1.0, help='边缘损失权重')
parser.add_argument('--bandwidth', type=float, default=0.1, help='高斯核带宽')

# 知识蒸馏参数
parser.add_argument('--kd_lr', type=float, default=0.001, help='知识蒸馏学习率')
parser.add_argument('--kd_temperature', type=float, default=4.0, help='知识蒸馏温度')
parser.add_argument('--kd_epochs', type=int, default=2000, help='知识蒸馏训练轮数')
parser.add_argument('--temperature', type=float, default=2.0, help='温度参数')
parser.add_argument('--kd_ce_weight', type=float, default=0.5, help='交叉熵损失权重')
parser.add_argument('--kd_low_weight', type=float, default=0.2, help='低频损失权重')

