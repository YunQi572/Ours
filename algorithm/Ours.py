from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse, coalesce, get_laplacian
from models.model import *
from datasets.dataset_loader import class_to_task
from utils import Generator, LinkPredictor, edge_distribution_low, construct_graph, DeepInversionHook, compute_led, KDE, gaussian_kernel, apply_kde_to_energy_dist, compute_js_divergence_from_prob_dist, get_SC, get_Shigh
from datasets.partition import get_subgraph_by_node
import copy
import matplotlib.pyplot as plt
import os
from datetime import datetime

class OursServer(BaseServer):
    def __init__(self, args, message_pool, device):
        super(OursServer, self).__init__(args, message_pool)
        self.args = args
        self.clients_num = self.args.clients_num
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.global_model = load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.client_model = [load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device) for _ in range(self.clients_num)]
        self.last_global_model = load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.per_task_class_num = self.args.per_task_class_num
        # 初始化生成器和链接预测器
        self.noise_dim = getattr(args, 'noise_dim', 128)  # 噪声维度
        self.bn_mmt = getattr(args, 'bn_momentum', 0.1)  # BN动量参数
        self.threshold = self.args.threshold
        #初始化存储客户端信息的列表
        self.clients_nodes_num = []
        self.clients_graph_energy = []
        
        # 初始化损失值存储列表 - 改为每个任务存储连续的epoch损失
        self.generator_losses = {}  # 存储每个任务所有通信轮次的生成器训练损失值 {task_id: [losses]}
        self.generator_loss_details = {}  # 存储每个任务所有通信轮次的生成器训练详细损失分项 {task_id: [details]}
        self.kd_losses = {}  # 存储每个任务所有通信轮次的知识蒸馏损失值 {task_id: [losses]}
        self.kd_loss_details = {}  # 存储每个任务所有通信轮次的知识蒸馏详细损失分项 {task_id: [details]}

    def aggregate(self):
        """
        聚合客户端模型参数到全局模型
        从message_pool中获取客户端模型参数
        """
        # 从message_pool中获取客户端模型参数
        client_weights = []
        for client_idx in range(self.clients_num):
            client_key = f"client_{client_idx}"
            if client_key in self.message_pool and "weight" in self.message_pool[client_key]:
                client_weights.append(self.message_pool[client_key]["weight"])
        
        # 如果没有获取到客户端参数，则直接返回
        if not client_weights:
            print("No client weights found in message_pool")
            return
            
        totoal_nodes_num = sum([self.message_pool[f"client_{client_id}"]["nodes_num"] for client_id in range(self.clients_num)])

        #更新服务器模型参数
        for i, client_id in enumerate(range(self.clients_num)):
            weight = self.message_pool[f"client_{client_id}"]["nodes_num"] / totoal_nodes_num
            for (client_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.global_model.parameters()):
                if i == 0:
                    global_param.data.copy_(weight * client_param)
                else:
                    global_param.data += weight * client_param
        

    
    #生成器（包括链接预测器）训练
    def train(self, task_id):
        """
        训练生成器和链接预测器
        使用随机噪声和随机标签，通过交叉熵损失训练生成器
        """
        # 生成器每轮需要重新初始化
        classes_num = task_id * self.per_task_class_num
        self.generator = Generator(noise_dim = self.noise_dim, input_dim = self.args.input_dim, output_dim = classes_num, dropout = self.args.dropout).to(self.device)
        self.link_predictor = LinkPredictor(input_dim = self.args.input_dim, hidden_dim=64 , dropout=0.5).to(self.device)

        # 为生成器的每个批归一化层添加DeepInversionHook钩子
        self.hooks = []
        for m in self.generator.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))
        # print(f"hooks: {self.hooks}")

        # 设置训练模式
        self.generator.train()
        self.link_predictor.train()
        self.global_model.eval()  # 全局模型用于预测，设为评估模式
        self.last_global_model.eval()  # 全局模型用于预测，设为评估模式
        
        # 优化器
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_g if hasattr(self.args, 'lr_g') else 0.01)
        link_optimizer = torch.optim.Adam(self.link_predictor.parameters(), lr=self.args.lr if hasattr(self.args, 'lr') else 0.01)
        
        # 损失函数
        cross_entropy_loss = nn.CrossEntropyLoss()
        
        # 训练参数
        num_epochs = getattr(self.args, 'gen_epochs', 100)  # 生成器训练轮数
        num_samples = getattr(self.args, 'num_samples', 1000)  # 每轮生成的样本数
        bn_weight = getattr(self.args, 'bn_weight', 1.0)  # BN损失权重，默认1.0
        sc_weight = getattr(self.args, 'sc_weight', 1.0)
        shigh_weight = self.args.shigh_weight

        print(f"========== 生成器 在第{task_id}个任务训练时的loss值为:==========\n")
        
        # 为当前任务初始化损失记录列表（如果还没有）
        if task_id not in self.generator_losses:
            self.generator_losses[task_id] = []
            self.generator_loss_details[task_id] = []
        
        for epoch in range(num_epochs):
            gen_optimizer.zero_grad()
            link_optimizer.zero_grad()
            
            # 1. 生成随机噪声和随机标签
            noise = torch.randn(num_samples, self.noise_dim).to(self.device)  # 随机噪声
            random_labels = torch.randint(0, classes_num, (num_samples,)).to(self.device)  # 随机标签(左闭右开)
            print(f"随机标签：{random_labels}")
            # 2. 使用生成器生成节点特征
            generated_features = self.generator(noise, random_labels)  # [num_samples, input_dim]
            
            # 3. 使用链接预测器构建图结构
            adj_matrix = construct_graph(generated_features, self.link_predictor, threshold=self.threshold)
            
            # 4. 构建图数据
            edge_index = dense_to_sparse(adj_matrix)[0]  # 转换为边索引
            edge_index = coalesce(edge_index)  # 合并重复边
            
            print("train")
            print("generated_features:", generated_features)
            print("adj_matrix:", adj_matrix)
            print("edge_index:", edge_index)
            print("edge_index shape:", edge_index.shape)

            # 创建图数据对象
            synthetic_data = Data(x=generated_features, edge_index=edge_index, y=random_labels)
            synthetic_data = synthetic_data.to(self.device)
            
            # 5. 使用全局模型对生成的节点特征进行预测
            # with torch.no_grad():
            # _, global_predictions = self.global_model(synthetic_data)  # [num_samples, output_dim]
            _, global_predictions = self.last_global_model(synthetic_data)  # [num_samples, output_dim]
            
            # 6. 计算损失
            # 6.1 交叉熵损失：生成的数据标签 vs 全局模型的预测
            loss_ce = cross_entropy_loss(global_predictions, random_labels)
            
            # 6.2 批归一化损失：从所有钩子中收集BN特征损失
            # loss_bn = sum([h.r_feature for h in self.hooks]) if self.hooks else torch.tensor(0.0, device=self.device)
            loss_bn = sum([h.r_feature for h in self.hooks])
            
            #6.3 SC损失： 
            loss_sc = get_SC(synthetic_data = synthetic_data, clients_nodes_num = self.clients_nodes_num, clients_graph_energy = self.clients_graph_energy, device = self.device, h = self.args.bandwidth)

            #6.4 边缘损失函数
            # 使用全局模型（当前模型）对生成数据进行预测
            _, global_logits = self.global_model(synthetic_data)
            global_probs = F.softmax(global_logits, dim=1)
            
            #6.5 高频
            loss_shigh = get_Shigh(synthetic_data=synthetic_data)

            # 使用上一轮模型（如果可用）对相同数据进行预测
            if task_id > 0:  # 确保不是第一个任务
                with torch.no_grad():
                    _, last_logits = self.last_global_model(synthetic_data)
                last_probs = F.softmax(last_logits, dim=1)
                
                # 计算两个模型对每个节点的预测类别
                global_pred = torch.argmax(global_probs, dim=1)
                last_pred = torch.argmax(last_probs, dim=1)
                
                # 计算指示器ω：对于每个节点，如果预测结果不同则为1，否则为0
                omega = (global_pred != last_pred).float()
                
                # 计算KL散度：KL(last_probs || global_probs)
                # 注意：KL散度公式为 sum(p * log(p/q))
                kl_divergence = F.kl_div(
                    F.log_softmax(global_logits, dim=1),  # log(q)
                    last_probs,                           # p
                    reduction='none'
                ).sum(dim=1)
                
                # 应用ω：只对预测不同的样本计算KL散度
                weighted_kl = omega * kl_divergence
                
                # 计算边缘损失（取负值，因为我们想要最大化不同）
                loss_div = -weighted_kl.mean()
                
                # 边缘损失权重
                div_weight = getattr(self.args, 'div_weight', 0.5)
            else:
                print(f"debug info loss_div")
                loss_div = torch.tensor(0.0, device=self.device)
                div_weight = 0.0
            

            # total_loss = loss_ce
            total_loss = loss_ce + bn_weight * loss_bn
            
            # total_loss = loss_ce + bn_weight * loss_bn + shigh_weight * loss_shigh
            # total_loss = loss_ce + bn_weight * loss_bn + shigh_weight * loss_shigh + div_weight * loss_div

            # total_loss = loss_ce + bn_weight * loss_bn + loss_sc * sc_weight
            # total_loss = loss_ce + bn_weight * loss_bn + loss_sc * sc_weight + div_weight * loss_div

            print(f"SC的值为:{loss_sc}")
            # print(f"S_high的值为:{loss_shigh}")
            print(f"生成器的loss:{total_loss}")
            # 7. 反向传播和优化
            total_loss.backward()
            gen_optimizer.step()
            link_optimizer.step()
            
            print(f"第{epoch}轮的loss为:{total_loss}\n")
            
            # 保存当前epoch的损失值到任务的连续epoch列表中
            current_epoch_in_task = len(self.generator_losses[task_id])
            self.generator_losses[task_id].append(total_loss.item())
            self.generator_loss_details[task_id].append({
                'epoch': current_epoch_in_task,
                'total_loss': total_loss.item(),
                'ce_loss': loss_ce.item(),
                'sc_loss': loss_sc.item(), 
                'bn_loss': loss_bn.item(),
                'shigh_loss': loss_shigh.item(),
                'div_loss': loss_div.item()
            })
            
            # 8. 更新钩子的动量统计
            if self.bn_mmt != 0:
                for h in self.hooks:
                    h.update_mmt()
            
            # 9. 打印训练信息
            if epoch % 10 == 0:
                print(f"SYS Epoch {epoch}/{num_epochs}, Total Loss: {total_loss.item():.4f}, "
                      f"CE Loss: {loss_ce.item():.4f}, SC Loss: {loss_sc.item():.4f}, BN Loss: {loss_bn.item():.4f}, DIV Loss: {loss_div.item():.4f}")
        
        # 清理钩子
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
        print("Generator and Link Predictor training completed!")

    def synthesis_data(self, task_id, num_samples_per_class=10):
        """
        使用训练好的生成器生成合成数据
        
        参数:
        num_samples_per_class: 每个类别生成的样本数量
        
        返回:
        synthetic_data: 生成的合成图数据
        """
        self.generator.eval()
        self.link_predictor.eval()
        classes_num = task_id * self.per_task_class_num
        
        with torch.no_grad():
            all_features = []
            all_labels = []
            
            # 为每个类别生成样本
            # for class_id in range(self.args.output_dim):
            for class_id in range(classes_num):
                # 生成噪声
                noise = torch.randn(num_samples_per_class, self.noise_dim).to(self.device)
                # 生成当前类别的标签
                labels = torch.full((num_samples_per_class,), class_id, dtype=torch.long).to(self.device)
                
                # 生成节点特征
                features = self.generator(noise, labels)
                
                all_features.append(features)
                all_labels.append(labels)
            


            # 合并所有特征和标签
            synthetic_features = torch.cat(all_features, dim=0)
            synthetic_labels = torch.cat(all_labels, dim=0)
            # print("synthetic_features shape:", synthetic_features.shape)
            # print("synthetic_features:", synthetic_features)
            # print("synthetic_features mean:", synthetic_features.mean().item())
            # print("synthetic_features std:", synthetic_features.std().item())
            # 使用链接预测器构建图结构
            adj_matrix = self.link_predictor.predict_links(synthetic_features, threshold=self.threshold)
            edge_index = dense_to_sparse(adj_matrix)[0]
            edge_index = coalesce(edge_index)
            
            print("adj_matrix:", adj_matrix)
            print("edge_index:", edge_index)
            print("edge_index shape:", edge_index.shape)

            # 创建合成数据
            synthetic_data = Data(x=synthetic_features, edge_index=edge_index, y=synthetic_labels)
            
            # 创建训练掩码（这里简单地将所有节点都作为训练节点）
            num_nodes = synthetic_features.shape[0]
            train_mask = torch.ones(num_nodes, dtype=torch.bool)
            
            # 构造与客户端任务格式相同的数据结构
            self.synthesis_task = {
                "local_data": synthetic_data,
                "train_mask": train_mask,
                "valid_mask": torch.zeros(num_nodes, dtype=torch.bool),  # 空的验证掩码
                "test_mask": torch.zeros(num_nodes, dtype=torch.bool)     # 空的测试掩码
            }
            
    

    #上一轮的全局模型对这一轮的全局模型的知识蒸馏
    def KD_train(self, task_id):
        """
        使用生成的合成数据和上一轮的全局模型对当前全局模型进行知识蒸馏
        参数:
        task_id: 当前任务ID
        """
        # 如果是第一个任务，没有需要蒸馏的知识
        if task_id == 0:
            print("First task, no knowledge distillation needed.")
            # 为第一个任务初始化空的KD损失列表
            if task_id not in self.kd_losses:
                self.kd_losses[task_id] = []
                self.kd_loss_details[task_id] = []
            return
            
        # 生成合成数据用于知识蒸馏
        self.synthesis_data(task_id, num_samples_per_class=self.args.num_samples_per_class)

        synthetic_task = self.synthesis_task
        synthetic_data = synthetic_task["local_data"]
        print(f"KD train SYS Data.y:{synthetic_data.y}")

        
        # 设置模型模式
        self.global_model.train()  # 当前全局模型设为训练模式
        self.last_global_model.eval()  # 上一轮全局模型设为评估模式
        
        # 优化器
        kd_optimizer = torch.optim.Adam(self.global_model.parameters(), 
                                       lr=getattr(self.args, 'kd_lr', 0.001))
        
        # 知识蒸馏参数
        temperature = getattr(self.args, 'kd_temperature', 4.0)  # 蒸馏温度
        num_epochs = getattr(self.args, 'kd_epochs', 50)  # 蒸馏轮数
        
        # 将数据移到设备上
        synthetic_data = synthetic_data.to(self.device)
        
        print(f"Starting knowledge distillation for task {task_id}...")
        
        # 为当前任务初始化KD损失记录列表（如果还没有）
        if task_id not in self.kd_losses:
            self.kd_losses[task_id] = []
            self.kd_loss_details[task_id] = []
        
        print(f"========== 全局模型 在第{task_id}个任务训练时的 KD loss值为:==========\n")
        for epoch in range(num_epochs):
            kd_optimizer.zero_grad()
            
            # 当前全局模型的输出（学生模型）
            _, student_logits = self.global_model(synthetic_data)
            
            # 上一轮全局模型的输出（教师模型）
            with torch.no_grad():
                _, teacher_logits = self.last_global_model(synthetic_data)
            
            #计算global_mode在生成数据集上的交叉熵损失
            loss_ce = nn.CrossEntropyLoss()(student_logits, synthetic_data.y)

            #计算低频蒸馏损失
            # 使用温度参数软化输出分布进行知识蒸馏
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = F.log_softmax(teacher_logits / temperature, dim=1)
            
            # print(f"KD student_log_probs.requires_grad: {student_log_probs.requires_grad}")
            # print(f"KD teacher_probs.requires_grad: {teacher_probs.requires_grad}")
            

            # 计算低频蒸馏损失（使用edge_distribution_low函数）
            loss_kd_low = edge_distribution_low(
                synthetic_data.edge_index,
                student_log_probs,
                teacher_probs
            )
            
            # 损失权重
            ce_weight = getattr(self.args, 'kd_ce_weight', 0.5)  # 交叉熵损失权重
            low_weight = getattr(self.args, 'kd_low_weight', 0.2) # 低频蒸馏损失权重
            
            # 总损失
            # total_loss = ce_weight * loss_ce
            total_loss = ce_weight * loss_ce + low_weight * loss_kd_low
            # print("KD loss_ce.requires_grad:", loss_ce.requires_grad)
            # print("KD loss_kd_low.requires_grad:", loss_kd_low.requires_grad)
            # print("KD loss_kd_low:", loss_kd_low.item())

            print(f"第{epoch}轮的loss为:{total_loss}\n")
            
            # 保存当前epoch的损失值到任务的连续epoch列表中
            current_epoch_in_task = len(self.kd_losses[task_id])
            self.kd_losses[task_id].append(total_loss.item())
            self.kd_loss_details[task_id].append({
                'epoch': current_epoch_in_task,
                'total_loss': total_loss.item(),
                'ce_loss': loss_ce.item(),
                'low_freq_loss': loss_kd_low.item()
            })
            
            # 反向传播
            total_loss.backward()
            kd_optimizer.step()
            
            # 打印训练信息
            if epoch % 10 == 0:
                print(f"KD Epoch {epoch}/{num_epochs}, Total Loss: {total_loss.item():.4f}, "
                      f"CE Loss: {loss_ce.item():.4f}, Low Freq Loss: {loss_kd_low.item():.8f}")
        
       

        print("Knowledge distillation completed!")

    def update_last_global_model(self):
        #将last_global_model的参数更新为global_model的参数    
        with torch.no_grad():
            for last_param, global_param in zip(self.last_global_model.parameters(), self.global_model.parameters()):
                last_param.data.copy_(global_param.data)
   
    def plot_loss_curves(self, save_dir="./loss_plots"):
        """
        绘制损失变化图像，包括生成器和知识蒸馏的各项损失
        """
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 绘制生成器总损失变化（每个任务一个图，所有通信轮次连续显示）
        if self.generator_losses:
            for task_id, task_losses in self.generator_losses.items():
                if task_losses:  # 确保任务有损失数据
                    plt.figure(figsize=(12, 8))
                    epochs = range(len(task_losses))
                    plt.plot(epochs, task_losses, label=f'Task {task_id + 1}', marker='o', markersize=3)
                    
                    plt.title(f'Generator Total Loss - Task {task_id + 1} (All Communication Rounds)')
                    plt.xlabel('Continuous Epoch (All Rounds)')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'generator_total_loss_task_{task_id + 1}_{timestamp}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
        # 2. 绘制生成器各项损失分解图（每个任务一个图，包含Shigh损失）
        if self.generator_loss_details:
            for task_id, task_details in self.generator_loss_details.items():
                if not task_details:  # 跳过空的任务
                    continue
                    
                plt.figure(figsize=(18, 12))
                
                # 提取各项损失
                epochs = [detail['epoch'] for detail in task_details]
                ce_losses = [detail['ce_loss'] for detail in task_details]
                sc_losses = [detail['sc_loss'] for detail in task_details]
                bn_losses = [detail['bn_loss'] for detail in task_details]
                shigh_losses = [detail['shigh_loss'] for detail in task_details]
                div_losses = [detail['div_loss'] for detail in task_details]
                total_losses = [detail['total_loss'] for detail in task_details]
                
                # 创建子图
                plt.subplot(2, 4, 1)
                plt.plot(epochs, ce_losses, 'b-', marker='o', markersize=2)
                plt.title('Cross Entropy Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 4, 2)
                plt.plot(epochs, sc_losses, 'g-', marker='o', markersize=2)
                plt.title('SC Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 4, 3)
                plt.plot(epochs, bn_losses, 'r-', marker='o', markersize=2)
                plt.title('Batch Norm Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 4, 4)
                plt.plot(epochs, shigh_losses, 'orange', marker='o', markersize=2)
                plt.title('Shigh Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 4, 5)
                plt.plot(epochs, div_losses, 'm-', marker='o', markersize=2)
                plt.title('Divergence Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 4, 6)
                plt.plot(epochs, total_losses, 'k-', marker='o', markersize=2)
                plt.title('Total Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 4, 7)
                plt.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
                plt.plot(epochs, sc_losses, 'g-', label='SC Loss', linewidth=2)
                plt.plot(epochs, bn_losses, 'r-', label='BN Loss', linewidth=2)
                plt.plot(epochs, shigh_losses, 'orange', label='Shigh Loss', linewidth=2)
                plt.plot(epochs, div_losses, 'm-', label='DIV Loss', linewidth=2)
                plt.plot(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
                plt.title('All Losses Combined')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Generator Loss Details - Task {task_id + 1} (All Communication Rounds)', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'generator_detailed_loss_task_{task_id + 1}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. 绘制知识蒸馏总损失变化（每个任务一个图）
        if self.kd_losses:
            for task_id, task_losses in self.kd_losses.items():
                if task_losses:  # 确保任务有损失数据
                    plt.figure(figsize=(12, 8))
                    epochs = range(len(task_losses))
                    plt.plot(epochs, task_losses, label=f'Task {task_id + 1}', marker='o', markersize=3)
                    
                    plt.title(f'Knowledge Distillation Total Loss - Task {task_id + 1} (All Communication Rounds)')
                    plt.xlabel('Continuous Epoch (All Rounds)')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'kd_total_loss_task_{task_id + 1}_{timestamp}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
        # 4. 绘制知识蒸馏各项损失分解图
        if self.kd_loss_details:
            for task_id, task_details in self.kd_loss_details.items():
                if not task_details:  # 跳过空的任务
                    continue
                    
                plt.figure(figsize=(12, 8))
                
                # 提取各项损失
                epochs = [detail['epoch'] for detail in task_details]
                ce_losses = [detail['ce_loss'] for detail in task_details]
                low_freq_losses = [detail['low_freq_loss'] for detail in task_details]
                total_losses = [detail['total_loss'] for detail in task_details]
                
                # 创建子图
                plt.subplot(2, 2, 1)
                plt.plot(epochs, ce_losses, 'b-', marker='o', markersize=2)
                plt.title('Cross Entropy Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 2)
                plt.plot(epochs, low_freq_losses, 'g-', marker='o', markersize=2)
                plt.title('Low Frequency Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 3)
                plt.plot(epochs, total_losses, 'k-', marker='o', markersize=2)
                plt.title('Total Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 4)
                plt.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
                plt.plot(epochs, low_freq_losses, 'g-', label='Low Freq Loss', linewidth=2)
                plt.plot(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
                plt.title('All KD Losses Combined')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Knowledge Distillation Loss Details - Task {task_id + 1} (All Communication Rounds)', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'kd_detailed_loss_task_{task_id + 1}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Server loss plots saved to: {save_dir}")

    def send_message(self):
        self.message_pool["server"] = {
            "weight" : list(self.global_model.parameters())   
        }


class OursClient(BaseClient):   
    def __init__(self, args, client_id, data, message_pool, device):   #data是分割完的子图
        super(OursClient, self).__init__(args, client_id, data)
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.global_model = load_model(name = self.args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.client_model = load_model(name = self.args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()            #交叉熵损失函数 
        self.message_pool = message_pool                #存储一些来自服务器的信息（服务器上模型的参数和服务器生成的数据）
        
        # 保存原始数据用于LED计算
        self.data = data
        
        # 从args中获取任务相关参数
        self.per_task_class_num = getattr(args, 'per_task_class_num', 2)
        self.train_prop = getattr(args, 'train_prop', 0.6)
        self.valid_prop = getattr(args, 'valid_prop', 0.2) 
        self.test_prop = getattr(args, 'test_prop', 0.2)
        self.shuffle_flag = getattr(args, 'shuffle_flag', False)
        
        self.tasks = class_to_task(data = data, per_task_class_num = self.per_task_class_num, train_prop = self.train_prop, valid_prop = self.valid_prop, test_prop = self.test_prop, shuffle_flag = self.shuffle_flag)
        
        print(f"!!!!!!!!!!!!!客户端{client_id}的任务集为：")
        print(f"self.data.x: {self.data.x}")
        print(f"self.data.edge_index: {self.data.edge_index}")
        print(f"self.data.y: {self.data.y}")
        for i, task in enumerate(self.tasks):
            print(f"客户端{self.client_id} - 任务{i}:")
            print(f"  节点总数: {task['local_data'].x.shape[0]}")
            print(f"  标签分布: {torch.unique(task['local_data'].y)}")
            print(f"  训练集节点数: {task['train_mask'].sum().item()}")
            print(f"  验证集节点数: {task['valid_mask'].sum().item()}")
            print(f"  测试集节点数: {task['test_mask'].sum().item()}")

            print("  训练集类别:", torch.unique(task['local_data'].y[task['train_mask']]))
            print("  验证集类别:", torch.unique(task['local_data'].y[task['valid_mask']]))
            print("  测试集类别:", torch.unique(task['local_data'].y[task['test_mask']]))
        self.local_epochs = args.local_epochs
        
        # 初始化损失值存储列表 - 改为每个任务存储连续的epoch损失
        self.client_losses = {}  # 存储每个任务所有通信轮次的客户端训练损失值 {task_id: [losses]}

    #本地模型训练，只使用本地数据
    def train(self, task_id):
        # 使用服务器发送的全局模型参数来更新本地模型 self.client_model 和 self.global_model 的参数。
        # 只有在服务器已经发送参数时才更新
        if "server" in self.message_pool and "weight" in self.message_pool["server"]:
            with torch.no_grad():       #客户端的局部模型
                for (local_param_old, agg_global_param) in zip(self.client_model.parameters(), self.message_pool["server"]["weight"]):
                    local_param_old.data.copy_(agg_global_param)
            with torch.no_grad():       #客户端的全局模型
                for (local_param_old, agg_global_param) in zip(self.global_model.parameters(), self.message_pool["server"]["weight"]):
                    local_param_old.data.copy_(agg_global_param)
        
        task = self.tasks[task_id]
        global_model = self.global_model                        #使用已加载参数的全局模型

        self.client_model.train()           # 设置模型为训练模式
        global_model.eval()                 # 设置全局模型为评估模式
        
        # 获取本地任务数据
        local_data = task["local_data"]
        local_train_mask = task["train_mask"]
        whole_data = self.data.to(self.device)
        # 将数据移到设备上
        local_data = local_data.to(self.device)
        
        # 配置优化器
        optimizer = torch.optim.Adam(self.client_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        # 训练模型
        print(f"=========={self.client_id} 在第{task_id}个任务训练时的loss值为:==========\n")
        
        # 为当前任务初始化损失记录列表（如果还没有）
        if task_id not in self.client_losses:
            self.client_losses[task_id] = []
        
        for epoch in range(self.local_epochs):
            # 清除梯度
            optimizer.zero_grad()
            
            # 1. 在本地数据上的训练
            _, local_student_out = self.client_model(local_data)
            #交叉熵损失
            loss = self.loss_fn(local_student_out[local_train_mask], whole_data.y[local_train_mask])
            # print("GCN loss.requires_grad:", loss.requires_grad)  true
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 保存当前epoch的损失值到任务的连续epoch列表中
            self.client_losses[task_id].append(loss.item())
            
            print(f"第{epoch}轮的loss为:{loss}\n")

        return loss.item()
    
    #获得LED(Laplacian Energy Distribution)值
    def get_LED(self, task_id):
        """
        计算拉普拉斯能量分布(LED)值
        根据公式: \bar{x}_n = \frac{\hat{x}_n^2}{\sum_{i=1}^N \hat{x}_i^2}
        """
        # 如果是第一个任务，返回0（没有之前学习的节点）
        if task_id == 0:
            return 0.0
            
        #根据任务获得截止到task_id时已学习的节点种类
        nodes_list = []              #节点编号列表
        
        # 需要获取原始完整数据
        original_data = self.data 
        # if original_data is None:
        #     # 如果没有原始数据，通过合并所有任务的数据来重建
        #     all_nodes = []
        #     for i in range(task_id):
        #         task = self.tasks[i]
        #         nodes_mask = task["train_mask"] | task["valid_mask"] | task["test_mask"]
        #         nodes_indices = torch.where(nodes_mask)[0].tolist()
        #         all_nodes.extend(nodes_indices)
            
        #     # 去重并排序
        #     all_nodes = sorted(list(set(all_nodes)))
            
        #     # 使用第一个任务的子图作为基础来获取完整图结构
        #     first_task_data = self.tasks[0]["local_data"]
        #     return len(all_nodes) / first_task_data.x.shape[0]  # 简化的LED值
        
        for i in range(task_id): #已经学习的任务编号是[0, task_id - 1]
            task = self.tasks[i]
            nodes_mask = task["train_mask"] | task["valid_mask"] | task["test_mask"]
            
            # 获取当前任务中所有节点的索引
            nodes_indices = torch.where(nodes_mask)[0].tolist()
            nodes_list.extend(nodes_indices)
        
        # 去重并排序，得到所有已学习的节点列表
        nodes_list = sorted(list(set(nodes_list)))
        
        # 获取包含已学习节点的子图
        subgraph = get_subgraph_by_node(original_data, nodes_list)


        # 直接计算LED
        nodes_feature = subgraph.x  # [N, d] 节点特征矩阵
        edge_index = subgraph.edge_index
        
        # 1. 计算图的拉普拉斯矩阵
        num_nodes = nodes_feature.shape[0]
        # 获取标准化拉普拉斯矩阵 L = I - D^(-1/2) A D^(-1/2)
        edge_index_laplacian, edge_weight_laplacian = get_laplacian(
            edge_index, 
            num_nodes=num_nodes, 
            normalization='sym'  # 对称标准化
        )
        
        # 转换为稠密矩阵
        L = to_dense_adj(edge_index_laplacian, edge_attr=edge_weight_laplacian, max_num_nodes=num_nodes)[0]
        
        # 2. 对拉普拉斯矩阵进行特征值分解，获取特征向量矩阵 U
        eigenvalues, eigenvectors = torch.linalg.eigh(L)  # U: [N, N]
        U = eigenvectors  # 特征向量矩阵
        U = U.to(self.device)
        nodes_feature = nodes_feature.to(self.device)
        # 3. 计算图傅里叶变换 \hat{X} = U^T X
        X_hat = torch.matmul(U.T, nodes_feature)  # [N, d] 傅里叶变换后的特征
        
        # 4. 根据公式计算 LED 值
        # \bar{x}_n = \frac{\hat{x}_n^2}{\sum_{i=1}^N \hat{x}_i^2}
        
        # 计算每个频率分量的能量（所有特征维度的平方和）
        energy_per_freq = torch.sum(X_hat ** 2, dim=1)  # [N,] 每个频率的能量
        print(f"led debug_info energy_per_freq: {energy_per_freq}")
        # 计算总能量
        total_energy = torch.sum(energy_per_freq)       #\sum_{i=1}^N \hat{x}_i^2
        
        # 计算能量分布（归一化）
        if total_energy > 0:
            energy_distribution = energy_per_freq / total_energy  # [N,] 归一化的能量分布
        else:
            energy_distribution = torch.zeros_like(energy_per_freq)
        
        return energy_distribution
        
    def evaluate(self, task_id, global_flag = True, mask = "test_mask"):
        """评估客户端模型在指定任务上的性能"""
        task = self.tasks[task_id]
        
        if global_flag:
            print(f"使用全局模型评估{task_id}")
            client_param_copy = copy.deepcopy(list(self.client_model.parameters()))
            with torch.no_grad():
                for(client_param, global_param) in zip(self.client_model.parameters(), self.message_pool["server"]["weight"]):
                    client_param.data.copy_(global_param)

        # 设置模型为评估模式
        self.client_model.eval()
        
        # 获取任务数据
        data = task["local_data"]
        
        # 将数据移到设备上
        data = data.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            _, out = self.client_model(data)
            
            # 计算验证集损失
            loss = self.loss_fn(out[task[mask]], data.y[task[mask]])
            
            # 计算验证集精度
            _, pred = out.max(dim=1)
            correct = pred[task[mask]].eq(data.y[task[mask]]).sum().item()
            acc = correct / task[mask].sum().item()
        
        # 在 evaluate 函数的前向传播后加：
        print("真实标签:", data.y[task[mask]])
        print("预测标签:", pred[task[mask]])
        print("类别分布:", torch.unique(data.y[task[mask]]), torch.unique(pred[task[mask]]))

        if global_flag:
            with torch.no_grad():
                for(global_param, client_param) in zip(self.client_model.parameters(), client_param_copy):
                    global_param.data.copy_(client_param)
        
        return {"loss": loss.item(), "acc": acc}
    
    def test(self, task_id):
        """测试客户端模型在指定任务的测试集上的性能"""
        task = self.tasks[task_id]
        
        # 设置模型为评估模式
        self.client_model.eval()
        
        # 获取任务数据
        data = task["local_data"]
        test_mask = task["test_mask"]
        
        # 将数据移到设备上
        data = data.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            _, out = self.client_model(data)
            
            # 计算测试集精度
            _, pred = out.max(dim=1)
            correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
            acc = correct / test_mask.sum().item()
            
        return {"acc": acc}
    
    def update_global_model(self):
        """从服务器获取全局模型参数"""
        self.global_model.load_state_dict(self.message_pool["global_model_params"])
    
    def update_client_model(self):
        """将客户端模型更新为全局模型"""
        self.client_model.load_state_dict(self.global_model.state_dict())

    #获取当前任务中客户端数据的节点数量
    def get_task_nodes_num(self, task_id):
        task = self.tasks[task_id]
        nodes_mask = task["train_mask"] | task["valid_mask"] | task["test_mask"]
        nodes_num = nodes_mask.sum()
        return nodes_num 

    #向服务器发送信息
    def send_message(self, task_id):
        self.message_pool[f"client_{self.client_id}"] = {
            "nodes_num" : self.get_task_nodes_num(task_id),         #节点数量
            "data_LED" : self.get_LED(task_id),                     #本地数据的
            "weight" : list(self.client_model.parameters())
        }


def plot_all_losses(server, clients, save_dir="./loss_plots"):
    """
    绘制所有损失变化图像，包括服务器和客户端的损失
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 绘制服务器损失
    server.plot_loss_curves(save_dir)
    
    # 2. 绘制所有客户端的本地训练损失
    if clients and clients[0].client_losses:
        # 获取所有任务ID
        all_task_ids = set()
        for client in clients:
            all_task_ids.update(client.client_losses.keys())
        
        # 为每个任务绘制所有客户端的损失对比图
        for task_id in sorted(all_task_ids):
            plt.figure(figsize=(12, 8))
            
            for client in clients:
                if task_id in client.client_losses and client.client_losses[task_id]:
                    epochs = range(len(client.client_losses[task_id]))
                    plt.plot(epochs, client.client_losses[task_id], 
                            label=f'Client {client.client_id}', 
                            marker='o', markersize=3)
            
            plt.title(f'Client Local Training Loss - Task {task_id + 1} (All Communication Rounds)')
            plt.xlabel('Continuous Epoch (All Rounds)')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'clients_local_loss_task_{task_id + 1}_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 绘制每个客户端跨任务的损失变化
        for client in clients:
            if client.client_losses:
                plt.figure(figsize=(12, 8))
                
                for task_id, task_losses in client.client_losses.items():
                    if task_losses:
                        epochs = range(len(task_losses))
                        plt.plot(epochs, task_losses, 
                                label=f'Task {task_id + 1}', 
                                marker='o', markersize=3)
                
                plt.title(f'Client {client.client_id} - Local Training Loss Over Tasks (Each Task All Rounds)')
                plt.xlabel('Continuous Epoch (All Rounds per Task)')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'client_{client.client_id}_loss_over_tasks_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. 绘制所有客户端和所有任务的损失综合图
        plt.figure(figsize=(15, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(clients)))
        
        for client_id, client in enumerate(clients):
            if client.client_losses:
                for task_id, task_losses in client.client_losses.items():
                    if task_losses:
                        # 为了在全局图中显示，需要调整x轴（考虑任务间的间隔）
                        global_epochs = [epoch + task_id * len(task_losses) * 1.2 for epoch in range(len(task_losses))]
                        plt.plot(global_epochs, task_losses, 
                                color=colors[client_id], 
                                linestyle='-' if task_id == 0 else '--' if task_id == 1 else ':',
                                label=f'Client {client_id} Task {task_id + 1}', 
                                marker='o', markersize=2)
        
        plt.title('All Clients Local Training Loss - All Tasks (Each Task All Rounds)')
        plt.xlabel('Global Training Progress (Continuous Epochs)')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'all_clients_all_tasks_loss_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All loss plots saved to: {save_dir}")
    return save_dir

#加载服务器和客户端
def load_server_clients(args, data, device):
    message_pool = {}
    clients_num = args.clients_num
    server = OursServer(args, message_pool, device)
    clients = [OursClient(args, client_id, data[client_id], message_pool, device) for client_id in range(clients_num)]

    return server, clients, message_pool