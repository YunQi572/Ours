import torch
import numpy as np
from datasets.dataset_loader import load_dataset, get_client_task
import matplotlib.pyplot as plt

# 测试参数
dataset_name = 'cora'
root_dir = './data'  # 可以根据需要更改
train_valid_test_split = [0.6, 0.2, 0.2]
clients_num = 3
per_task_class_num = 2
shuffle_flag = False

# 加载数据集
print(f"加载数据集: {dataset_name}")
data = load_dataset(train_valid_test_split, root_dir, dataset_name)
print(f"数据集信息:")
print(f"- 节点数量: {data.num_nodes}")
print(f"- 边数量: {data.edge_index.shape[1]}")
print(f"- 节点特征维度: {data.x.shape[1]}")
print(f"- 类别数量: {data.y.max().item() + 1}")

# 获取客户端任务
print(f"\n使用Louvain方法将数据分割为 {clients_num} 个客户端")
clients_tasks, in_dim, out_dim = get_client_task(
    data, 
    clients_num, 
    per_task_class_num, 
    train_valid_test_split[0], 
    train_valid_test_split[1], 
    train_valid_test_split[2]
)

print(f"\n特征维度: {in_dim}, 输出维度: {out_dim}")

# 打印每个客户端的信息
for client_id in range(clients_num):
    client_data = clients_tasks[client_id]["data"]
    client_tasks = clients_tasks[client_id]["task"]
    
    print(f"\n客户端 {client_id} 信息:")
    print(f"- 节点数量: {client_data.num_nodes}")
    print(f"- 边数量: {client_data.edge_index.shape[1]}")
    print(f"- 任务数量: {len(client_tasks)}")
    
    # 打印每个任务的信息
    for task_idx, task in enumerate(client_tasks):
        print(f"  任务 {task_idx}:")
        train_mask = task["train_mask"]
        valid_mask = task["valid_mask"]
        test_mask = task["test_mask"]
        local_data = task["local_data"]
        
        print(f"  - 训练集大小: {train_mask.sum().item()}")
        print(f"  - 验证集大小: {valid_mask.sum().item()}")
        print(f"  - 测试集大小: {test_mask.sum().item()}")
        
        # 打印局部子图信息
        if local_data is not None:
            print(f"  - 局部子图节点数量: {local_data.num_nodes}")
            print(f"  - 局部子图边数量: {local_data.edge_index.shape[1]}")
            
            # 打印类别分布
            unique_classes, counts = torch.unique(local_data.y, return_counts=True)
            class_distribution = {int(cls.item()): int(count.item()) for cls, count in zip(unique_classes, counts)}
            print(f"  - 类别分布: {class_distribution}")
