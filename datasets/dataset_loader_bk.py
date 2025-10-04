import os
import torch
import numpy as np
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from partition import get_subgraph_by_node, louvain_partitioner

def load_dataset(train_valid_test_split, root_dir, dataset_name):
    """加载数据集"""
    assert dataset_name in ('cora', 'citeseer'), 'Invalid dataset'

    if len(train_valid_test_split) != 3:
        print("Invalid splits, will use default split proportion")
        train_valid_test_split = [0.6, 0.2, 0.2]

    train_prop = train_valid_test_split[0]
    valid_prop = train_valid_test_split[1]
    test_prop = train_valid_test_split[2]

    if dataset_name in ['cora', 'citeseer']:
        dataset = Planetoid(root=root_dir, name=dataset_name)
        data = dataset[0]  # 数据集只有一张大图，所以只用取出第一个图就行

    return data

def class_to_task(data, per_task_class_num, train_prop, valid_prop, test_prop, shuffle_flag=False):
    """将加载的数据按类分割为任务集"""
    nodes_num = data.x.shape[0]
    classes_num = data.y.max().item() + 1

    train_mask = torch.zeros(nodes_num, dtype=torch.bool)
    valid_mask = torch.zeros(nodes_num, dtype=torch.bool)
    test_mask = torch.zeros(nodes_num, dtype=torch.bool)

    classes_nodes = []  # 类i包含的所有节点
    # 将节点分为训练、验证、测试集
    for class_i in range(classes_num):
        class_i_node_mask = data.y == class_i  # 子图中属于这一类的所有节点
        class_i_node_num = class_i_node_mask.sum().item()  # 所有这一类节点的数量

        class_i_node_list = torch.where(class_i_node_mask)[0].numpy()  # 属于这一类的节点列表
        classes_nodes.append(class_i_node_list)
        np.random.shuffle(class_i_node_list)  # 打乱节点顺序

        # 将一个类的节点按比例分为训练、验证、测试集
        train_num = int(class_i_node_num * train_prop)  # 训练集个数
        valid_num = int(class_i_node_num * valid_prop)  # 验证集个数
        test_num = int(class_i_node_num * test_prop)  # 测试集个数

        train_idx = class_i_node_list[:train_num]
        valid_idx = class_i_node_list[train_num:train_num + valid_num]
        test_idx = class_i_node_list[train_num + valid_num:train_num + valid_num + test_num]

        # 标记每个节点属于哪一个集合当中
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True

    # 计算任务数量，任务数量向上取整，之后判断是否舍去凑不够的最后一组当中的类
    tasks_num = (classes_num + per_task_class_num - 1) // per_task_class_num

    task_classes = [[] for _ in range(tasks_num)]  # 任务i包含了哪些节点的类别
    label_task = {}  # 存储标签与任务id之间的对应
    drop_flag = False  # 是否有舍的标志

    classes_ind_list = list(range(classes_num))  # 所有类编号列表

    if shuffle_flag:  # 如果要打乱
        random.shuffle(classes_ind_list)

    # 给每个类标记属于哪一个任务，label[i] = j -> i类在任务j中
    for task_i in range(tasks_num):
        l = task_i * per_task_class_num
        r = min((task_i + 1) * per_task_class_num, classes_num)  # 左闭右开

        if r < (task_i + 1) * per_task_class_num:
            drop_flag = True

        for i in range(l, r):
            label_task[classes_ind_list[i]] = task_i
            task_classes[task_i].append(classes_ind_list[i])

    if drop_flag:
        tasks_num = tasks_num - 1

    tasks = [{"train_mask": torch.zeros_like(train_mask).bool(),
              "valid_mask": torch.zeros_like(valid_mask).bool(),
              "test_mask": torch.zeros_like(test_mask).bool()} for _ in range(tasks_num)]

    # 把每个类的每个节点分到对应任务的对应集合中
    for i in range(classes_num):
        # 这个类的哪些节点属于训练、验证、测试集中
        class_i_train = train_mask & (data.y == i)
        class_i_valid = valid_mask & (data.y == i)
        class_i_test = test_mask & (data.y == i)
        if i not in label_task:  # 如果这个类被舍弃了
            continue
        task_i = label_task[i]

        tasks[task_i]["train_mask"] = tasks[task_i]["train_mask"] | class_i_train
        tasks[task_i]["valid_mask"] = tasks[task_i]["valid_mask"] | class_i_valid
        tasks[task_i]["test_mask"] = tasks[task_i]["test_mask"] | class_i_test

    # 把每个任务的所有类的子图保存下到对应任务的对应集合中
    for task_i in range(tasks_num):
        nodes_list = []
        for class_idx in task_classes[task_i]:
            nodes_list.extend(classes_nodes[class_idx])
        sub_graph = get_subgraph_by_node(data, nodes_list)

        tasks[task_i]["local_data"] = sub_graph

    if shuffle_flag:
        np.random.shuffle(tasks)

    return tasks

def get_client_task(data, clients_num, per_task_class_num, train_prop, valid_prop, test_prop, shuffle_flag=False):
    """获得每个客户端的数据"""
    # 每个客户端的子图
    clients_data = louvain_partitioner(data, clients_num)

    clients_tasks = {client_id: {"data": None,
                                 "task": None} for client_id in range(clients_num)}

    known_class_list = []

    for client_i in range(clients_num):
        client_data = clients_data[client_i]  # 加载数据
        clients_tasks[client_i]["data"] = client_data

        # 分割任务
        client_tasks = class_to_task(client_data, per_task_class_num, train_prop, valid_prop, test_prop, shuffle_flag)
        clients_tasks[client_i]["task"] = client_tasks

        for task_i in client_tasks:
            client_i_task_i_mask = task_i["train_mask"] | task_i["valid_mask"] | task_i["test_mask"]
            client_i_task_i_known_classes = torch.unique(client_data.y[client_i_task_i_mask])
            known_class_list.append(client_i_task_i_known_classes)

        print(f"client {client_i} has {len(clients_tasks[client_i]['task'])} tasks.")

    known_class = torch.unique(torch.cat(known_class_list))
    classes_used_num = known_class.shape[0]

    in_dim = data.x.shape[1]
    out_dim = classes_used_num

    if classes_used_num != data.y.max().item() + 1:
        print(f"DROPS {data.y.max().item() + 1 - classes_used_num} CLASS(ES).")

    return clients_tasks, in_dim, out_dim

# 主测试程序
if __name__ == "__main__":
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
        train_valid_test_split[2],
        shuffle_flag
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
