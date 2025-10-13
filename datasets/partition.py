from community import community_louvain
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch_geometric.data import Data



def get_subgraph_by_node(dataset, node_list, flag):   #node_list是节点在dataset图中的索引
    node_id_set = set(node_list)
    global_id_to_local_id = {}
    local_id_to_global_id = []
    local_edge_list = []
    for local_id, global_id in enumerate(node_list):
        global_id_to_local_id[global_id] = local_id
        local_id_to_global_id.append(global_id)

    for edge_id in range(dataset.edge_index.shape[1]):          #遍历dataset中的所有边
        src = dataset.edge_index[0, edge_id].item()
        tgt = dataset.edge_index[1, edge_id].item()
        if src in node_id_set and tgt in node_id_set:
            local_id_src = global_id_to_local_id[src]
            local_id_tgt = global_id_to_local_id[tgt]
            local_edge_list.append((local_id_src, local_id_tgt))

    local_edge_index = torch.tensor(local_edge_list).t()
    if not local_edge_list:
        local_edge_index = torch.empty((2, 0), dtype=torch.int64)
    if flag:        #为True表示不要每边的点
        local_subgraph = Data(x=dataset.x[node_list], edge_index=local_edge_index, y=dataset.y[node_list])
    else:
        local_subgraph = Data(x=dataset.x, edge_index=local_edge_index, y=dataset.y)
    local_subgraph.global_map = local_id_to_global_id

    return local_subgraph

def louvain_partitioner(data, num_clients):     
    G = nx.Graph()                          #空的无向图对象
    for i in range(data.num_nodes):         #遍历原图的每一个节点，并将它们添加到新创建的NetwoekX图中
        G.add_node(i)
   
   
    edges = data.edge_index.numpy()         #提取边索引，并转换为Numpy数组   第一行为边的源节点，第二行是边的目标节点
    for i in range(edges.shape[1]):         #edge.shape[1]边的数量
        G.add_edge(edges[0, i], edges[1, i])    #先NetworkX图中添加边
    
    for i, feature in enumerate(data.x.numpy()):    #遍历每个节点及其特征
        G.nodes[i]['feature'] = feature             #将节点特征加入图中


    
    le = LabelEncoder()                 #一个LabelEncoder的对象，将可能是字符串或其他类型的类别标签编码为整数
    labels = le.fit_transform(data.y.numpy())       #将节点标签张良转换为Numpy数组然后转为整数
    for i, label in enumerate(labels):              
        G.nodes[i]['label'] = label

    partition = community_louvain.best_partition(G)     #执行Louvain社区发现算法
                                                        #返回一个字典，键是节点ID,值是该节点所属的社区ID
   
    community_dict = {}                                 #键是社区ID，值是属于该社区的节点列表
    #将节点按社区划分
    for node, community in partition.items():           
        if community not in community_dict:
            community_dict[community] = []
        community_dict[community].append(node)

    num_communities = len(community_dict)               #社区数量

    clients_nodes = [[] for _ in range(num_clients)]    #存储分给一个客户端的节点ID

    # while len(community_dict) < num_communities:
    while len(community_dict) < num_clients:            #将最大的社区分为两份，增加社区数量 
        community_sizes = [len(community_dict[i]) for i in range(len(community_dict))]
        max_len = max(community_sizes)
        min_len = min(community_sizes)
        max_index = np.argmax(community_sizes)
        if max_len < 2 * min_len:
            min_len = max_len // 2
        max_len_nodes = community_dict[max_index]
        new_list_id = len(community_dict)
        community_dict[new_list_id] = max_len_nodes[:min_len]
        community_dict[max_index] = max_len_nodes[min_len:]
    community_sizes = [len(community_dict[i]) for i in range(len(community_dict))]
    #社区数量从小到大进行分配
    community_ids = np.argsort(community_sizes)
    for comid in community_ids:                             #数量最少的客户端先分配
        clid = np.argmin([len(cs) for cs in clients_nodes])
        clients_nodes[clid].extend(community_dict[comid])

    #为每个客户端构建子图
    clients_data = []
    for nodes in clients_nodes:
        node_map = {node: i for i, node in enumerate(nodes)}            #全局节点id到局部节点id的映射
        sub_edge_index = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:                               #检查边的两个端点是不是都在当前客户端的节点集合中
                sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])             #将这条边的两个端点转换为局部节点ID，并添加到sub_edge_index中
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()                                            #边列表改为pytorch张量，并转置，各式为[2, num_edges]


        sub_x = data.x[nodes]           #当前客户端节点的特征和标签
        sub_y = data.y[nodes]
        sub_data = Data(x=sub_x, edge_index=sub_edge_index,y =sub_y)        #当前客户端的子图

        # if hasattr(data, "train_mask"):
        #     train_mask = torch.zeros(len(nodes), dtype=torch.bool)
        #     for node in nodes:
        #         train_mask[node_map[node]] = data.train_mask[node]
        #     sub_data.train_mask = train_mask

        # if hasattr(data, "val_mask"):
        #     val_mask = torch.zeros(len(nodes), dtype=torch.bool)
        #     for node in nodes:
        #         val_mask[node_map[node]] = data.val_mask[node]
        #     sub_data.val_mask = val_mask

        # if hasattr(data, "test_mask"):
        #     test_mask = torch.zeros(len(nodes), dtype=torch.bool)
        #     for node in nodes:
        #         test_mask[node_map[node]] = data.test_mask[node]
        #     sub_data.test_mask = test_mask

        clients_data.append(sub_data)

    # import matplotlib.pyplot as plt
    # listy = data.y.tolist()
    # num_classes = len(np.unique(listy))
    # label_distribution = [[] for _ in range(num_classes)]
    # for cid, client_data in enumerate(clients_data):
    #     for label in client_data.y:
    #         label_distribution[label].append(cid)
    # plt.hist(label_distribution, stacked=True, label=range(num_classes))
    # plt.xlabel("client_id")
    # plt.ylabel("num_samples")
    # plt.show()

    return clients_data


def dirichlet_partitioner1(data, num_clients, alpha, a, b):
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
    edges = data.edge_index.numpy()
    for i in range(edges.shape[1]):
        G.add_edge(edges[0, i], edges[1, i])

    for i, feature in enumerate(data.x.numpy()):
        G.nodes[i]['feature'] = feature


    le = LabelEncoder()
    labels = le.fit_transform(data.y.numpy())
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label

    labels = nx.get_node_attributes(G, 'label')
    unique_labels = list(set(labels.values()))
    num_classes = len(unique_labels)

    label_distribution = np.random.dirichlet([alpha]*num_clients, num_classes)
    class_indices = [[] for _ in range(num_classes)]
    for i in range(data.num_nodes):
        class_indices[data.y[i]].append(i)

    clients_nodes = [[] for _ in range(num_clients)]
    for k_idcs, fracs in zip(class_indices, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            clients_nodes[i].extend(idcs)




    clients_data = []
    for nodes in clients_nodes:
        node_map = {node: i for i, node in enumerate(nodes)}

        sub_edge_index = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:
                sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()


        sub_x = data.x[nodes]
        sub_y = data.y[nodes]

        sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)

        if hasattr(data, "train_mask"):
            train_mask = torch.zeros(len(nodes), dtype=torch.bool)
            train_mask1 = data.train_mask.tolist()
            for node in nodes:

                train_mask[node_map[node]] = train_mask1[node]
            sub_data.train_mask = train_mask

        if hasattr(data, "val_mask"):
            val_mask = torch.zeros(len(nodes), dtype=torch.bool)
            val_mask1 = data.val_mask.tolist()
            for node in nodes:

                val_mask[node_map[node]] = val_mask1[node]
            sub_data.val_mask = val_mask

        if hasattr(data, "test_mask"):
            test_mask = torch.zeros(len(nodes), dtype=torch.bool)
            test_mask1 = data.test_mask.tolist()
            for node in nodes:

                test_mask[node_map[node]] = test_mask1[node]
            sub_data.test_mask = test_mask

        clients_data.append(sub_data)

    import matplotlib.pyplot as plt
    label_distribution = [[] for _ in range(num_classes)]
    for cid, client_data in enumerate(clients_data):
        for label in client_data.y:
            label_distribution[label].append(cid)
    plt.hist(label_distribution, stacked=True, label=range(num_classes))
    plt.xlabel("client_id")
    plt.ylabel("num_samples")
    plt.show()

    return clients_nodes


def dirichlet_partitioner(data, num_clients, alpha, least_samples, dirichlet_try_cnt):
    graph_labels = data.y.numpy()
    num_clients = num_clients
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)

    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")

    min_size = 0
    K = len(unique_labels)
    N = graph_labels.shape[0]

    client_indices = [[] for _ in range(num_clients)]

    try_cnt = 0
    while min_size < least_samples:
        if try_cnt > dirichlet_try_cnt:
            print("alpha("+str(alpha)+") is too small, no solution")
            break

        client_indices = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(graph_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_indices = [idx_j + idx.tolist() for idx_j, idx in zip(client_indices, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in client_indices])
        try_cnt += 1

    clients_data = []
    for nodes in client_indices:
        list.sort(nodes)
        node_map = {node: i for i, node in enumerate(nodes)}
        
        sub_edge_index = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:
                sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()


        sub_x = data.x[nodes]
        sub_y = data.y[nodes]

        sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)



        clients_data.append(sub_data)

    num_classes = len(unique_labels)
    import matplotlib.pyplot as plt
    label_distribution = [[] for _ in range(num_classes)]
    for cid, client_data in enumerate(clients_data):
        for label in client_data.y:
            label_distribution[label].append(cid)
    plt.hist(label_distribution, stacked=True, label=range(num_classes))
    plt.xlabel("client_id")
    plt.ylabel("num_samples")
    plt.show()

    return client_indices


