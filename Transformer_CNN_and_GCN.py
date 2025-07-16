import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import false
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from scipy.fftpack import fft
import scipy.io as scio
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
import networkx as nx
# 配置参数
class Config:
    num_sensors = 4
    seq_length = 2048
    input_channels = 1
    num_classes = 5
    batch_size = 64
    test_size = 0.2
    val_size = 0.1
    random_state = 42


# 1. 自定义数据集类
class SensorDataset(Dataset):
    def __init__(self, fft1,num_samples=5000):
        self.data = []
        self.labels = []
        def zscore(Z):
            Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
            Z = (Z - Zmin.reshape(-1, 1)) / (Zmax.reshape(-1, 1) - Zmin.reshape(-1, 1))
            return Z

        def min_max(Z):
            Zmin = Z.min(axis=1)

            Z = np.log(Z - Zmin.reshape(-1, 1) + 1)
            return Z

        def add_noise_to_data(data, snr_db):
            # 计算信号的功率
            signal_power = np.mean(np.square(data))
            # 将信噪比从 dB 转换为线性比例
            snr_linear = 10 ** (snr_db / 10)
            # 计算噪声的功率
            noise_power = signal_power / snr_linear
            # 生成具有指定功率的高斯噪声
            noise_std = np.sqrt(noise_power)
            noise = np.random.normal(0, noise_std, data.shape)
            # 将噪声添加到数据中
            noisy_data = data + noise
            return noisy_data



        #生成模拟数据
        # 源域数据 (正弦波+噪声)
        mean = 0  # 均值
        std = 0.05  # 标准差
        noise = np.random.normal(mean, std, (5000, 2048))
        sensor_1_data = scio.loadmat('beng_data\sensor01_beng.mat')
        sensor_2_data = scio.loadmat('beng_data\sensor02_beng.mat')
        sensor_3_data = scio.loadmat('beng_data\sensor03_beng.mat')
        sensor_4_data = scio.loadmat('beng_data\sensor04_beng.mat')

        #sensor_1_data_noisy = add_noise_to_data(sensor_1_data['sensor01_beng'],-12)
        #sensor_2_data_noisy = add_noise_to_data(sensor_2_data['sensor02_beng'], -12)
        #sensor_3_data_noisy = add_noise_to_data(sensor_3_data['sensor03_beng'], -12)
        sensor_4_data_noisy = add_noise_to_data(sensor_4_data['sensor04_beng'], -12)
        if fft1 == True:
            train_fea_1 = zscore((min_max(abs(fft(sensor_1_data['sensor01_beng']))[:, 0:1600])))
            train_fea_2 = zscore((min_max(abs(fft(sensor_2_data['sensor02_beng']))[:, 0:1600])))
            train_fea_3 = zscore((min_max(abs(fft(sensor_3_data['sensor03_beng']))[:, 0:1600])))
            train_fea_4 = zscore((min_max(abs(fft(sensor_4_data['sensor04_beng']))[:, 0:1600])))
        if fft1 == False:
            train_fea_1 = zscore(sensor_1_data['sensor01_beng'])
            train_fea_2 = zscore(sensor_2_data['sensor02_beng'])
            train_fea_3 = zscore(sensor_3_data['sensor03_beng'])
            train_fea_4 = zscore(sensor_4_data['sensor04_beng'])

        for i in range(1000 * 5):
            self.labels.append(i // 1000)
        data_group = [train_fea_1,train_fea_2,train_fea_3,train_fea_4]

        for col_idx in range(5000):
            sub_list = []
            for group in data_group:
                sub_list.append(group[col_idx, :].reshape(-1, 1))
            self.data.append(sub_list)



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回4个传感器的数据和标签
        sensor_data = [torch.FloatTensor(x) for x in self.data[idx]]
        label = torch.LongTensor([self.labels[idx]])
        return sensor_data, label.squeeze()


# 2. 数据划分与加载器
def create_dataloaders(dataset):
    # 先划分训练+测试，再从训练集中划分验证集
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=Config.test_size,
        random_state=Config.random_state,
        stratify=dataset.labels
    )

    train_val_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # 从训练集中划分验证集
    train_idx, val_idx = train_test_split(
        range(len(train_val_dataset)),
        test_size=Config.val_size / (1 - Config.test_size),
        random_state=Config.random_state,
        stratify=[dataset.labels[i] for i in train_idx]
    )

    train_dataset = torch.utils.data.Subset(train_val_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(train_val_dataset, val_idx)

    # 创建数据加载器
    def collate_fn(batch):
        sensor_data = list(zip(*[item[0] for item in batch]))
        labels = torch.stack([item[1] for item in batch])
        # 将每个传感器的数据堆叠成 [batch, seq_len, channels]
        sensor_data = [torch.stack(x) for x in sensor_data]
        return sensor_data, labels

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size,
                             collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


# 3. 模型定义
class PositionalEncoding(nn.Module):
    """Transformer位置编码"""
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)]
        return x


class SensorTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=8, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.d_model = d_model
        #self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.pool_transformer = nn.MaxPool1d(kernel_size=8)

        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, stride=1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = torch.transpose(x, 1, 2)
        x = self.conv_net(x)

        #x = self.pool_transformer(x)
        x = torch.transpose(x, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # 取序列的均值作为传感器特征
        return x.mean(dim=1)  # [batch_size, d_model]


class ParallelTransformer_GCN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 四个独立的Transformer分支
        self.transformer_branches = nn.ModuleList([
            SensorTransformer(input_dim=1, d_model=64) for _ in range(Config.num_sensors)
        ])





        self.adj_matrix = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ], dtype=torch.float32)  # 转换为浮点型

        # 元学习控制器
        self.meta_controller = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid())
        # 相似度计算参数
        self.W_s = nn.Linear(128, 1)  # 输入是两个节点特征的拼接
        self.b_s = nn.Parameter(torch.zeros(1)).cuda()

        # 残差连接的初始遗忘因子
        self.lambda_init = nn.Parameter(torch.tensor(0.5)).cuda()

        # 动态阈值参数
        self.k = 0

        # GCN融合模块
        self.gcn1 = GCNConv(64, 128)
        self.gcn2 = GCNConv(128, 256)

        self.classifier_loss_1 = nn.Sequential(
            nn.Linear(64, num_classes),
        )
        self.classifier_loss_2 = nn.Sequential(
            nn.Linear(64, num_classes),
        )
        self.classifier_loss_3 = nn.Sequential(
            nn.Linear(64, num_classes),
        )
        self.classifier_loss_4 = nn.Sequential(
            nn.Linear(64, num_classes),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        # 存储上一时刻的邻接矩阵
        self.A_prev = None

    def build_dynamic_adjacency(self, node_features, epoch):
        """构建动态邻接矩阵"""
        # node_features: [batch, num_sensors, feat_dim]
        self.A_base = nn.Parameter(torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ],dtype=torch.float32, requires_grad=True)).unsqueeze(0).cuda()

        batch_size, num_nodes, feat_dim = node_features.shape

        # 1. 计算初始相似度矩阵
        s_matrix = torch.zeros(batch_size, num_nodes, num_nodes, device=node_features.device)
        for i in range(num_nodes):
            for j in range(num_nodes):
                # 拼接节点特征
                pair_features = torch.cat([node_features[:, i], node_features[:, j]], dim=-1)
                # 计算相似度
                s_matrix[:, i, j] = torch.sigmoid(self.W_s(pair_features).squeeze() + self.b_s)

        # 2. 高阶拓扑传播
        # 计算度矩阵
        deg = s_matrix.sum(dim=2, keepdim=True)
        deg_inv = torch.eye(num_nodes, device=node_features.device).unsqueeze(0) * (1.0 / (deg + 1e-8))

        # 传播公式: S_hat = ReLU(S · D^{-1} · A_base)
        a = torch.bmm(s_matrix, deg_inv)
        a_shape = a.shape
        self.A_base.data =self.A_base.expand(a_shape[0], -1, -1).cuda()
        S_hat = torch.relu(
            torch.mul(a, self.A_base)
        )

        # 3. 动态邻接矩阵更新 (残差连接)
        if self.A_prev is None or self.A_prev.shape[0] != batch_size:
            self.A_prev = torch.eye(num_nodes, device=node_features.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # 元学习控制器动态调整lambda
        graph_embed = node_features.mean(dim=1)  # [batch, feat_dim]
        lambda_val = self.meta_controller(graph_embed) * self.lambda_init

        # 更新邻接矩阵: A = λ * A_prev + (1-λ) * S_hat
        lambda_val = lambda_val.unsqueeze(-1)
        A_next = lambda_val * self.A_prev + (1 - lambda_val) * S_hat

        # 4. 稀疏化约束
        tau = self.tau_0 * math.exp(-self.k * epoch)
        A_next[A_next < tau] = 0

        # 保存当前邻接矩阵供下一时刻使用
        self.A_prev = A_next.detach()

        return A_next

    def build_adjacency(self, features):
        """构建动态邻接矩阵"""
        # features: [batch, num_sensors, feat_dim]
        norm_features = F.normalize(features, p=2, dim=-1)
        adj = torch.bmm(norm_features, norm_features.transpose(1, 2))
        return F.softmax(adj, dim=-1)

    def plot_graph_structure(self,adj, node_features, title="Graph Structure"):
        """绘制传感器关系图"""
        G = nx.DiGraph()
        adj_np = adj.cpu().detach().numpy()

        # 添加节点（传感器）
        for i in range(Config.num_sensors):
            G.add_node(f"Sensor{i + 1}", feature=node_features[i].mean().item())

        # 添加边（权重>0.1的连接）
        for i in range(Config.num_sensors):
            for j in range(Config.num_sensors):
                if adj_np[i, j] > 0.1:  # 过滤弱连接
                    G.add_edge(f"Sensor{i + 1}", f"Sensor{j + 1}", weight=adj_np[i, j])

        # 绘制图形
        plt.figure(figsize=(10, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        pos = nx.spring_layout(G)
        node_colors = [0.1, 0.25, 0.5, 0.9]
        edge_weights = [G.edges[e]['weight'] * 5 for e in G.edges]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               cmap=plt.cm.viridis, node_size=2000)
        nx.draw_networkx_edges(G, pos, width=2.5,
                               edge_color='black', arrows=False)
        nx.draw_networkx_labels(G, pos, font_size=16,font_color='red')

        # 添加边权重标签
        edge_labels = {(u, v): f"{G.edges[(u, v)]['weight']:.2f}"
                       for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        #plt.title(f"{title} (Epoch {epoch})", fontsize=14)
        #plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis),label="Node Feature Mean")
        plt.savefig(f"graph_structure.png", dpi=350, bbox_inches='tight')
        plt.close()

    def forward(self, sensor_data,epoch,return_features=False):
        # sensor_data: 包含num_sensors个[batch, seq_len, ch]张量的列表
        fixed_adj = self.adj_matrix.to(sensor_data[0].device)

        # 特征提取
        features = [branch(data) for branch, data in zip(self.transformer_branches, sensor_data)]
        sensor_1_label = self.classifier_loss_1(features[0])
        sensor_2_label = self.classifier_loss_1(features[1])
        sensor_3_label = self.classifier_loss_1(features[2])
        sensor_4_label = self.classifier_loss_1(features[3])


        node_features = torch.stack(features, dim=1)  # [batch, num_sensors, 64]
        batch_size = node_features.size(0)
        dynamic_weights = torch.sigmoid(torch.bmm(node_features, node_features.transpose(1, 2)))
        # 构建图结构
        #adj = self.build_adjacency(node_features)  # [batch, num_sensors, num_sensors]
        adj = self.build_dynamic_adjacency(node_features, epoch)
        #adj = fixed_adj.unsqueeze(0) * dynamic_weights


        #self.plot_graph_structure(adj[2], node_features[2])

        # 准备GCN输入
        weight = adj[0]
        edge_indices = []
        for b in range(batch_size):
            # 全连接图的边索引
            src, dst = torch.meshgrid(torch.arange(Config.num_sensors),
                                      torch.arange(Config.num_sensors))
            edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
            edge_indices.append(edge_index + b * Config.num_sensors)

        edge_index = torch.cat(edge_indices, dim=1).to(node_features.device)
        edge_weight = adj.reshape(-1)  # [batch*num_sensors^2]
        x = node_features.reshape(-1, 64)  # [batch*num_sensors, 64]

        # 图卷积
        x = F.relu(self.gcn1(x, edge_index, edge_weight=edge_weight))
        graph_features = self.gcn2(x, edge_index, edge_weight=edge_weight)  # [batch*num_sensors, 256]

        # 全局池化
        x = graph_features.view(batch_size, Config.num_sensors, -1)  # [batch, num_sensors, 256]
        x = torch.mean(x, dim=1)  # [batch, 256]
        if return_features:
            return self.classifier(x), x, graph_features,features,sensor_1_label,sensor_2_label,sensor_3_label,sensor_4_label
        return self.classifier(x)


# 4. 训练与验证函数
def train_epoch(model, loader, optimizer, criterion, device,epoch):
    model.train()
    total_loss = 0
    correct = 0
    all_features =[]
    all_labels = []

    for sensor_data, labels in loader:
        sensor_data = [x.to(device) for x in sensor_data]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs,_,_,features,sensor_1_label,sensor_2_label,sensor_3_label,sensor_4_label = model(sensor_data,epoch,return_features=True)
        loss_total = criterion(outputs, labels)
        loss_sensor_1 = criterion(sensor_1_label, labels)
        loss_sensor_2 = criterion(sensor_2_label, labels)
        loss_sensor_3 = criterion(sensor_3_label, labels)
        loss_sensor_4 = criterion(sensor_4_label, labels)
        loss = loss_total+loss_sensor_1+loss_sensor_2+loss_sensor_3+loss_sensor_4



        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        all_features.append(features)
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    all_labels_cpu = all_labels.cpu()
    all_labels_cpu = all_labels_cpu.numpy()

    return total_loss / len(loader), correct / len(loader.dataset),all_features,all_labels_cpu


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_features = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for sensor_data, labels in loader:
            sensor_data = [x.to(device) for x in sensor_data]
            labels = labels.to(device)

            outputs,_,_,features,sensor_1_label,sensor_2_label,sensor_3_label,sensor_4_label = model(sensor_data,epoch,return_features=True)
            loss_total = criterion(outputs, labels)
            loss_sensor_1 = criterion(sensor_1_label, labels)
            loss_sensor_2 = criterion(sensor_2_label, labels)
            loss_sensor_3 = criterion(sensor_3_label, labels)
            loss_sensor_4 = criterion(sensor_4_label, labels)

            loss = loss_total + loss_sensor_1 + loss_sensor_2 + loss_sensor_3 + loss_sensor_4
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            all_features.append(features)
            all_labels.append(labels)
            all_outputs.append(outputs)

        all_labels = torch.cat(all_labels, dim=0)
        all_labels_cpu = all_labels.cpu().numpy()
        all_outputs = torch.cat(all_outputs, dim=0)
        all_outputs_cpu = all_outputs.cpu().numpy()

    return total_loss / len(loader), correct / len(loader.dataset),all_features,all_labels_cpu,all_outputs_cpu

# ================== 数据准备 ==================
def plot_tsne(features, labels, num_classes,enpoch, title="t-SNE Visualization"):
    """绘制t-SNE特征分布图"""
    handles = []
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    colors = ['r', 'g', 'b', 'c', 'm']
    label_mapping = {0: 'Normal', 1: 'Misalignment fault', 2:'Gas-containing fault', 3:'Outer ring fault',4:'Inner ring fault'}
    tsne = TSNE(n_components=2, random_state=Config.random_state)
    embeddings = tsne.fit_transform(features)
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in np.unique(labels):
        # 筛选出当前标签的数据
        mask = labels == label
        # 绘制散点图
        #scatter = ax.scatter(embeddings[mask, 0], embeddings[mask, 1], c=colors[label], label=label_mapping[label])
        scatter = ax.scatter(embeddings[mask, 0], embeddings[mask, 1], c=colors[label])
        # 存储散点图句柄
        handles.append(scatter)
    ax.tick_params(axis='both', direction='in')
    ax.legend(handles=handles, fontsize=16)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    fig.savefig(f"tsne_visualization_{title}_{enpoch}.png", dpi=350, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, num_classes,enpoch,title="Confusion Matrix"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes),annot_kws={"size": 16})
    ax.set_xlabel('Predictive label', fontsize=20)
    ax.set_ylabel('True label', fontsize=20)
    ax.tick_params(axis='both', direction='in')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    fig.savefig(f"confusion_matrix_{enpoch}.png", dpi=350, bbox_inches='tight')
    plt.close()

def test(model, test_loader, criterion, device, visualize,enpoch):
    model.eval()
    total_loss = 0
    correct = 0
    all_features = []
    all_graph_features = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sensor_data, labels in tqdm(test_loader, desc="Testing"):
            sensor_data = [x.to(device) for x in sensor_data]
            labels = labels.to(device)

            outputs, features_x, graph_features,features,sensor_1_label,sensor_2_label,sensor_3_label,sensor_4_label = model(sensor_data,epoch, return_features=True)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            # 收集特征和预测结果
            all_features.append(features_x)
            all_graph_features.append(graph_features)
            all_labels.append(labels)
            all_preds.append(preds)

    # 合并所有批次的结果
    features = torch.cat(all_features, dim=0)
    features_cpu = features.cpu().numpy()
    graph_features = torch.cat(all_graph_features, dim=0)
    graph_features_cpu = graph_features.cpu().numpy()
    labels = torch.cat(all_labels, dim=0)
    labels_cpu = labels.cpu().numpy()
    preds = torch.cat(all_preds, dim=0)
    preds_cpu = preds.cpu().numpy()

    # 可视化
    if visualize:
        print("Generating visualizations...")
        plot_tsne(features_cpu, labels_cpu, Config.num_classes, enpoch,"Pooled")
        plot_tsne(graph_features_cpu, np.repeat(labels_cpu, 4), Config.num_classes, enpoch,"Node Features ")
        plot_confusion_matrix(labels_cpu, preds_cpu,Config.num_classes,enpoch)

    return total_loss / len(test_loader), correct / len(test_loader.dataset)
def tsne_test(features,labels, epoch):
    all_data = []
    all_labels = []
    sensor1_feature = []

    sensor2_feature = []

    sensor3_feature = []


    sensor4_feature = []
    all_data = features
    all_labels =labels

    for j in range(len(all_data)):
        sensor1_feature.append(all_data[j][0])
        sensor2_feature.append(all_data[j][1])
        sensor3_feature.append(all_data[j][2])
        sensor4_feature.append(all_data[j][3])
    sensor1_feature = torch.cat(sensor1_feature, dim=0).reshape(-1,64)
    sensor1_feature_cpu = sensor1_feature.cpu()
    sensor1_feature = np.array(sensor1_feature_cpu)

    sensor2_feature = torch.cat(sensor2_feature, dim=0).reshape(-1,64)
    sensor2_feature_cpu = sensor2_feature.cpu()
    sensor2_feature = np.array(sensor2_feature_cpu)

    sensor3_feature = torch.cat(sensor3_feature, dim=0).reshape(-1,64)
    sensor3_feature_cpu = sensor3_feature.cpu()
    sensor3_feature = np.array(sensor3_feature_cpu)

    sensor4_feature = torch.cat(sensor4_feature, dim=0).reshape(-1,64)
    sensor4_feature_cpu = sensor4_feature.cpu()
    sensor4_feature = np.array(sensor4_feature_cpu)

    def sensor_tsne(sensor1_feature,all_labels,sensor):
        handles = []
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False
        colors = ['r', 'g', 'b', 'c', 'm']
        label_mapping = {0: 'Normal', 1: 'Misalignment fault', 2:'Gas-containing fault', 3:'Outer ring fault',4:'Inner ring fault'}
        tsne = TSNE(n_components=2, random_state=Config.random_state)
        embeddings = tsne.fit_transform(sensor1_feature)
        fig, ax = plt.subplots(figsize=(10, 8))
        for label in np.unique(all_labels):
            # 筛选出当前标签的数据
            mask = all_labels == label
            # 绘制散点图
            #scatter = ax.scatter(embeddings[mask, 0], embeddings[mask, 1], c=colors[label], label=label_mapping[label])
            scatter = ax.scatter(embeddings[mask, 0], embeddings[mask, 1], c=colors[label])
            # 存储散点图句柄
            handles.append(scatter)
        ax.tick_params(axis='both', direction='in')
        ax.legend(handles=handles,fontsize=16)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        fig.savefig(f"{sensor}_{epoch}_tsne.png", dpi=350, bbox_inches='tight')
        plt.close()

    sensor_tsne(sensor1_feature,all_labels,sensor='sensor1')
    sensor_tsne(sensor2_feature, all_labels,sensor='sensor2')
    sensor_tsne(sensor3_feature, all_labels,sensor='sensor3')
    sensor_tsne(sensor4_feature, all_labels,sensor='sensor4')
def mean_hist(features,labels, epoch):
    feature_0 = []
    feature_1 = []
    feature_2 = []
    feature_3 = []
    feature_4 = []
    for j in range(len(labels)):
        if labels[j]==0:
           feature_0.append(features[j])
        if labels[j] == 1:
           feature_1.append(features[j])
        if labels[j] == 2:
           feature_2.append(features[j])
        if labels[j] == 3:
           feature_3.append(features[j])
        if labels[j] == 4:
           feature_4.append(features[j])

    sensor_1_std_0 = np.mean(np.array(feature_0).T, axis=0)
    sensor_1_std_1 = np.mean(np.array(feature_1).T, axis=0)
    sensor_1_std_2 = np.mean(np.array(feature_2).T, axis=0)
    sensor_1_std_3 = np.mean(np.array(feature_3).T, axis=0)
    sensor_1_std_4 = np.mean(np.array(feature_4).T, axis=0)

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    #fig, ax = plt.figure()
    n1, bins1, patches1 = plt.hist(sensor_1_std_0, bins=80, density=True, alpha=1, label='Noraml', color='blue')
    n2, bins2, patches2 = plt.hist(sensor_1_std_1, bins=80, density=True, alpha=1, label='Misalignment fault', color='red')
    n3, bins3, patches3 = plt.hist(sensor_1_std_2, bins=80, density=True, alpha=1, label='Gas-containing fault', color='green')
    n4, bins4, patches4 = plt.hist(sensor_1_std_3, bins=80, density=True, alpha=1, label='Outer ring fault', color='yellow')
    n5, bins5, patches5 = plt.hist(sensor_1_std_4, bins=80, density=True, alpha=1, label='Inner ring fault', color='orange')
    plt.legend(fontsize=16)
    plt.xlabel('Standard deviation',fontsize=20)
    plt.ylabel('Frequency',fontsize=20)
    plt.tight_layout()
    plt.savefig(f'Fusion_feature_{epoch}.png')
    plt.close()

def mean_hist_5(features,labels, epoch):
    all_data = []
    all_labels = []
    sensor1_feature = []
    sensor1_feature_0 = []
    sensor1_feature_1 = []
    sensor1_feature_2 = []
    sensor1_feature_3 = []
    sensor1_feature_4 = []

    sensor2_feature = []
    sensor2_feature_0 = []
    sensor2_feature_1 = []
    sensor2_feature_2 = []
    sensor2_feature_3 = []
    sensor2_feature_4 = []

    sensor3_feature = []
    sensor3_feature_0 = []
    sensor3_feature_1 = []
    sensor3_feature_2 = []
    sensor3_feature_3 = []
    sensor3_feature_4 = []

    sensor4_feature = []
    sensor4_feature_0 = []
    sensor4_feature_1 = []
    sensor4_feature_2 = []
    sensor4_feature_3 = []
    sensor4_feature_4 = []
    all_data = features
    all_labels =labels

    for j in range(len(all_data)):
        sensor1_feature.append(all_data[j][0])
        sensor2_feature.append(all_data[j][1])
        sensor3_feature.append(all_data[j][2])
        sensor4_feature.append(all_data[j][3])
    sensor1_feature = torch.cat(sensor1_feature, dim=0).reshape(-1,64)
    sensor1_feature_cpu = sensor1_feature.cpu()
    sensor1_feature = np.array(sensor1_feature_cpu)

    sensor2_feature = torch.cat(sensor2_feature, dim=0).reshape(-1,64)
    sensor2_feature_cpu = sensor2_feature.cpu()
    sensor2_feature = np.array(sensor2_feature_cpu)

    sensor3_feature = torch.cat(sensor3_feature, dim=0).reshape(-1,64)
    sensor3_feature_cpu = sensor3_feature.cpu()
    sensor3_feature = np.array(sensor3_feature_cpu)

    sensor4_feature = torch.cat(sensor4_feature, dim=0).reshape(-1,64)
    sensor4_feature_cpu = sensor4_feature.cpu()
    sensor4_feature = np.array(sensor4_feature_cpu)

    for i in range(len(all_labels)):
        if all_labels[i] == 0:
            sensor1_feature_0.append(sensor1_feature[i])
            sensor2_feature_0.append(sensor2_feature[i])
            sensor3_feature_0.append(sensor3_feature[i])
            sensor4_feature_0.append(sensor4_feature[i])
        elif all_labels[i] == 1:
            sensor1_feature_1.append(sensor1_feature[i])
            sensor2_feature_1.append(sensor2_feature[i])
            sensor3_feature_1.append(sensor3_feature[i])
            sensor4_feature_1.append(sensor4_feature[i])
        elif all_labels[i] == 2:
            sensor1_feature_2.append(sensor1_feature[i])
            sensor2_feature_2.append(sensor2_feature[i])
            sensor3_feature_2.append(sensor3_feature[i])
            sensor4_feature_2.append(sensor4_feature[i])
        elif all_labels[i] == 3:
            sensor1_feature_3.append(sensor1_feature[i])
            sensor2_feature_3.append(sensor2_feature[i])
            sensor3_feature_3.append(sensor3_feature[i])
            sensor4_feature_3.append(sensor4_feature[i])
        elif all_labels[i] == 4:
            sensor1_feature_4.append(sensor1_feature[i])
            sensor2_feature_4.append(sensor2_feature[i])
            sensor3_feature_4.append(sensor3_feature[i])
            sensor4_feature_4.append(sensor4_feature[i])

    sensor_1_std_0 = np.mean(np.array(sensor1_feature_0).T, axis=0)
    sensor_1_std_1 = np.mean(np.array(sensor1_feature_1).T, axis=0)
    sensor_1_std_2 = np.mean(np.array(sensor1_feature_2).T, axis=0)
    sensor_1_std_3 = np.mean(np.array(sensor1_feature_3).T, axis=0)
    sensor_1_std_4 = np.mean(np.array(sensor1_feature_4).T, axis=0)

    sensor_2_std_0 = np.mean(np.array(sensor2_feature_0).T, axis=0)
    sensor_2_std_1 = np.mean(np.array(sensor2_feature_1).T, axis=0)
    sensor_2_std_2 = np.mean(np.array(sensor2_feature_2).T, axis=0)
    sensor_2_std_3 = np.mean(np.array(sensor2_feature_3).T, axis=0)
    sensor_2_std_4 = np.mean(np.array(sensor2_feature_4).T, axis=0)

    sensor_3_std_0 = np.mean(np.array(sensor3_feature_0).T, axis=0)
    sensor_3_std_1 = np.mean(np.array(sensor3_feature_1).T, axis=0)
    sensor_3_std_2 = np.mean(np.array(sensor3_feature_2).T, axis=0)
    sensor_3_std_3 = np.mean(np.array(sensor3_feature_3).T, axis=0)
    sensor_3_std_4 = np.mean(np.array(sensor3_feature_4).T, axis=0)

    sensor_4_std_0 = np.mean(np.array(sensor4_feature_0).T, axis=0)
    sensor_4_std_1 = np.mean(np.array(sensor4_feature_1).T, axis=0)
    sensor_4_std_2 = np.mean(np.array(sensor4_feature_2).T, axis=0)
    sensor_4_std_3 = np.mean(np.array(sensor4_feature_3).T, axis=0)
    sensor_4_std_4 = np.mean(np.array(sensor4_feature_4).T, axis=0)

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    #fig, ax = plt.figure()
    n1, bins1, patches1 = plt.hist(sensor_1_std_0, bins=80, density=True, alpha=1, label='Sensor1_Noraml', color='blue')
    n2, bins2, patches2 = plt.hist(sensor_1_std_1, bins=80, density=True, alpha=1, label='Sensor1_Misalignment fault', color='red')
    n3, bins3, patches3 = plt.hist(sensor_1_std_2, bins=80, density=True, alpha=1, label='Sensor1_Gas-containing fault', color='green')
    n4, bins4, patches4 = plt.hist(sensor_1_std_3, bins=80, density=True, alpha=1, label='Sensor1_Outer ring fault', color='yellow')
    n5, bins5, patches5 = plt.hist(sensor_1_std_4, bins=80, density=True, alpha=1, label='Sensor1_Inner ring fault', color='orange')
    plt.legend(fontsize=16)
    plt.xlabel('Standard deviation',fontsize=20)
    plt.ylabel('Frequency',fontsize=20)
    plt.tight_layout()
    plt.savefig(f'sensor1_{epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    n6, bins6, patches6 = plt.hist(sensor_2_std_0, bins=80, density=True, alpha=1, label='Sensor2_Noraml', color='blue')
    n7, bins7, patches7 = plt.hist(sensor_2_std_1, bins=80, density=True, alpha=1, label='Sensor2_Misalignment fault', color='red')
    n8, bins8, patches8 = plt.hist(sensor_2_std_2, bins=80, density=True, alpha=1, label='Sensor2_Gas-containing fault', color='green')
    n9, bins9, patches9 = plt.hist(sensor_2_std_3, bins=80, density=True, alpha=1, label='Sensor2_Outer ring fault', color='yellow')
    n10, bins10, patches10 = plt.hist(sensor_2_std_4, bins=80, density=True, alpha=1, label='Sensor2_Inner ring fault', color='orange')

    plt.legend(fontsize=16)
    plt.xlabel('Standard deviation',fontsize=20)
    plt.ylabel('Frequency',fontsize=20)
    plt.tight_layout()
    plt.savefig(f'sensor2_{epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    n11, bins11, patches11 = plt.hist(sensor_3_std_0, bins=80, density=True, alpha=1, label='Sensor3_Noraml', color='blue')
    n12, bins12,patches12 = plt.hist(sensor_3_std_1, bins=80, density=True, alpha=1, label='Sensor3_Misalignment fault', color='red')
    n13, bins13, patches13 = plt.hist(sensor_3_std_2, bins=80, density=True, alpha=1, label='Sensor3_Gas-containing fault', color='green')
    n14, bins14, patches14 = plt.hist(sensor_3_std_3, bins=80, density=True, alpha=1, label='Sensor3_Outer ring fault', color='yellow')
    n15, bins15, patches15 = plt.hist(sensor_3_std_4, bins=80, density=True, alpha=1, label='Sensor3_Inner ring fault', color='orange')

    plt.legend(fontsize=16)
    plt.xlabel('Standard deviation',fontsize=20)
    plt.ylabel('Frequency',fontsize=20)
    plt.tight_layout()
    plt.savefig(f'sensor3_{epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    n16, bins16, patches16 = plt.hist(sensor_4_std_0, bins=80, density=True, alpha=1, label='Sensor4_Noraml', color='blue')
    n17, bins17, patches17 = plt.hist(sensor_4_std_1, bins=80, density=True, alpha=1, label='Sensor4_Misalignment fault', color='red')
    n18, bins18, patches18 = plt.hist(sensor_4_std_2, bins=80, density=True, alpha=1, label='Sensor4_Gas-containing fault', color='green')
    n19, bins19, patches19 = plt.hist(sensor_4_std_3, bins=80, density=True, alpha=1, label='Sensor4_Outer ring fault', color='yellow')
    n20, bins20, patches20 = plt.hist(sensor_4_std_4, bins=80, density=True, alpha=1, label='Sensor4_Inner ring fault', color='orange')

    #plt.title(f'Feature distribution (Epoch {epoch})')
    plt.legend(fontsize=16)
    plt.xlabel('Standard deviation',fontsize=20)
    plt.ylabel('Frequency',fontsize=20)
    plt.tight_layout()
    plt.savefig(f'sensor4_{epoch}.png')
    plt.close()



# 5. 主程序
if __name__ == "__main__":
    # 初始化
    fft = False
    visualize = True
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SensorDataset(fft, num_samples=5000)
    train_loader, val_loader, test_loader = create_dataloaders(dataset)

    model = ParallelTransformer_GCN(num_classes=Config.num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_val_acc = 0
    tra_acc_all = []
    val_acc_all = []
    tra_loss_all = []
    val_loss_all = []
    epoch_all =[]
    for epoch in range(1, 50):
        val_loss, val_acc,all_features,all_labels,all_outputs = evaluate(model, test_loader, criterion, device)
        if epoch % 5 == 0 or epoch == 1:
            _, _ = test(model, val_loader, criterion, device,visualize,epoch)
            mean_hist_5(all_features,all_labels,epoch)
            tsne_test(all_features,all_labels,epoch)
            mean_hist(all_outputs,all_labels,epoch)

        train_loss, train_acc,_,_ = train_epoch(model, train_loader, optimizer, criterion, device,epoch)


        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        tra_acc_all.append(train_acc)
        val_acc_all.append(val_acc)
        tra_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
        epoch_all.append(epoch)



            # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_GCN_1.pth")

    # 最终测试
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(epoch_all,tra_acc_all,label="Training set accuracy curve", color="red", linestyle="--", marker="o")
    plt.plot(epoch_all,val_acc_all,label="Test set accuracy curve", color="green", linestyle="--", marker="*")
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.xlabel('Epoch',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.savefig(f'Train_and_val_accury.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(epoch_all,tra_loss_all,label="Training set loss curve", color="red", linestyle="--", marker="o")
    plt.plot(epoch_all,val_loss_all,label="Test set loss curve", color="green", linestyle="--", marker="*")
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.xlabel('Epoch',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.savefig(f'Train_and_val_loss.png')
    plt.close()

    epoch=1

    model.load_state_dict(torch.load("noise_model_GCN.pth"))
    test_loss, test_acc,all_features,all_labels,all_outputs = evaluate(model, val_loader, criterion, device)
    mean_hist_5(all_features, all_labels, epoch)
    #tsne_test(all_features, all_labels, epoch)
    #mean_hist(all_outputs, all_labels, epoch)
    _, _ = test(model, val_loader, criterion, device, visualize, epoch)
    print(f"\nTest Accuracy: {test_acc:.4f}")