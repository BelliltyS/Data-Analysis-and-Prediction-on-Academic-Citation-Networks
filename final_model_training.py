import requests
import os
from torch_geometric.data import Dataset
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

NUM_CLASSES = 40

class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_url = self.url.replace(' ', '%20')
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file, status code: {response.status_code}")

        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            f.write(response.content)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])


class GATModel(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_dim, num_heads):
        super(GATModel, self).__init__()

        self.conv1 = GATConv(dataset.num_features, hidden_dim, heads=num_heads)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
            for _ in range(num_layers - 1)
        ])
        self.lin = nn.Linear(hidden_dim * num_heads, NUM_CLASSES)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x.to(device), edge_index.to(device)))

        for conv in self.convs:
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(conv(x.to(device), edge_index.to(device)))

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask].squeeze())  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss.to('cpu')


def test(mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[data.val_mask] == data.y[data.val_mask].squeeze()  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(correct.shape[0])  # Derive ratio of correct predictions.
    return acc


if __name__ == '__main__':
    #dataset = HW3Dataset(root='data/hw3/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #data = dataset[0].to(device)
    processed_paths_data = 'data/hw3/processed/data.pt'
    dataset = torch.load(processed_paths_data)
    data = dataset.to(device)
    print(data)

    print("data \n",data)

    # Normalize the input features
    x_mean = data.x.mean(dim=0)
    x_std = data.x.std(dim=0)
    data.x = (data.x - x_mean) / x_std


    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(f'training nodes: {data.train_mask}')
    print(f'training nodes: {data.y}')
    print(f'features nodes: {data.x}')

    # dataset.num_classes

    model = GATModel(num_features=dataset.num_features, num_classes=NUM_CLASSES,
            num_layers=2, hidden_dim=64, num_heads=8).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()

    
    top_accuracy = 0
    for epoch in range(1, 201):
        loss = train()
        val_acc = test(data.val_mask)
        # test_acc = test(data.test_mask)
        print(f':Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

        if val_acc > top_accuracy:
            top_accuracy = val_acc
            top_model = model

    torch.save(top_model.to('cpu'), 'model'+str(val_acc))




