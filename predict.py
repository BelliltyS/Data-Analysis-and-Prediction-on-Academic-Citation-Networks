

import requests
import os
from torch_geometric.data import Dataset
import torch_geometric
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import pandas as pd


class GATModel(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_dim, num_heads):
        super(GATModel, self).__init__()

        self.conv1 = GATConv(dataset.num_features, hidden_dim, heads=num_heads)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
            for _ in range(num_layers - 1)
        ])
        self.lin = nn.Linear(hidden_dim * num_heads, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))

        for conv in self.convs:
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(conv(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)


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


if __name__ == '__main__':
    model = 'model0.5834'
    model = torch.load(model)
    model.eval()
    dataset = HW3Dataset(root='data/hw3/')
    out = model(dataset[0])
    pred = out.argmax(dim=1)


    predictions = pd.DataFrame({'idx': range(len(pred)), 'prediction': pred})
    predictions.to_csv("predictions.csv", index=False)

