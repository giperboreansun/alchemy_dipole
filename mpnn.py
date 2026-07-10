#!/usr/bin/env python
# encoding: utf-8
# File Name: mpnn.py
# Author: Jiezhong Qiu (updated for modern compatibility)
# Create Time: 2019/04/23 17:38

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from Alchemy_dataset import TencentAlchemyDataset
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import time
import logging
import pandas as pd


# ✅ Optional: lightweight transform (disable Complete() to save memory)
class DistanceOnly(object):
    """Add scalar distances as edge_attr (compatible with T.Distance)"""
    def __call__(self, data):
        if data.pos is not None and data.edge_index is not None:
            row, col = data.edge_index
            dist = torch.norm(data.pos[row] - data.pos[col], p=2, dim=1, keepdim=True)
            if data.edge_attr is not None:
                data.edge_attr = torch.cat([data.edge_attr, dist], dim=-1)
            else:
                data.edge_attr = dist
        return data


class MPNN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim=1,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6):
        super(MPNN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), 
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
        )
        self.conv = NNConv(node_hidden_dim,
                           node_hidden_dim,
                           edge_network,
                           aggr='mean',  # Can be changed to 'add' for MPNN*
                           root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim,
                               processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for _ in range(self.num_step_message_passing):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


def run(prop="dipole", gpuid="0", epoch=500, dataset_name="alchemy", size=100000):
    # ✅ Property index mapping (Alchemy has 12 properties)
    PROP_INDEX = {
        'dipole': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4,
        'r2': 5, 'zpve': 6, 'U0': 7, 'U': 8, 'H': 9, 'G': 10, 'Cv': 11,
        # Also accept integer indices
        **{str(i): i for i in range(12)}
    }
    prop_idx = PROP_INDEX.get(prop.lower(), 0)


    # ✅ Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    # Set logger
    task_name = f"MPNN_{dataset_name}_{prop}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logname = f"./logs/{task_name}.log"
    
    log = logging.getLogger(task_name)
    log.setLevel(logging.INFO)
    log.handlers = []  # Clear existing handlers

    writer = SummaryWriter(log_dir=f"./logs/tensorboard/{task_name}")
    
    fmt = "%(asctime)s %(levelname)s %(filename)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    fh = logging.FileHandler(logname, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    
    log.addHandler(fh)
    log.addHandler(ch)
    log.info(f"Experiment: {task_name}, dataset size: {size}, property: {prop} (index {prop_idx})")

    # ✅ Device setup with CPU fallback
    if gpuid.isdigit() and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpuid}")
        log.info(f"Using GPU {gpuid}: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")

    # ✅ Transform: use lightweight version to avoid OOM
    # Comment out Complete() to keep only chemical bonds
    transform = T.Compose([DistanceOnly()])  # ✅ Memory-safe
    # transform = T.Compose([Complete(), T.Distance(norm=False)])  # ❌ Full graph (high memory)

    # ✅ Load dataset (removed invalid 'dataset' and 'prop' args)
    dataset = TencentAlchemyDataset(
        root='./tdata/',
        mode='dev',
        transform=transform
    ).shuffle()
    
    dataset = dataset[:size]
    trainset = dataset[:size - 20000] if size > 20000 else dataset[:int(0.8*size)]
    valset = dataset[size - 20000:size - 10000] if size > 20000 else dataset[int(0.8*size):int(0.9*size)]
    testset = dataset[size - 10000:] if size > 10000 else dataset[int(0.9*size):]
    
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valset, batch_size=64)
    test_loader = DataLoader(testset, batch_size=64)
    
    log.info(f"Train/Val/Test: {len(trainset)}/{len(valset)}/{len(testset)} molecules")

    # ✅ Model: output_dim=1 for single-property regression
    model = MPNN(node_input_dim=trainset.num_features, output_dim=1).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    state_dict = torch.load(
        "./logs/MPNN_alchemy_dipole_0709_155429_best.pt",
        weights_only=False
    )

    model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    st = time.time()
    best_valid = float("inf")
    best_test = None

    for it in range(epoch):
        model.train()
        loss_all = 0
        mae_train = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # ✅ Select target property: data.y is [B, 12], we need [B, 1]
            y_true = data.y[:, prop_idx:prop_idx+1]
            
            y_pred = model(data)
            loss = F.mse_loss(y_pred, y_true)
            mae_train += F.l1_loss(y_pred, y_true).item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ✅ Gradient clipping
            optimizer.step()
            
            loss_all += loss.item() * data.num_graphs
        
        avg_train_loss = loss_all / len(trainset)
        avg_train_mae = mae_train / len(train_loader)

        # Validation
        model.eval()
        mae_val = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                y_true = data.y[:, prop_idx:prop_idx+1]
                y_pred = model(data)
                mae_val += F.l1_loss(y_pred, y_true).item()
        avg_val_mae = mae_val / len(val_loader)

        # Test
        mae_test = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                y_true = data.y[:, prop_idx:prop_idx+1]
                y_pred = model(data)
                mae_test += F.l1_loss(y_pred, y_true).item()
        avg_test_mae = mae_test / len(test_loader)

        scheduler.step(avg_val_mae)
        
        log.info(f"Epoch {it+1:3d} | Train Loss: {avg_train_loss:.6f} | "
                 f"Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f} | Test MAE: {avg_test_mae:.4f}")

        if avg_val_mae < best_valid:
            best_valid = avg_val_mae
            best_test = avg_test_mae
            # ✅ Save best model
            torch.save(model.state_dict(), f"./logs/{task_name}_best.pt")
            log.info(f"  → New best! Saved to {task_name}_best.pt")

        writer.add_scalar('Loss/train', avg_train_loss, it)
        writer.add_scalar('MAE/train', avg_train_mae, it)
        writer.add_scalar('MAE/val', avg_val_mae, it)
        writer.add_scalar('MAE/test', avg_test_mae, it)
        #writer.add_scalar('LR', current_lr, it)

        if it % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'weights/{name}', param, it)
                if param.grad is not None:
                    writer.add_histogram(f'grads/{name}', param.grad, it)

    ed = time.time()
    log.info(f"\n✅ Finished | Best Val MAE: {best_valid:.4f} | Related Test MAE: {best_test:.4f} | Time: {ed-st:.0f}s")
    writer.close()
    return best_valid, best_test


if __name__ == "__main__":
    # Usage: python mpnn.py <gpu_id> <size> <dataset_name> <property>
    # Example: python mpnn.py 0 10000 alchemy dipole
    if len(sys.argv) < 5:
        print("Usage: python mpnn.py <gpu_id> <size> <dataset_name> <property>")
        print("  gpu_id: '0' for GPU, 'cpu' for CPU")
        print("  size: number of molecules (e.g., 10000); 0 for full dataset")
        print("  dataset_name: 'alchemy' (default)")
        print("  property: dipole, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv, or 0-11")
        sys.exit(1)
    
    gpuid = sys.argv[1]
    size = int(sys.argv[2]) if sys.argv[2] != '0' else 119487  # Full Alchemy dev set
    dataset_name = sys.argv[3]
    prop = sys.argv[4]
    
    run(prop=prop, gpuid=gpuid, epoch=500, dataset_name=dataset_name, size=size)
