#!/usr/bin/env python
# encoding: utf-8
# File Name: mpnn.py
# Author: Jiezhong Qiu (updated for modern compatibility + multi-task + dipole logging)
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


# ✅ Names of the 12 properties (for logging)
PROP_NAMES = ['dipole', 'alpha', 'homo', 'lumo', 'gap', 'r2', 
              'zpve', 'U0', 'U', 'H', 'G', 'Cv']

# ✅ Dipole moment is always at index 0
DIPOLE_IDX = 0


class DistanceOnly(object):
    """Add scalar distances as edge_attr (memory-safe transform)"""
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
                 output_dim=12,           # ✅ Multi-task: 12 properties
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 dropout=0.15):
        super(MPNN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.dropout = nn.Dropout(dropout)
        
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), 
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
        )
        self.conv = NNConv(node_hidden_dim,
                           node_hidden_dim,
                           edge_network,
                           aggr='mean',
                           root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim,
                               processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        out = self.dropout(out)
        h = out.unsqueeze(0)

        for _ in range(self.num_step_message_passing):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            m = self.dropout(m)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.dropout(out)
        out = self.lin2(out)
        return out


def run(prop="all", gpuid="0", epoch=500, dataset_name="alchemy", size=100000):
    # ✅ Multi-task mode: predict all 12 properties
    is_multi_task = (prop.lower() == "all")
    
    # For single-task mode, map property name to index
    PROP_INDEX = {
        'dipole': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4,
        'r2': 5, 'zpve': 6, 'U0': 7, 'U': 8, 'H': 9, 'G': 10, 'Cv': 11,
        **{str(i): i for i in range(12)}
    }
    prop_idx = PROP_INDEX.get(prop.lower(), 0) if not is_multi_task else None

    # ✅ Create logs_broad directory
    log_dir = "./logs_broad"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)
    
    # Set logger
    task_name = f"MPNN_{dataset_name}_{prop}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logname = f"{log_dir}/{task_name}.log"
    
    log = logging.getLogger(task_name)
    log.setLevel(logging.INFO)
    log.handlers = []

    writer = SummaryWriter(log_dir=f"{log_dir}/tensorboard/{task_name}")
    
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
    
    mode_str = "MULTI-TASK (all 12 properties)" if is_multi_task else f"single-task ({prop}, index {prop_idx})"
    log.info(f"Experiment: {task_name}")
    log.info(f"Dataset size: {size}, Mode: {mode_str}")

    # ✅ Device setup
    if gpuid.isdigit() and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpuid}")
        log.info(f"Using GPU {gpuid}: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")

    # ✅ Transform
    transform = T.Compose([DistanceOnly()])

    # ✅ Load dataset
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

    # ✅ Model: multi-task with wider layers
    output_dim = 12 if is_multi_task else 1
    model = MPNN(
        node_input_dim=trainset.num_features,
        edge_input_dim=5,
        output_dim=output_dim,
        node_hidden_dim=128,
        edge_hidden_dim=256,
        dropout=0.15
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {num_params:,}")
    log.info(f"Architecture: node_hidden=128, edge_hidden=256, dropout=0.15, output_dim={output_dim}")

    # ✅ Weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    st = time.time()
    best_valid = float("inf")
    best_test = None
    best_test_per_prop = None
    
    # ✅ NEW: Track best dipole MAE separately
    best_dipole_val = float("inf")
    best_dipole_test = None
    
    no_improve_count = 0
    patience = 50

    for it in range(epoch):
        epoch_start = time.time()
        
        # === TRAIN ===
        model.train()
        loss_all = 0
        mae_train_per_prop = [0.0] * 12
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            y_pred = model(data)  # [B, 12] or [B, 1]
            
            if is_multi_task:
                y_true = data.y  # [B, 12]
                loss = F.mse_loss(y_pred, y_true)
                for i in range(12):
                    mae_train_per_prop[i] += F.l1_loss(y_pred[:, i], y_true[:, i]).item()
            else:
                y_true = data.y[:, prop_idx:prop_idx+1]
                loss = F.mse_loss(y_pred, y_true)
                mae_train_per_prop[prop_idx] += F.l1_loss(y_pred, y_true).item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_all += loss.item() * data.num_graphs
        
        avg_train_loss = loss_all / len(trainset)
        avg_train_mae_per_prop = [m / len(train_loader) for m in mae_train_per_prop]

        # === VALIDATION ===
        model.eval()
        mae_val_per_prop = [0.0] * 12
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                y_pred = model(data)
                
                if is_multi_task:
                    y_true = data.y
                    for i in range(12):
                        mae_val_per_prop[i] += F.l1_loss(y_pred[:, i], y_true[:, i]).item()
                else:
                    y_true = data.y[:, prop_idx:prop_idx+1]
                    mae_val_per_prop[prop_idx] += F.l1_loss(y_pred, y_true).item()
        
        avg_val_mae_per_prop = [m / len(val_loader) for m in mae_val_per_prop]

        # === TEST ===
        mae_test_per_prop = [0.0] * 12
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                y_pred = model(data)
                
                if is_multi_task:
                    y_true = data.y
                    for i in range(12):
                        mae_test_per_prop[i] += F.l1_loss(y_pred[:, i], y_true[:, i]).item()
                else:
                    y_true = data.y[:, prop_idx:prop_idx+1]
                    mae_test_per_prop[prop_idx] += F.l1_loss(y_pred, y_true).item()
        
        avg_test_mae_per_prop = [m / len(test_loader) for m in mae_test_per_prop]

        # === METRICS ===
        if is_multi_task:
            mean_train_mae = sum(avg_train_mae_per_prop) / 12
            mean_val_mae = sum(avg_val_mae_per_prop) / 12
            mean_test_mae = sum(avg_test_mae_per_prop) / 12
            scheduler.step(mean_val_mae)
        else:
            mean_train_mae = avg_train_mae_per_prop[prop_idx]
            mean_val_mae = avg_val_mae_per_prop[prop_idx]
            mean_test_mae = avg_test_mae_per_prop[prop_idx]
            scheduler.step(mean_val_mae)

        # ✅ NEW: Extract dipole MAE specifically
        dipole_train_mae = avg_train_mae_per_prop[DIPOLE_IDX]
        dipole_val_mae = avg_val_mae_per_prop[DIPOLE_IDX]
        dipole_test_mae = avg_test_mae_per_prop[DIPOLE_IDX]

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # === LOGGING (ENHANCED WITH DIPOLE) ===
        log.info(f"Epoch {it+1:3d} | Loss: {avg_train_loss:.6f} | "
                 f"Mean MAE: {mean_train_mae:.4f}/{mean_val_mae:.4f}/{mean_test_mae:.4f} | "
                 f"Dipole MAE: {dipole_train_mae:.4f}/{dipole_val_mae:.4f}/{dipole_test_mae:.4f} | "
                 f"LR: {current_lr:.2e} | {epoch_time:.1f}s")
        
        # Detailed per-property log every 10 epochs
        if (it + 1) % 10 == 0:
            log.info("  Property-wise MAE (train / val / test):")
            for i, name in enumerate(PROP_NAMES):
                if is_multi_task or i == prop_idx:
                    marker = " ←" if i == DIPOLE_IDX else ""
                    log.info(f"    {name:6s}: {avg_train_mae_per_prop[i]:.4f} / "
                             f"{avg_val_mae_per_prop[i]:.4f} / "
                             f"{avg_test_mae_per_prop[i]:.4f}{marker}")

        # === TENSORBOARD ===
        writer.add_scalar('Loss/train', avg_train_loss, it)
        writer.add_scalar('MAE/mean_train', mean_train_mae, it)
        writer.add_scalar('MAE/mean_val', mean_val_mae, it)
        writer.add_scalar('MAE/mean_test', mean_test_mae, it)
        writer.add_scalar('LR', current_lr, it)
        
        # ✅ NEW: Dipole-specific scalars (every epoch)
        writer.add_scalar('Dipole_MAE/train', dipole_train_mae, it)
        writer.add_scalar('Dipole_MAE/val', dipole_val_mae, it)
        writer.add_scalar('Dipole_MAE/test', dipole_test_mae, it)
        
        if is_multi_task:
            for i, name in enumerate(PROP_NAMES):
                writer.add_scalar(f'MAE_val/{name}', avg_val_mae_per_prop[i], it)
        
        if it % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'weights/{name}', param, it)
                if param.grad is not None:
                    writer.add_histogram(f'grads/{name}', param.grad, it)

        # === EARLY STOPPING (based on mean Val MAE) ===
        if mean_val_mae < best_valid:
            best_valid = mean_val_mae
            best_test = mean_test_mae
            best_test_per_prop = avg_test_mae_per_prop.copy()
            no_improve_count = 0
            torch.save(model.state_dict(), f"{log_dir}/{task_name}_best.pt")
            log.info(f"  → New best! Mean Val MAE: {mean_val_mae:.4f} | Dipole Val MAE: {dipole_val_mae:.4f} | Saved")
        else:
            no_improve_count += 1
        
        # ✅ NEW: Track best dipole Val MAE separately
        if dipole_val_mae < best_dipole_val:
            best_dipole_val = dipole_val_mae
            best_dipole_test = dipole_test_mae
            # Save separate model for best dipole performance
            torch.save(model.state_dict(), f"{log_dir}/{task_name}_best_dipole.pt")
        
        if no_improve_count >= patience:
            log.info(f"Early stopping at epoch {it+1} (no improvement for {patience} epochs)")
            break

    # === FINAL SUMMARY ===
    ed = time.time()
    log.info(f"\n{'='*70}")
    log.info(f"✅ FINISHED | Time: {ed-st:.0f}s")
    log.info(f"{'='*70}")
    log.info(f"Best Mean Val MAE:     {best_valid:.4f}  →  Test: {best_test:.4f}")
    log.info(f"Best Dipole Val MAE:   {best_dipole_val:.4f}  →  Test: {best_dipole_test:.4f}  ← (Debye)")
    log.info(f"{'='*70}")
    
    if is_multi_task and best_test_per_prop is not None:
        log.info("Final Test MAE per property (at best mean Val MAE epoch):")
        for i, name in enumerate(PROP_NAMES):
            marker = " ← DIPOLE" if i == DIPOLE_IDX else ""
            log.info(f"  {name:6s}: {best_test_per_prop[i]:.4f}{marker}")
    
    log.info(f"\nBest overall model saved to:  {log_dir}/{task_name}_best.pt")
    log.info(f"Best dipole model saved to:   {log_dir}/{task_name}_best_dipole.pt")
    
    writer.close()
    return best_valid, best_test


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python mpnn.py <gpu_id> <size> <dataset_name> <property>")
        print("  gpu_id: '0' for GPU, 'cpu' for CPU")
        print("  size: number of molecules (e.g., 10000); 0 for full dataset")
        print("  dataset_name: 'alchemy' (default)")
        print("  property: 'all' for multi-task, or dipole/alpha/homo/lumo/gap/r2/zpve/U0/U/H/G/Cv, or 0-11")
        sys.exit(1)
    
    gpuid = sys.argv[1]
    size = int(sys.argv[2]) if sys.argv[2] != '0' else 119487
    dataset_name = sys.argv[3]
    prop = sys.argv[4]
    
    run(prop=prop, gpuid=gpuid, epoch=500, dataset_name=dataset_name, size=size)