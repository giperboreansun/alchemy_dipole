#!/usr/bin/env python
# encoding: utf-8
# File Name: Alchemy_dataset.py
# Author: Jiezhong Qiu (updated for modern compatibility)
# Create Time: 2019/05/08 14:55

import os
import os.path as osp
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import pathlib
import pandas as pd

_urls = {
    'dev': 'https://alchemy.tencent.com/data/dev.zip',
    'valid': 'https://alchemy.tencent.com/data/valid.zip',
    'test': 'https://alchemy.tencent.com/data/test.zip',
}


class TencentAlchemyDataset(InMemoryDataset):
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def __init__(self,
                 root,
                 mode='dev',
                 prop=None,  # Added: property name or index (0-11)
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.mode = mode
        self.prop = prop  # Store property for later use
        assert mode in _urls
        super(TencentAlchemyDataset, self).__init__(root, transform,
                                                    pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        if self.mode == 'dev':
            return [osp.join(self.mode, 'sdf'), osp.join(self.mode, 'dev_target.csv')]
        return [osp.join(self.mode, 'sdf')]

    @property
    def processed_file_names(self):
        return f'TencentAlchemy_{self.mode}.pt'

    def download(self):
        # Auto-download disabled; user must download manually
        pass

    def alchemy_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):  # ✅ Fixed: g.nodes (not g.node)
            h_t = []
            # Atom type (One-hot: H, C, N, O, F, S, Cl)
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']]
            # Atomic number
            h_t.append(d['a_num'])
            # Acceptor / Donor
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            # Aromatic
            h_t.append(int(d['aromatic']))
            # Hybridization
            h_t += [int(d['hybridization'] == x) 
                    for x in (Chem.rdchem.HybridizationType.SP, 
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def alchemy_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, 
                             Chem.rdchem.BondType.DOUBLE, 
                             Chem.rdchem.BondType.TRIPLE, 
                             Chem.rdchem.BondType.AROMATIC)]
            e[(n1, n2)] = e_t

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def sdf_graph_reader(self, sdf_file):
        with open(sdf_file, 'r', encoding='utf-8', errors='ignore') as f:
            sdf_string = f.read()
        mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        if mol is None:
            print(f"[Warning] RDKit cannot parse {sdf_file}")
            return None
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)

        g = nx.DiGraph()

        # Load target values for dev set
        if self.mode == 'dev':
            idx = int(sdf_file.stem)
            if idx in self.target.index:
                l = torch.FloatTensor(self.target.loc[idx].tolist()).unsqueeze(0)
            else:
                return None
        else:
            l = torch.LongTensor([int(sdf_file.stem)])

        # Create nodes
        if len(mol.GetConformers()) == 0:
            return None
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs())

        # Mark donor/acceptor atoms
        for feat in feats:
            if feat.GetFamily() == 'Donor':
                for i in feat.GetAtomIds():
                    g.nodes[i]['donor'] = 1  # ✅ Fixed: g.nodes
            elif feat.GetFamily() == 'Acceptor':
                for i in feat.GetAtomIds():
                    g.nodes[i]['acceptor'] = 1  # ✅ Fixed: g.nodes

        # Read edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType())

        node_attr = self.alchemy_nodes(g)
        edge_index, edge_attr = self.alchemy_edges(g)
        
        data = Data(
            x=node_attr,
            pos=torch.FloatTensor(geom),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=l,
        )
        return data

    def process(self):
        if self.mode == 'dev':
            csv_path = osp.join(self.raw_dir, 'dev_target.csv')
            if not osp.exists(csv_path):
                raise FileNotFoundError(f"Missing {csv_path}. Please download the dataset.")
            self.target = pd.read_csv(csv_path, index_col=0)
            # Keep only property columns
            prop_cols = [f'property_{i}' for i in range(12) if f'property_{i}' in self.target.columns]
            self.target = self.target[prop_cols]

        sdf_dir = pathlib.Path(osp.join(self.raw_dir, 'sdf'))
        if not sdf_dir.exists():
            raise FileNotFoundError(f"Missing SDF directory: {sdf_dir}")

        data_list = []
        for sdf_file in sdf_dir.glob("**/*.sdf"):
            alchemy_data = self.sdf_graph_reader(sdf_file)
            if alchemy_data is not None:
                data_list.append(alchemy_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])