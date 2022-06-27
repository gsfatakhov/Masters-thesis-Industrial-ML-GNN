import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class GNNTEP(nn.Module):    
    def __init__(self, nnodes=52, window_size=100, ngnn=1, gsllayer='directed', nhidden=256,
                 alpha=0.1, k, out_channels = 29, device='cpu'):
        super(GNNTEP, self).__init__()
        self.window_size = window_size
        self.nhidden = nhidden
        self.nnodes = nnodes
        self.device = device
        self.idx = torch.arange(self.num_nodes).to(device)
        self.adj = [0 for i in range(ngnn)]
        self.h = [0 for i in range(ngnn)]
        self.skip = [0 for i in range(ngnn)]
        self.z = (torch.ones(nnodes, nnodes) - torch.eye(nnodes)).to(device)
        self.ngnn = ngnn
        
        self.graph_struct = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()
        
        for i in range(self.num_layers):
            if gsllayer == 'relu':
                self.graph_struct.append(Graph_ReLu_W(nnodes, device))
            elif gsllayer == 'directed':
                self.graph_struct.append(Graph_Directed_A(nnodes, window_size, alpha, k, device))
            elif gsllayer == 'unidirected':
                self.graph_struct.append(Graph_Uni_Directed_A(nnodes, window_size, alpha, k, device))
            elif gsllayer == 'undirected':
                self.graph_struct.append(Graph_Undirected_A(nnodes, window_size, alpha, k, device))
            else:
                print('Wrong name of graph structure learning layer!')
            self.conv1.append(GCNLayer(window_size, nhidden))
            self.bnorm1.append(nn.BatchNorm1d(nnodes))
            self.conv2.append(GCNLayer(nhidden, nhidden))
            self.bnorm2.append(nn.BatchNorm1d(nnodes))
        
        self.fc = nn.Linear(ngnn*nhidden, out_channels)
    
    
    def forward(self, X):
        
        X = X.to(device)
        
        for i in range(self.ngnn):
            self.adj[i] = self.graph_struct[i](self.idx)
            self.adj[i] = self.adj[i] * self.z
            self.h[i] = self.conv1[i](self.adj[i], X).relu()
            self.h[i] = self.bnorm1[i](self.h[i])
            self.skip[i], _ = torch.min(self.h[i],dim=1)
            self.h[i] = self.conv2[i](self.adj[i], self.h[i]).relu()
            self.h[i] = self.bnorm2[i](self.h[i])
            self.h[i], _ = torch.min(self.h[i],dim=1)
            self.h[i] = self.h[i] + self.skip[i]
                
        h = torch.cat(self.h, 1)
        output = self.fc(h)
        
        return output
    
    def get_adj(self):
        return self.adj
    

class CNN1DTEP(nn.Module):    
    def __init__(self, batch_size=512, window_size=100, features_size=52):        
        super(CNN1DTEP, self).__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.features_size = features_size
        
        self.conv1 = nn.Conv1d(features_size, features_size * 10, window_size, groups=52)
        self.fc1 = nn.Linear(features_size * 10, 128)
        self.fc2 = nn.Linear(128, 29)
        
    def forward(self, X):
        X = X.to(device)
        self.bach_size = X.shape[0]
        X = F.relu(self.conv1(X))
        X = X.reshape(self.bach_size, self.features_size * 10)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        
        return X  
