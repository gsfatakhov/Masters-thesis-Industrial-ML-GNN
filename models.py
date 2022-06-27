from layers import *

class GNNTEP(nn.Module):    
    def __init__(self, num_nodes=52, window_size=100, num_layers=1, gsllayer='directed', hid_dim=256,
                 tanhalpha=3, out_channels = 29, device=None):
        super(GNNTEP, self).__init__()
        self.window_size = window_size
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.device = device
        self.idx = torch.arange(self.num_nodes).to(device)
        self.adj = [0 for i in range(num_layers)]
        self.h = [0 for i in range(num_layers)]
        self.skip = [0 for i in range(num_layers)]
        self.z = (torch.ones(52, 52) - torch.eye(52)).to(device)
        self.num_layers = num_layers
        
        self.gc = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.gc.append(Graph_Unidirected_A(num_nodes, 100, alpha=0.1))
            self.conv1.append(GCNLayer(window_size, hid_dim))
            self.bnorm1.append(nn.BatchNorm1d(num_nodes))
            self.conv2.append(GCNLayer(hid_dim, hid_dim))
            self.bnorm2.append(nn.BatchNorm1d(num_nodes))
        
        self.fc = nn.Linear(num_layers*hid_dim,29)
    
    
    def forward(self, X):
        
        X = X.to(device)
        
        for i in range(self.num_layers):
            self.adj[i] = self.gc[i](self.idx)
            self.adj[i] = self.adj[i] * self.z
            self.h[i] = self.conv1[i](self.adj[i], X).relu()
            self.h[i] = self.bnorm1[i](self.h[i])
            self.skip[i], _ = torch.min(self.h[i],dim=1)
            self.h[i] = self.conv2[i](self.adj[i], self.h[i]).relu()
            self.h[i] = self.bnorm2[i](self.h[i])
            self.h[i], _ = torch.min(self.h[i],dim=1)
            self.h[i] = self.h[i] + self.skip[i]
                
        h = torch.cat(self.h, 1)
        out = self.fc(h)
        
        return out
    
    def get_adj(self):
        return self.adjA, self.adjB, self.adjC, self.adjD
