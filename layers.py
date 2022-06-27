class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, feat):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        h = self.dense(feat)
        norm = adj.sum(1)**(-1/2)
        h = norm[None, :] * adj * norm[:, None] @ h
        return h
      
class Graph_Relu_W(nn.Module):
    def __init__(self, nnodes=52):
        super(Graph_Relu_W, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)

class Graph_Directed_A(nn.Module):
      
    def __init__(self, num_nodes=52, window_size=10, alpha=3):
        super(Graph_Directed_A, self).__init__()
        
        self.alpha = alpha
        self.device = device
        
        self.emb1 = nn.Embedding(num_nodes, window_size)
        self.emb2 = nn.Embedding(num_nodes, window_size)
        self.lin1 = nn.Linear(window_size,window_size)
        self.lin2 = nn.Linear(window_size,window_size)
        
    def forward(self, idx):
        
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        
        return adj
