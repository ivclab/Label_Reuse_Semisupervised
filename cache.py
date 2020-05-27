import torch 
from torch import nn 

class Cache(nn.Module): 
    def __init__(self, n_entries, entry_size): 
        super(Cache, self).__init__() 
        
        self.n_entries = n_entries 
        self.entry_size = entry_size 
        self.register_buffer(
            name='idx_sparse', 
            tensor=torch.zeros((n_entries, entry_size), dtype=torch.long)
        )
        self.register_buffer(
            name='value_sparse', 
            tensor=torch.zeros((n_entries, entry_size))
        )
    
    def forward(self): 
        return 
        
    def read(self, idx): 
        return self.idx_sparse[idx], self.value_sparse[idx] 
    
    def write(self, idx, idx_sparse, value_sparse): 
        self.idx_sparse[idx] = idx_sparse 
        self.value_sparse[idx] = value_sparse 
        return 