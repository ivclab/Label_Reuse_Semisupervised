import torch 
from torch.utils.data import Sampler

def compute_num_batches(n, n_partitions, batch_size): 
    p, _ = divmod(n, n_partitions) 
    n_batches_per_partition, _ = divmod(p, batch_size) 
    n_batches = n_partitions * n_batches_per_partition 
    return n_batches 

def partitioned_repeated_randperm(n, n_partitions, n_repeats, batch_size): 
    n_batches = compute_num_batches(n, n_partitions, batch_size) 
    epoch_size = n_batches * batch_size 
    idx_selected = torch.randperm(n)[:epoch_size]  
    for idx_partitioned in torch.chunk(idx_selected, chunks=n_partitions, dim=0): 
        idx_cache = torch.arange(len(idx_partitioned))  
        idx_partitioned_with_idx_cache = torch.stack([idx_partitioned, idx_cache], dim=-1) 
        for i in range(n_repeats): 
            for idx_p, idx_c in idx_partitioned_with_idx_cache[torch.randperm(len(idx_partitioned_with_idx_cache))].tolist(): 
                yield idx_p, idx_c, (i == 0)  

class PartitionedRepeatedShuffledSampler(Sampler): 
    def __init__(self, n, n_partitions, n_repeats, batch_size): 
        self.n = n 
        self.n_partitions = n_partitions 
        self.n_repeats = n_repeats 
        self.batch_size = batch_size 
        self.n_batches = compute_num_batches(n, n_partitions, batch_size) 
        
    def __len__(self): 
        return self.n_batches * self.batch_size 
    
    def __iter__(self): 
        return partitioned_repeated_randperm(
            n=self.n, 
            n_partitions=self.n_partitions, 
            n_repeats=self.n_repeats, 
            batch_size=self.batch_size 
        )