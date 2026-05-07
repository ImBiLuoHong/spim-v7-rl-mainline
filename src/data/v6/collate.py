import torch
from torch_geometric.data import Batch, Data

def v6_collate_fn(data_list):
    """
    Collates a list of V6 Data objects into a PyG Batch.
    Handles 'x' with shape (Total_N, 9) [Static V3.2 Spec].
    
    Output:
    - batch.x: (Total_N, 9)
    - batch.edge_index: (2, Total_E) [Spatially disjoint]
    - batch.edge_attr: (Total_E, 1)
    - batch.batch: (Total_N,)
    - batch.y: (B,)
    """
    # Filter Nones
    data_list = [d for d in data_list if d is not None]
    if len(data_list) == 0:
        return None
        
    # Use PyG Batch
    batch = Batch.from_data_list(data_list)
    
    return batch
