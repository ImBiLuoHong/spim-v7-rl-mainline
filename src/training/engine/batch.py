import torch


def batch_to_device(batch, device: torch.device):
    """Move a DataLoader batch to target device; return a 6-tuple.
    Supports:
    - (features, soft_targets, edge_index, edge_attr)
    - (features, soft_targets, edge_index, edge_attr, eidx_v, eattr_v)
    - dict style with same keys
    This module does not alter batch dict content for causal/TV keys.
    """
    def _to_dev(x):
        return None if x is None else x.to(device, non_blocking=True)

    if isinstance(batch, (list, tuple)):
        if len(batch) == 6:
            features, soft_targets, edge_index, edge_attr, eidx_v, eattr_v = batch
        elif len(batch) == 4:
            features, soft_targets, edge_index, edge_attr = batch
            eidx_v = None
            eattr_v = None
        else:
            raise ValueError(f"Unsupported batch tuple length: {len(batch)}")
    elif isinstance(batch, dict):
        features = batch.get('features')
        soft_targets = batch.get('soft_targets')
        edge_index = batch.get('edge_index')
        edge_attr = batch.get('edge_attr')
        eidx_v = batch.get('eidx_v')
        eattr_v = batch.get('eattr_v')
    elif hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
        # PyG Batch Object support
        features = batch.x
        soft_targets = batch.y
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        eidx_v = None
        eattr_v = None
    else:
        raise ValueError("Unsupported batch type; expected tuple/list or dict")

    return (
        _to_dev(features),
        _to_dev(soft_targets),
        _to_dev(edge_index),
        _to_dev(edge_attr),
        _to_dev(eidx_v),
        _to_dev(eattr_v),
    )


class DevicePrefetcher:
    """
    Prefetches batches to device asynchronously using CUDA streams.
    Optimized for Phase 6 Directive: Operation "CPU Breakout".
    Uses non_blocking transfer and records stream events.
    Now supports multi-epoch iteration by resetting iterator in __iter__.
    """
    def __init__(self, loader, device):
        self.orig_loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self.loader_iter = None
        
    def preload(self):
        try:
            self.next_data = next(self.loader_iter)
        except StopIteration:
            self.next_data = None
            return
            
        with torch.cuda.stream(self.stream):
            # Non-blocking transfer is the key
            # Support both PyG Batch objects and standard Tensors
            if hasattr(self.next_data, 'to'):
                self.next_data = self.next_data.to(self.device, non_blocking=True)
            elif isinstance(self.next_data, (list, tuple)):
                self.next_data = [x.to(self.device, non_blocking=True) if hasattr(x, 'to') else x for x in self.next_data]
            elif isinstance(self.next_data, dict):
                self.next_data = {k: v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in self.next_data.items()}

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        
        if data is not None:
            # Record stream for safety
            if hasattr(data, 'record_stream'):
                data.record_stream(torch.cuda.current_stream())
            elif isinstance(data, (list, tuple)):
                for x in data:
                    if hasattr(x, 'record_stream'):
                        x.record_stream(torch.cuda.current_stream())
            elif isinstance(data, dict):
                for v in data.values():
                    if hasattr(v, 'record_stream'):
                        v.record_stream(torch.cuda.current_stream())
        
        self.preload()
        return data

    def __iter__(self):
        self.loader_iter = iter(self.orig_loader)
        self.preload()
        return self

    def __next__(self):
        data = self.next()
        if data is None:
            raise StopIteration
        return data

    def __len__(self):
        return len(self.orig_loader)

    def __getattr__(self, name):
        # Forward other attributes to the loader (e.g. dataset)
        return getattr(self.orig_loader, name)
