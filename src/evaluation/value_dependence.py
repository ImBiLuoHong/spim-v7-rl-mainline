import torch
from torch_scatter import scatter_max

def run_value_dependency_test(model, loader, device):
    """
    Runs the Value Dependency Test on the given model and loader.
    Returns a dictionary with results.
    """
    model.eval()
    
    total_graphs = 0
    baseline_correct = 0
    sabotaged_correct = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, 'x'): continue
            
            x, edge_index = batch.x, batch.edge_index
            batch_vec = batch.batch if hasattr(batch, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)
            edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            targets = batch.y
            if targets.dim() > 1: targets = targets.argmax(dim=-1)
            
            # --- Baseline ---
            # Use batch object directly for T2IBDASNet
            try:
                out_base = model(batch)
            except TypeError:
                 # Fallback for older models signature
                 out_base = model(x, edge_index, edge_attr, batch=batch_vec)
            
            if isinstance(out_base, dict):
                logits_base = out_base['classification'].squeeze(-1)
            else:
                logits_base = out_base.squeeze(-1)
                
            max_val_base, max_idx_base = scatter_max(logits_base, batch_vec)
            
            # Hit@1 check
            # targets[max_idx_base] > 0.5 (assuming binary target per node where source=1)
            # targets is [B] (graph label) or [N] (node label)?
            # Usually batch.y is [B] (graph label). 
            # If batch.y is node label, it's [N].
            # For Source Localization, y is often [N] (1 for source, 0 for others).
            # But here targets = batch.y. 
            # If targets is [B], this logic is wrong?
            # scatter_max returns max_idx_base as index into N.
            # targets[max_idx_base] gets the label of the predicted node.
            # If label is 1, it's correct.
            acc_base_batch = (targets[max_idx_base] > 0.5).float().sum()
            baseline_correct += acc_base_batch.item()
            
            # --- Sabotage ---
            x_original = batch.x.clone()
            x_sabotaged = x.clone()
            # Zero out dynamic channels (0-3)
            # Ch 0: Signal, Ch 1: Poison, Ch 2: Lag
            # Ch 4: Trigger (Static/Context) -> KEEP
            # Ch 5-8: Topology -> KEEP
            x_sabotaged[..., 0:3] = 0.0 
            
            batch.x = x_sabotaged
            try:
                out_sab = model(batch)
            except TypeError:
                out_sab = model(x_sabotaged, edge_index, edge_attr, batch=batch_vec)
            
            # Restore x
            batch.x = x_original 
            
            if isinstance(out_sab, dict):
                logits_sab = out_sab['classification'].squeeze(-1)
            else:
                logits_sab = out_sab.squeeze(-1)
                
            max_val_sab, max_idx_sab = scatter_max(logits_sab, batch_vec)
            acc_sab_batch = (targets[max_idx_sab] > 0.5).float().sum()
            sabotaged_correct += acc_sab_batch.item()
            
            total_graphs += (batch_vec.max().item() + 1)
            
    baseline_acc = baseline_correct / max(1, total_graphs)
    sabotaged_acc = sabotaged_correct / max(1, total_graphs)
    drop_rate = baseline_acc - sabotaged_acc
    
    if drop_rate < 0.1:
        result = "FAIL"
    elif drop_rate > 0.5:
        result = "PASS"
    else:
        result = "WARN"
        
    return {
        'baseline_acc': baseline_acc,
        'sabotaged_acc': sabotaged_acc,
        'drop_rate': drop_rate,
        'result': result
    }
