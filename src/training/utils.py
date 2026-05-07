"""
Facade of training utilities after refactor.
This module re-exports SRP functions from utils.training_engine to preserve backward compatibility.
Keep API surface identical to the pre-refactor version so existing imports/tests continue working.
"""

from src.training.engine.device import select_device
from src.training.engine.batch import batch_to_device as _batch_to_device
from src.training.engine.metrics_basic import (
    compute_metrics as _compute_metrics,
    compute_rank_metrics as _compute_rank_metrics,
    hard_labels_from,
)
from src.training.engine.confusion import (
    confusion_counts_from_logits_targets as _confusion_counts_from_logits_targets,
    balanced_acc_and_macro_f1_from_counts as _balanced_acc_and_macro_f1_from_counts,
)
from src.training.engine.data_inspect import (
    infer_edge_dims,
    infer_num_features_from_batch,
    print_travel_time_samples,
)
from src.training.engine.topk_cov import (
    compute_topk_label_mass_and_cov_tilde,
)
from src.training.engine.scheduler import (
    build_scheduler,
)
