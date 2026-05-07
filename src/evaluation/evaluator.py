import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from src.training.utils import _compute_metrics, _compute_rank_metrics, _confusion_counts_from_logits_targets, _balanced_acc_and_macro_f1_from_counts

class Evaluator:
    """
    Slot 13: Evaluator (Standardized Entry Point)
    
    Responsibilities:
    1. Centralize metric calculation logic (Batch & Episode).
    2. Ensure consistency between Train and Eval metrics.
    3. Provide unified summarization.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.rank_ks = getattr(cfg.training, 'rank_ks', [1, 3, 5])
        self.ndcg_k = getattr(cfg.training, 'ndcg_k', 5)
        self.reset()

    def reset(self):
        """Reset internal state for a new epoch/evaluation."""
        # Batch Metrics Accumulators
        self.batch_counts = 0
        self.total_loss = 0.0
        self.total_samples = 0
        
        self.hit_sums = defaultdict(float)
        self.mrr_sum = 0.0
        self.ndcg_sum = 0.0
        
        # Confusion Matrix Accumulators
        self.tp_sum = None
        self.fp_sum = None
        self.fn_sum = None
        self.p_sum = None
        
        # Binary Metrics Sums (Acc, Prec, Rec, F1)
        self.binary_sums = defaultdict(float)

        # Episode Metrics Accumulators (Phase 4.5)
        self.episodes = []
        
        # Auxiliary Sums
        self.aux_sums = defaultdict(float)

    def update_batch(self, logits, targets, loss=None):
        """
        Update metrics with a batch of dense results.
        Args:
            logits: [B, N]
            targets: [B, N]
            loss: float (optional)
        """
        bsz = logits.size(0)
        self.batch_counts += 1
        self.total_samples += bsz
        
        if loss is not None:
            self.total_loss += float(loss)

        # 1. Binary / Node-Level
        acc, prec, rec, f1 = _compute_metrics(logits.detach(), targets.detach())
        self.binary_sums['acc'] += acc
        self.binary_sums['prec'] += prec
        self.binary_sums['rec'] += rec
        self.binary_sums['f1'] += f1
        
        # 2. Rank Metrics
        hits, mrr, ndcg = _compute_rank_metrics(logits.detach(), targets.detach(), ks=self.rank_ks, ndcg_k=self.ndcg_k)
        
        for k, v in hits.items():
            self.hit_sums[k] += v * bsz
        self.mrr_sum += mrr * bsz
        self.ndcg_sum += ndcg * bsz
        
        # 3. Confusion Matrix (for Balanced Acc / Macro F1)
        try:
            tp, fp, fn, p = _confusion_counts_from_logits_targets(logits.detach(), targets)
            if self.tp_sum is None:
                self.tp_sum, self.fp_sum, self.fn_sum, self.p_sum = tp.clone(), fp.clone(), fn.clone(), p.clone()
            else:
                self.tp_sum += tp
                self.fp_sum += fp
                self.fn_sum += fn
                self.p_sum += p
        except Exception:
            pass

    def update_aux(self, name, value, weight=1.0):
        """Update auxiliary metrics (loss components, etc.)"""
        self.aux_sums[name] += float(value) * weight

    def update_episode(self, episode_data):
        """
        Ingest a single episode's result (Phase 4.5).
        Args:
            episode_data (dict):
                - 'success': bool
                - 'steps': int
                - 'budget': float
                - 'predict_hit': bool
                - ...
        """
        self.episodes.append(episode_data)

    def summarize(self):
        """
        Return unified dictionary of metrics.
        """
        metrics = {}
        
        # --- Batch Metrics ---
        if self.total_samples > 0:
            denom = max(1, self.batch_counts)
            samp_denom = max(1, self.total_samples)
            
            metrics['loss'] = self.total_loss / denom
            
            # Binary
            for k, v in self.binary_sums.items():
                metrics[k] = v / denom
                
            # Rank
            for k, v in self.hit_sums.items():
                metrics[f'hit@{k}'] = v / samp_denom
            metrics['mrr'] = self.mrr_sum / samp_denom
            metrics['ndcg'] = self.ndcg_sum / samp_denom
            
            # Balanced Acc & Macro F1
            try:
                if self.tp_sum is not None:
                    bal_acc, macro_f1 = _balanced_acc_and_macro_f1_from_counts(
                        self.tp_sum, self.fp_sum, self.fn_sum, self.p_sum
                    )
                    metrics['bal_acc'] = bal_acc
                    metrics['macro_f1'] = macro_f1
            except Exception:
                pass
            
            # Aux
            for k, v in self.aux_sums.items():
                metrics[k] = v / denom

        # --- Episode Metrics (Phase 4.5) ---
        if self.episodes:
            n = len(self.episodes)
            def ep_mean(key, default=0.0):
                values = [float(e.get(key, default)) for e in self.episodes]
                return float(sum(values) / max(len(values), 1))

            metrics['official/core_mass_before'] = ep_mean('core_mass_before')
            metrics['official/core_mass_after'] = ep_mean('core_mass_after')
            metrics['official/core_mass_delta'] = ep_mean('core_mass_delta')
            metrics['official/core_size_before'] = ep_mean('core_size_before')
            metrics['official/core_size_after'] = ep_mean('core_size_after')
            metrics['official/core_size_delta'] = ep_mean('core_size_delta')
            metrics['official/uncertainty_before'] = ep_mean('uncertainty_before')
            metrics['official/uncertainty_after'] = ep_mean('uncertainty_after')
            metrics['official/uncertainty_collapse'] = ep_mean('uncertainty_collapse')
            metrics['official/closure_rate'] = ep_mean('closure_success')
            metrics['official/decisive_closure_rate'] = ep_mean('decisive_closure')
            metrics['official/budget_used'] = ep_mean('budget')
            metrics['official/budget_to_closure'] = ep_mean('budget_to_closure')
            metrics['official/budget_efficiency'] = ep_mean('budget_efficiency')
            metrics['official/evidence_gain_per_sample'] = ep_mean('evidence_gain_per_sample')
            metrics['gate/harmful_drift'] = ep_mean('harmful_drift')
            metrics['gate/focus_core_delta'] = ep_mean('focus_core_delta')
            metrics['gate/wasted_budget_fraction'] = ep_mean('wasted_budget_fraction')
            metrics['gate/empty_selection_fraction'] = ep_mean('empty_selection_fraction')
            metrics['official/terminal_budget_bonus'] = ep_mean('terminal_budget_bonus')

            valid_predict_episodes = [e for e in self.episodes if e.get('predict_hit_valid', False)]
            if valid_predict_episodes:
                predict_hit_count = sum(1 for e in valid_predict_episodes if e.get('predict_hit', False))
                predict_hit5_count = sum(1 for e in valid_predict_episodes if e.get('predict_hit_at_5', e.get('predict_hit', False)))
                metrics['legacy/Predict_Hit@1'] = predict_hit_count / len(valid_predict_episodes)
                metrics['legacy/Predict_Hit@5'] = predict_hit5_count / len(valid_predict_episodes)
            else:
                metrics['legacy/Predict_Hit@1'] = 0.0
                metrics['legacy/Predict_Hit@5'] = 0.0
            metrics['legacy/Predict_Hit_Invalid_Rate'] = 1.0 - (len(valid_predict_episodes) / n if n > 0 else 0.0)
            metrics['legacy/Success_Rate'] = ep_mean('success')
            metrics['legacy/Avg_Physical_Time'] = ep_mean('physical_time_mins', default=0.0) or ep_mean('steps', default=0.0)
            metrics['legacy/Avg_Episodes'] = ep_mean('episodes_completed', default=0.0) or ep_mean('rounds', default=0.0)
            metrics['legacy/Avg_Total_Samples'] = ep_mean('budget')

            hard_episodes = [e for e in self.episodes if e.get('geodesic_dist', 0) > 3]
            metrics['legacy/Num_Hard_Samples'] = len(hard_episodes)
            if hard_episodes:
                hard_success_count = sum(1 for e in hard_episodes if e.get('success', False))
                metrics['legacy/Hard_Success_Rate'] = hard_success_count / len(hard_episodes)
            else:
                metrics['legacy/Hard_Success_Rate'] = float('nan')

            y_true_ep = [1 if e.get('success', False) else 0 for e in self.episodes]
            y_score_ep = [e.get('max_hit_prob', 0.0) for e in self.episodes]
            if len(set(y_true_ep)) > 1:
                try:
                    metrics['legacy/hit_auroc_episode'] = roc_auc_score(y_true_ep, y_score_ep)
                except Exception:
                    pass

            y_true_step = []
            y_score_step = []
            for e in self.episodes:
                traj = e.get('trajectory_probs', [])
                hits = e.get('trajectory_hits', [])
                if traj and hits:
                    y_true_step.extend(hits)
                    y_score_step.extend(traj)
            if len(y_true_step) > 0 and len(set(y_true_step)) > 1:
                try:
                    metrics['legacy/hit_auroc_step'] = roc_auc_score(y_true_step, y_score_step)
                except Exception:
                    pass

        return metrics
