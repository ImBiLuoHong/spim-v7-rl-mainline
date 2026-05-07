import os
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Iterable

from src.shared.logging.logging.common import _get, _to_serializable


def write_effective_config_snapshot(run_dir: str, cfg, effective: Optional[dict] = None, applied_overrides: Optional[Iterable] = None):
    """Write a comprehensive effective_config.json under {run_dir}/logs.

    Schema:
    {
      "version": 1,
      "timestamp": ISO8601,
      "seed": cfg.seed,
      "applied_overrides": ["a.b", ...],
      "effective": {
         "paths": { "run_dir": ..., "logs_dir": ... },
         "data":  { "use_dataloader_v2": ..., "enable_flow_edge": ..., "enable_travel_time_edge": ... },
         "features": { ... },
         "loss": { ... },
         "training": { ... },
         "model": { ... },
         "summary": { ... }
         ,"diagnostics": { "enabled": ..., "probes": ..., "log_jsonl": ..., "log_csv": ..., "events_filename": ... }
      }
    }
    """
    try:
        # Derive logs_dir from cfg.paths and the provided run_dir parameter (prefer run_dir when absolute)
        logs_dir = None
        try:
            p_run_cfg = _get(cfg, 'paths.run_dir', '')
            p_logs_cfg = _get(cfg, 'paths.logs_dir', '')
            # Prefer the explicit run_dir parameter if provided; if cfg run_dir is relative, override with param
            p_run_effective = run_dir or p_run_cfg
            try:
                if p_run_cfg and run_dir and not os.path.isabs(p_run_cfg):
                    p_run_effective = run_dir
            except Exception:
                # Fallback to parameter if available
                p_run_effective = run_dir or p_run_cfg
            logs_dir = p_logs_cfg or (os.path.join(p_run_effective, 'logs') if p_run_effective else '')
        except Exception:
            logs_dir = os.path.join(run_dir, 'logs') if run_dir else ''

        payload = {
            'version': 1,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'seed': _get(cfg, 'training.seed', _get(cfg, 'seed', None)),
            'applied_overrides': [str(x) for x in (applied_overrides or [])],
            'effective': {
                'paths': {
                    'run_dir': (run_dir or _get(cfg, 'paths.run_dir', run_dir)) or run_dir,
                    'logs_dir': logs_dir,
                    'root_dir': _get(cfg, 'paths.root_dir', None),
                    'experiments_dir': _get(cfg, 'paths.experiments_dir', None),
                    'checkpoints_dir': _get(cfg, 'paths.checkpoints_dir', None),
                    'artifacts_dir': _get(cfg, 'paths.artifacts_dir', None),
                    'artifacts_strategy': _get(cfg, 'paths.artifacts_strategy', None),
                    'data_split_dir': _get(cfg, 'paths.data_split_dir', None),
                    'node_map_path': _get(cfg, 'paths.node_map_path', None),
                    'pipe_map_path': _get(cfg, 'paths.pipe_map_path', None),
                    'sensor_nodes_path': _get(cfg, 'paths.sensor_nodes_path', None),
                    'cmp_nodes_path': _get(cfg, 'paths.cmp_nodes_path', None),
                    'soft_label_path': _get(cfg, 'paths.soft_label_path', None),
                    't0_artifact_path': _get(cfg, 'paths.t0_artifact_path', None),
                    'baseline_cnh2cl_path': _get(cfg, 'paths.baseline_cnh2cl_path', None),
                    'posenc_path': _get(cfg, 'paths.posenc_path', None),
                    'baseline_dir': _get(cfg, 'paths.baseline_dir', None),
                },
                'data': {
                    'use_dataloader_v2': _get(cfg, 'data.use_dataloader_v2', None),
                    'v2_dataset': _get(cfg, 'data.v2_dataset', {}),
                    'window_size': _get(cfg, 'data.window_size', None),
                    'num_workers': _get(cfg, 'data.num_workers', None),
                    'pin_memory': _get(cfg, 'data.pin_memory', None),
                    'prefetch_factor': _get(cfg, 'data.prefetch_factor', None),
                    'persistent_workers': _get(cfg, 'data.persistent_workers', None),
                    'normalize': _get(cfg, 'data.normalize', None),
                    'noise_std': _get(cfg, 'data.noise_std', None),
                    'use_augmentation': _get(cfg, 'data.use_augmentation', None),
                    'include_sensor_mask_channel': _get(cfg, 'data.include_sensor_mask_channel', None),
                    'include_cmp_mask_channel': _get(cfg, 'data.include_cmp_mask_channel', None),
                    'include_node_type_channel': _get(cfg, 'data.include_node_type_channel', None),
                    'toc_channel_enabled': _get(cfg, 'data.toc_channel_enabled', None),
                    'delta_t_channel_enabled': _get(cfg, 'data.delta_t_channel_enabled', None),
                    'include_reveal_channel': _get(cfg, 'data.include_reveal_channel', None),
                    'include_sample_residual_channel': _get(cfg, 'data.include_sample_residual_channel', None),
                    'residual_mask_union': _get(cfg, 'data.residual_mask_union', None),
                    'include_baseline_diff_channel': _get(cfg, 'data.include_baseline_diff_channel', None),
                    'mask_baseline_diff_to_sensors': _get(cfg, 'data.mask_baseline_diff_to_sensors', None),
                    'temporal_diff_enabled': _get(cfg, 'data.temporal_diff_enabled', None),
                    'mask_residual_to_sensors': _get(cfg, 'data.mask_residual_to_sensors', None),
                    'tail_broadcast_k': _get(cfg, 'data.tail_broadcast_k', None),
                    'node_type_dim': _get(cfg, 'data.node_type_dim', None),
                    'sample_residual_mode': _get(cfg, 'data.sample_residual_mode', None),
                    'pipe_geometry_path': _get(cfg, 'data.pipe_geometry_path', None),
                    'use_inp_for_geometry': _get(cfg, 'data.use_inp_for_geometry', None),
                    'network_inp_path': _get(cfg, 'data.network_inp_path', None),
                    'virtual_edges_enabled': _get(cfg, 'data.virtual_edges_enabled', None),
                    'virtual_edge_weight': _get(cfg, 'data.virtual_edge_weight', None),
                    'default_diameter_m': _get(cfg, 'data.default_diameter_m', None),
                    'smoke_collate': _get(cfg, 'data.smoke_collate', None),
                    'v2_make_subgraph_mask': _get(cfg, 'data.v2_make_subgraph_mask', None),
                    'enable_flow_edge': _get(cfg, 'data.enable_flow_edge', None),
                    'enable_travel_time_edge': _get(cfg, 'data.enable_travel_time_edge', None),
                    'alert_manager': _get(cfg, 'data.alert_manager', None),
                    'alert': _get(cfg, 'data.alert', None),
                    'time': _get(cfg, 'data.time', None),
                },
                'features': {
                    'selector_head_enabled': _get(cfg, 'features.selector_head_enabled', _get(cfg, 'features.enable_selector_head', None)),
                    'enable_selector_head': _get(cfg, 'features.enable_selector_head', _get(cfg, 'features.selector_head_enabled', None)),
                    'selector_normalization': _get(cfg, 'features.selector_normalization', None),
                    'selector_target_k': _get(cfg, 'features.selector_target_k', None),
                    'selector_hidden_dim': _get(cfg, 'features.selector_hidden_dim', None),
                    'softmax_temperature': _get(cfg, 'features.softmax_temperature', None),
                    'enable_subgraph_loss': _get(cfg, 'features.enable_subgraph_loss', None),
                    'subgraph_lambda_bce': _get(cfg, 'features.subgraph_lambda_bce', None),
                    'subgraph_lambda_tv': _get(cfg, 'features.subgraph_lambda_tv', None),
                    'subgraph_lambda_size': _get(cfg, 'features.subgraph_lambda_size', None),
                    'subgraph_target_k': _get(cfg, 'features.subgraph_target_k', None),
                    'subgraph_pos_beta': _get(cfg, 'features.subgraph_pos_beta', None),
                    'coverage_alpha': _get(cfg, 'features.coverage_alpha', None),
                    'prune_upstream': {
                        'enabled': _get(cfg, 'features.prune_upstream.enabled', None),
                        'window_s': _get(cfg, 'features.prune_upstream.window_s', None),
                    },
                    'connected_eval_enabled': _get(cfg, 'features.connected_eval_enabled', None),
                    'connected_eval': {
                        'enabled': _get(cfg, 'features.connected_eval.enabled', _get(cfg, 'features.connected_eval_enabled', None)),
                        'metrics_full_graph': _get(cfg, 'features.connected_eval.metrics_full_graph', True),
                    },
                    'prior': {
                        'enabled': _get(cfg, 'features.prior.enabled', False),
                        'mode': _get(cfg, 'features.prior.mode', None),
                        'alpha': _get(cfg, 'features.prior.alpha', None),
                        'beta': _get(cfg, 'features.prior.beta', None),
                        'eps': _get(cfg, 'features.prior.eps', None),
                    },
                    'contract': {
                        'audit': {
                            'enabled': _get(cfg, 'features.contract.audit.enabled', None),
                            'require_single_source': _get(cfg, 'features.contract.audit.require_single_source', None),
                            'drop_invalid_samples': _get(cfg, 'features.contract.audit.drop_invalid_samples', None),
                            'log_details': _get(cfg, 'features.contract.audit.log_details', None),
                        }
                    },
                },
                'loss': {
                    'causal': {
                        'enabled': _get(cfg, 'loss.causal.enabled', None),
                        'weight': _get(cfg, 'loss.causal.weight', None),
                        'tau': _get(cfg, 'loss.causal.tau', None),
                        'mode': _get(cfg, 'loss.causal.mode', 'static'),
                    },
                    'tv': {
                        'enabled': _get(cfg, 'loss.tv.enabled', None),
                        'weight': _get(cfg, 'loss.tv.weight', None),
                        'tau': _get(cfg, 'loss.tv.tau', None),
                    },
                    'conn': {
                        'enabled': _get(cfg, 'loss.conn.enabled', None),
                        'weight': _get(cfg, 'loss.conn.weight', None),
                        'mode': _get(cfg, 'loss.conn.mode', 'laplacian'),
                        'tau_time': _get(cfg, 'loss.conn.tau_time', None),
                        'use_tt': _get(cfg, 'loss.conn.use_tt', None),
                    },
                    'coverage': {
                        'enabled': _get(cfg, 'loss.coverage.enabled', None),
                        'alpha': _get(cfg, 'loss.coverage.alpha', None),
                        'weight': _get(cfg, 'loss.coverage.weight', None),
                    },
                    'size': {
                        'enabled': _get(cfg, 'loss.size.enabled', None),
                        'weight': _get(cfg, 'loss.size.weight', None),
                    },
                },
                'labels': {
                    'type': _get(cfg, 'labels.type', None),
                    'normalize_sum': _get(cfg, 'labels.normalize_sum', True),
                    'sigma': _get(cfg, 'labels.sigma', None),
                    'epsilon': _get(cfg, 'labels.epsilon', None),
                    'max_hops': _get(cfg, 'labels.max_hops', None),
                    'direction': _get(cfg, 'labels.direction', None),
                    'edge_weight': _get(cfg, 'labels.edge_weight', None),
                    'use_row_normalization': _get(cfg, 'labels.use_row_normalization', None),
                },
                'training': {
                    'num_epochs': _get(cfg, 'training.num_epochs', None),
                    'batch_size': _get(cfg, 'training.batch_size', None),
                    'rank_ks': _get(cfg, 'training.rank_ks', None),
                    'select_best': _get(cfg, 'training.select_best', None),
                    'ndcg_k': _get(cfg, 'training.ndcg_k', None),
                    'early_stop': _get(cfg, 'training.early_stop', None),
                    'patience': _get(cfg, 'training.patience', None),
                    'use_amp': _get(cfg, 'training.use_amp', None),
                    'gradient_clip_norm': _get(cfg, 'training.gradient_clip_norm', None),
                    'learning_rate': _get(cfg, 'training.learning_rate', None),
                    'weight_decay': _get(cfg, 'training.weight_decay', None),
                    'lr_scheduler': _get(cfg, 'training.lr_scheduler', None),
                    'lr_factor': _get(cfg, 'training.lr_factor', None),
                    'lr_patience': _get(cfg, 'training.lr_patience', None),
                    'cosine_tmax_epochs': _get(cfg, 'training.cosine_tmax_epochs', None),
                    'cosine_eta_min': _get(cfg, 'training.cosine_eta_min', None),
                    'run_name': _get(cfg, 'training.run_name', None),
                    'resume': _get(cfg, 'training.resume', None),
                    'enable_eval': _get(cfg, 'training.enable_eval', None),
                    'test_best': _get(cfg, 'training.test_best', None),
                    'grad_accum_steps': _get(cfg, 'training.grad_accum_steps', None),
                    'force_gpu': _get(cfg, 'training.force_gpu', None),
                    'cudnn_benchmark': _get(cfg, 'training.cudnn_benchmark', None),
                    'cudnn_deterministic': _get(cfg, 'training.cudnn_deterministic', None),
                    'val_every_n_epochs': _get(cfg, 'training.val_every_n_epochs', None),
                    'save_every_n_epochs': _get(cfg, 'training.save_every_n_epochs', None),
                    'use_subgraph_head': _get(cfg, 'training.use_subgraph_head', None),
                    'subgraph_lambda_bce': _get(cfg, 'training.subgraph_lambda_bce', None),
                    'subgraph_lambda_tv': _get(cfg, 'training.subgraph_lambda_tv', None),
                    'subgraph_lambda_size': _get(cfg, 'training.subgraph_lambda_size', None),
                    'subgraph_target_k': _get(cfg, 'training.subgraph_target_k', None),
                    'subgraph_pos_beta': _get(cfg, 'training.subgraph_pos_beta', None),
                    'softmax_temperature': _get(cfg, 'training.softmax_temperature', None),
                    'diag_print_grad': _get(cfg, 'training.diag_print_grad', None),
                    'diag_print_top1': _get(cfg, 'training.diag_print_top1', None),
                    'grad_log_every_n_batches': _get(cfg, 'training.grad_log_every_n_batches', None),
                    'max_eval_steps': _get(cfg, 'training.max_eval_steps', None),
                    'overfit_one_batch': _get(cfg, 'training.overfit_one_batch', None),
                    'overfit_steps': _get(cfg, 'training.overfit_steps', None),
                    'ofb_fixed_batch_cpu': _get(cfg, 'training.ofb_fixed_batch_cpu', None),
                    'ofb_lr': _get(cfg, 'training.ofb_lr', None),
                    'ofb_save_every': _get(cfg, 'training.ofb_save_every', None),
                    'print_ce_kl': _get(cfg, 'training.print_ce_kl', None),
                    'kl_temperature': _get(cfg, 'training.kl_temperature', None),
                    'label_smoothing': _get(cfg, 'training.label_smoothing', None),
                    'logit_adj_mode': _get(cfg, 'training.logit_adj_mode', None),
                    'logit_tau': _get(cfg, 'training.logit_tau', None),
                    'use_logit_adjustment': _get(cfg, 'training.use_logit_adjustment', None),
                    'use_weighted_sampler': _get(cfg, 'training.use_weighted_sampler', None),
                    'use_soft_labels': _get(cfg, 'training.use_soft_labels', None),
                    'soft_alpha': _get(cfg, 'training.soft_alpha', None),
                    'soft_alpha_start': _get(cfg, 'training.soft_alpha_start', None),
                    'soft_alpha_target': _get(cfg, 'training.soft_alpha_target', None),
                    'soft_alpha_ramp_epochs': _get(cfg, 'training.soft_alpha_ramp_epochs', None),
                    'lr_warmup_epochs': _get(cfg, 'training.lr_warmup_epochs', None),
                    'seed': _get(cfg, 'training.seed', None),
                    'classification_weight': _get(cfg, 'training.classification_weight', None),
                    'class_weight_strategy': _get(cfg, 'training.class_weight_strategy', None),
                    'class_weight_beta': _get(cfg, 'training.class_weight_beta', None),
                    'class_weight_epsilon': _get(cfg, 'training.class_weight_epsilon', None),
                },
                'model': {
                    'backbone': _get(cfg, 'model.backbone', _get(cfg, 'model.temporal_backbone', _get(cfg, 'model.arch', None))),
                    'temporal_backbone': _get(cfg, 'model.temporal_backbone', _get(cfg, 'model.backbone', None)),
                    'arch': _get(cfg, 'model.arch', None),
                    'hidden_dim': _get(cfg, 'model.hidden_dim', None),
                    'num_gnn_layers': _get(cfg, 'model.num_gnn_layers', None),
                    'num_tcn_layers': _get(cfg, 'model.num_tcn_layers', None),
                    'tcn_kernel_size': _get(cfg, 'model.tcn_kernel_size', None),
                    'dropout_rate': _get(cfg, 'model.dropout_rate', None),
                    'disable_dropout': _get(cfg, 'model.disable_dropout', None),
                    'posenc_dim': _get(cfg, 'model.posenc_dim', None),
                    'attention_heads': _get(cfg, 'model.attention_heads', None),
                    'use_dual_branch': _get(cfg, 'model.use_dual_branch', None),
                    'enable_feature_mixer': _get(cfg, 'model.enable_feature_mixer', None),
                    'virtual_edge_mix_alpha': _get(cfg, 'model.virtual_edge_mix_alpha', None),
                    'temporal': {
                        'use_residuals': _get(cfg, 'model.temporal.use_residuals', False),
                    },
                    'spatial': {
                        'use_residuals': _get(cfg, 'model.spatial.use_residuals', False),
                    },
                    'spatial_backbone': _get(cfg, 'model.spatial_backbone', None),
                    'pooling_type': _get(cfg, 'model.pooling_type', None),
                    'mc_dropout_eval': _get(cfg, 'model.mc_dropout_eval', None),
                    'use_subgraph_head': _get(cfg, 'model.use_subgraph_head', None),
                    'selector_head_enabled': _get(cfg, 'model.selector_head_enabled', _get(cfg, 'model.enable_selector_head', None)),
                },
                'summary': {
                    'num_nodes': _get(cfg, 'summary.num_nodes', None),
                    'num_features': _get(cfg, 'summary.num_features', None),
                    'edge_dim_main': _get(cfg, 'summary.edge_dim_main', None),
                    'temporal_backbone': _get(cfg, 'summary.temporal_backbone', None),
                    'trace_time_limit': _get(cfg, 'summary.trace_time_limit', None),
                },
                'diagnostics': {
                    'enabled': _get(cfg, 'diagnostics.enabled', None),
                    'probes': _get(cfg, 'diagnostics.probes', None),
                    'log_jsonl': _get(cfg, 'diagnostics.log_jsonl', None),
                    'log_csv': _get(cfg, 'diagnostics.log_csv', None),
                    'events_filename': _get(cfg, 'diagnostics.events_filename', None),
                }
            }
        }

        if isinstance(effective, dict):
            try:
                eff = payload.get('effective', {})
                for k, v in effective.items():
                    eff[k] = _to_serializable(v)
                payload['effective'] = eff
            except Exception:
                pass

        try:
            ao_list = list(payload.get('applied_overrides', []) or [])
            if 'features.selector_head_enabled' not in ao_list:
                ao_list.append('features.selector_head_enabled')
            if 'model.selector_head_enabled' not in ao_list:
                ao_list.append('model.selector_head_enabled')
            payload['applied_overrides'] = ao_list
        except Exception:
            pass

        try:
            norm = None
            try:
                norm = payload['effective']['features'].get('selector_normalization', None)
            except Exception:
                norm = _get(cfg, 'features.selector_normalization', None)
            f_head = bool(_get(cfg, 'features.selector_head_enabled', _get(cfg, 'features.enable_selector_head', False)) or False)
            m_head = bool(_get(cfg, 'model.selector_head_enabled', _get(cfg, 'model.enable_selector_head', False)) or False)
            trigger_softmaxK = (str(norm).lower() == 'softmaxk')
            trigger_override = (f_head or m_head)
            if trigger_softmaxK or trigger_override:
                try:
                    payload['effective']['features']['selector_head_enabled'] = True
                    payload['effective']['model']['selector_head_enabled'] = True
                except Exception:
                    pass
                try:
                    run_dir_eff = payload['effective']['paths'].get('run_dir') or (run_dir or '')
                    print(f"[SELECTOR][gate] forced=true reason=softmaxK_or_override run_dir={run_dir_eff}")
                except Exception:
                    pass
                try:
                    if logs_dir:
                        os.makedirs(logs_dir, exist_ok=True)
                        fb_path = os.path.join(logs_dir, 'eval_fallback.jsonl')
                        rec = {
                            'event': 'config',
                            'reason': 'selector_head_forced_true',
                            'trigger': 'softmaxK_or_override',
                            'ts': datetime.now().isoformat(timespec='seconds'),
                        }
                        with open(fb_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            canonical_eff = json.dumps(_to_serializable(payload.get('effective', {})), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            payload['sha256'] = hashlib.sha256(canonical_eff.encode('utf-8')).hexdigest()
        except Exception:
            pass

        out_path = None
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            out_path = os.path.join(logs_dir, 'effective_config.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(_to_serializable(payload), ensure_ascii=False, indent=2))

        try:
            if run_dir:
                os.makedirs(run_dir, exist_ok=True)
                root_path = os.path.join(run_dir, 'effective_config.json')
                if not os.path.exists(root_path):
                    stub = {
                        'message': 'See logs/effective_config.json',
                        'redirect': 'logs/effective_config.json',
                    }
                    with open(root_path, 'w', encoding='utf-8') as rf:
                        rf.write(json.dumps(stub, ensure_ascii=False))
        except Exception:
            pass

        return out_path
    except Exception as e:
        try:
            print(f"[WARN] write_effective_config_snapshot failed: {e}")
        except Exception:
            pass
        return None


def write_config_snapshot(logs_dir: str, cfg, runtime: Optional[Dict] = None, env_vars: Optional[Dict] = None) -> Optional[str]:
    """Write a config_snapshot.json with cfg.to_dict plus runtime and env_vars."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'config_snapshot.json')
        data = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'seed': _get(cfg, 'seed', None),
            'runtime': _to_serializable(runtime or {}),
            'env': _to_serializable(env_vars or {}),
            'cfg': _to_serializable(cfg.to_dict() if hasattr(cfg, 'to_dict') else {}),
        }
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_config_snapshot failed: {e}")
        except Exception:
            pass
        return None


def write_config_summary(logs_dir: str, cfg) -> Optional[str]:
    """Write a human-readable summary to logs/config_summary.txt."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'config_summary.txt')
        lines = []
        lines.append(f"run_dir={_get(cfg, 'paths.run_dir', '')}")
        lines.append(f"logs_dir={_get(cfg, 'paths.logs_dir', '')}")
        lines.append(f"model.backbone={_get(cfg, 'model.backbone', _get(cfg, 'model.spatial_backbone', 'gatv2'))}")
        lines.append(f"training.epochs={_get(cfg, 'training.num_epochs', 1)}")
        lines.append(f"training.batch_size={_get(cfg, 'training.batch_size', '')}")
        lines.append(f"features.selector_target_k={_get(cfg, 'features.selector_target_k', '')}")
        lines.append(f"loss.coverage.enabled={_get(cfg, 'loss.coverage.enabled', False)} alpha={_get(cfg, 'loss.coverage.alpha', 0.0)} weight={_get(cfg, 'loss.coverage.weight', 1.0)}")
        try:
            labels_line = (
                f"labels.type={_get(cfg, 'labels.type', 'gaussian')} "
                f"normalize_sum={_get(cfg, 'labels.normalize_sum', True)} "
                f"sigma={_get(cfg, 'labels.sigma', '')} "
                f"epsilon={_get(cfg, 'labels.epsilon', '')} "
                f"max_hops={_get(cfg, 'labels.max_hops', '')} "
                f"direction={_get(cfg, 'labels.direction', '')} "
                f"edge_weight={_get(cfg, 'labels.edge_weight', '')} "
                f"use_row_normalization={_get(cfg, 'labels.use_row_normalization', '')}"
            )
            lines.append(labels_line)
        except Exception:
            pass
        txt = '\n'.join(lines) + '\n'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(txt)
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_config_summary failed: {e}")
        except Exception:
            pass
        return None


def append_summary_lines(logs_dir: str, lines: Iterable[str]) -> Optional[str]:
    """Append lines to logs/config_summary.txt (creates file if missing)."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'config_summary.txt')
        with open(path, 'a', encoding='utf-8') as f:
            for line in (lines or []):
                f.write(str(line).rstrip() + '\n')
        return path
    except Exception as e:
        try:
            print(f"[WARN] append_summary_lines failed: {e}")
        except Exception:
            pass
        return None


def append_applied_overrides_to_summary(logs_dir: str, overrides: Dict, effective: Optional[Dict] = None) -> Optional[str]:
    """Append a structured 'Applied Overrides' block to logs/config_summary.txt."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'config_summary.txt')
        lines = []
        lines.append('=== Applied Overrides ===')
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                try:
                    frm = v.get('from')
                    to = v.get('to')
                    rsn = v.get('reason')
                    lines.append(f"{k}: {frm} -> {to} reason={rsn}")
                except Exception:
                    lines.append(f"{k}: {v}")
        if isinstance(effective, dict) and effective:
            try:
                eff_json = json.dumps(_to_serializable(effective), ensure_ascii=False)
                lines.append(f"effective={eff_json}")
            except Exception:
                pass
        with open(path, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(str(line).rstrip() + '\n')
        return path
    except Exception as e:
        try:
            print(f"[WARN] append_applied_overrides_to_summary failed: {e}")
        except Exception:
            pass
        return None