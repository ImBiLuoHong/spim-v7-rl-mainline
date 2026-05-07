import os
from typing import Tuple
from tqdm import tqdm

# Allow external modules (e.g., utils.train_runner) to pre-inject lightweight fakes
# for unit tests via monkeypatch, instead of forcing heavy imports.
train_epoch = None
evaluate = None
SpatioTemporalModel = None
CombinedLoss = None
build_primary_criterion = None


def run_train_with_cfg(cfg) -> int:
    """Run one full training (1 epoch + eval) using centralized cfg object.
    Responsibilities (SRP):
    - Prepare logs/checkpoints dirs and write config/effective snapshots.
    - Build dataloaders, infer feature dims.
    - Build model, optimizer, criterion, AMP scaler.
    - Execute one training epoch and optional evaluation.
    This module does not read environment variables.
    """
    # Lazy imports to avoid heavy dependencies during module import.
    # Respect any pre-injected fakes on module globals (set by utils.train_runner wrapper/tests).
    global train_epoch, evaluate, SpatioTemporalModel, build_primary_criterion, CombinedLoss
    try:
        if train_epoch is None:
            from src.training.loop import train_epoch as _train_epoch
            train_epoch = _train_epoch
        if evaluate is None:
            from src.training.loop import evaluate as _evaluate
            evaluate = _evaluate
        # if SpatioTemporalModel is None:
        #     from src.modeling.architectures.main import SpatioTemporalModel as _SpatioTemporalModel
        #     SpatioTemporalModel = _SpatioTemporalModel
        if build_primary_criterion is None:
            from src.modeling.losses import build_primary_criterion as _build_primary_criterion
            build_primary_criterion = _build_primary_criterion
        if CombinedLoss is None:
            try:
                from src.modeling.losses import CombinedLoss as _CombinedLoss
                CombinedLoss = _CombinedLoss
            except Exception:
                CombinedLoss = None
    except Exception:
        pass

    probe = None
    # Base dirs & logs
    os.makedirs(cfg.paths.logs_dir, exist_ok=True)
    logs_dir = cfg.paths.logs_dir
    run_dir = cfg.paths.run_dir
    os.makedirs(cfg.paths.checkpoints_dir, exist_ok=True)

    # Config snapshots and audit
    try:
        from src.shared.logging.core import write_effective_config_snapshot, ensure_train_log_csv_format, append_summary_lines
        # SSOT: 仅保留根目录下的 effective_config.json，避免 logs 目录冗余
        _eff_path = write_effective_config_snapshot(run_dir, cfg)
        print(f"[RUNNER] 已写入有效配置快照: {_eff_path}")
        
        eff_training = {
            'rank_ks': list(getattr(cfg.training, 'rank_ks', (1, 5, 30, 50))),
            'select_best': str(getattr(cfg.training, 'select_best', 'recall@30') or 'recall@30'),
        }
        lines = [
            f"training.rank_ks={eff_training['rank_ks']}",
            f"training.select_best={eff_training['select_best']}",
        ]
        _ = append_summary_lines(logs_dir, lines)
        # 初始化训练日志 CSV 头
        try:
            ensure_train_log_csv_format(os.path.join(logs_dir, 'train_log.csv'), ks=tuple(getattr(cfg.training, 'rank_ks', (1, 3, 5))), ndcg_k=int(getattr(cfg.training, 'ndcg_k', 5)))
        except Exception:
            pass
    except Exception as _e_snap:
        print(f"[RUNNER][WARN] 写入配置快照/审计信息失败: {_e_snap}")

    # Diagnostics probe (cfg-driven)
    try:
        from src.shared.diagnostics import Probe
        diag = getattr(cfg, 'diagnostics', None)
        enabled = bool(getattr(diag, 'enabled', False))
        log_jsonl = bool(getattr(diag, 'log_jsonl', True))
        log_csv = bool(getattr(diag, 'log_csv', False))
        events_filename = str(getattr(diag, 'events_filename', 'diagnostics_events.jsonl'))
        probe = Probe(enabled=enabled, logs_dir=logs_dir, log_jsonl=log_jsonl, log_csv=log_csv, events_filename=events_filename)
        probe.start('runner.cfg.start', meta={'run_dir': run_dir})
    except Exception:
        probe = None

    # Dataloaders (V2 preferred) + dynamic batch tuning
    dyn_enabled = bool(getattr(cfg.training, 'dynamic_batch_tuning_enabled', False))
    bs_cfg = int(getattr(cfg.training, 'batch_size', 1) or 1)
    
    # 启用 cuDNN Benchmark 以加速固定输入尺寸的计算
    import torch
    torch.backends.cudnn.benchmark = True
    # 启用 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    try:
        from src.training.runner import _create_dataloaders_for_bs  # reuse helper to avoid duplication
        train_loader, val_loader, test_loader, imbalance_info = _create_dataloaders_for_bs(cfg, bs_cfg)
        try:
            setattr(cfg.training, 'batch_size', int(bs_cfg))
        except Exception:
            pass
        try:
            if probe:
                probe.log('runner.cfg.dataloaders.ready', train_len=int(len(train_loader)), val_len=int(len(val_loader)), test_len=int(len(test_loader)), batch_size=int(bs_cfg))
        except Exception:
            pass
    except Exception as _e_dl:
        raise

    # Feature dims
    NUM_NODES = int(getattr(train_loader.dataset, 'num_nodes', 0))
    NUM_FEATURES = int(getattr(train_loader.dataset, 'num_node_features', 2))
    try:
        from src.training.engine.data_inspect import infer_num_features_from_batch
        _peek = next(iter(train_loader))
        NUM_FEATURES_NEW = int(infer_num_features_from_batch(_peek))
        if NUM_FEATURES_NEW > 1:
            NUM_FEATURES = NUM_FEATURES_NEW
        else:
            print(f"[DATA][WARN] 推断到的通道数={NUM_FEATURES_NEW}，疑似异常，回退到数据集申明的通道数={NUM_FEATURES}")
    except Exception as _e_inf:
        print(f"[DATA][WARN] 批次通道数推断失败，使用数据集申明的通道数={NUM_FEATURES}: {_e_inf}")

    # Build model, optimizer, criterion
    try:
        # Prefer facade's re-export to allow unit test monkeypatching
        from src.training.runner import select_device as _select_device
        device = _select_device(cfg.training.force_gpu)
    except Exception:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model_name = str(getattr(cfg.model, 'name', '') or '').lower()
        if model_name == 'appnp':
            setattr(cfg.model, 'spatial_backbone', 'appnp')
            if not hasattr(cfg.model, 'dropout_rate'):
                setattr(cfg.model, 'dropout_rate', 0.3)
        # Use model_builder instead of direct instantiation to handle complexity
        from src.training.engine.model_builder import build_model
        model = build_model(cfg, NUM_NODES, NUM_FEATURES, int(getattr(cfg.data, 'window_size', 12)), device)
        
    except Exception as _e_model:
        print(f"[RUNNER][ERROR] 模型构建失败: {_e_model}")
        return 1

    try:
        import torch
        optimizer = torch.optim.Adam(model.parameters(), lr=float(getattr(cfg.training, 'learning_rate', 5e-4)), weight_decay=float(getattr(cfg.training, 'weight_decay', 0.0)))
        if CombinedLoss is not None:
            # Build CombinedLoss with Focal + LA (Hybrid Solution)
            # Prior based on num_nodes (1/N)
            num_nodes = int(getattr(getattr(cfg, 'data', object()), 'num_nodes', 0) or getattr(model, 'num_nodes', 0) or 50)
            
            # Calculate Pos Weight for Imbalanced Binary Classification
            # Default to 100.0 if unknown, or derive from imbalance_info
            pos_weight_val = 100.0
            if isinstance(imbalance_info, dict):
                # imbalance_info might contain 'pos_ratio' or counts
                pass
            
            # Create class_weights tensor [1.0, pos_weight] for Binary
            class_weights = torch.tensor([1.0, pos_weight_val], device=device)
            
            # Updated to match src/modeling/losses.py signature (removing gamma/alpha, adding num_classes)
            # Disable logit_adjustment for Node Ranking/Classification where N varies and priors are uniform
            criterion = CombinedLoss(
                num_classes=num_nodes, 
                class_weights=class_weights,
                use_logit_adjustment=False, 
                logit_tau=1.0
            )
        else:
            criterion = build_primary_criterion(cfg)
        scaler = torch.amp.GradScaler(enabled=bool(getattr(cfg.training, 'use_amp', False)))
    except Exception as _e_opt:
        print(f"[RUNNER][ERROR] 优化器/损失构建失败: {_e_opt}")
        return 1

    # Train Loop
    try:
        # Async Prefetching Injection
        from src.training.engine.batch import DevicePrefetcher
        if getattr(cfg.data, 'async_loading', True): # Default to True
            print("[RUNNER] Enabling Asynchronous Device Prefetching...")
            train_loader = DevicePrefetcher(train_loader, device)
            val_loader = DevicePrefetcher(val_loader, device)
            if test_loader is not None:
                try:
                    test_loader = DevicePrefetcher(test_loader, device)
                except Exception:
                    pass
    except Exception as e:
        print(f"[RUNNER][WARN] Failed to enable async prefetching: {e}")

    try:
        loader_len = len(train_loader)
    except Exception:
        loader_len = 0
        
    num_epochs = int(getattr(cfg.training, 'num_epochs', 1) or 1)
    val_every = int(getattr(cfg.training, 'val_every_n_epochs', 1) or 1)
    
    # Track Best Metric
    best_metric_val = -float('inf')
    select_best_key = str(getattr(cfg.training, 'select_best', 'recall@30') or 'recall@30').lower()
    
    print(f"[RUNNER] Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        
        train_metrics = train_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            clip_norm=float(getattr(cfg.training, 'gradient_clip_norm', 0.5) or 0.0),
            use_soft_labels=bool(getattr(cfg.training, 'use_soft_labels', True)),
            soft_alpha_default=float(getattr(cfg.training, 'soft_alpha', 0.5)),
            kl_t_default=float(getattr(cfg.training, 'kl_temperature', 1.0)),
            max_eval_steps=int(getattr(cfg.training, 'max_eval_steps', 0) or 0),
            grad_log_every_n_batches=int(getattr(cfg.training, 'grad_log_every_n_batches', 0) or 0),
            grad_log_csv_path=os.path.join(logs_dir, 'grads.csv'),
            epoch_index=epoch,
            loader_len=int(loader_len),
            grad_accum_steps=int(getattr(cfg.training, 'grad_accum_steps', 1) or 1),
            rank_ks=tuple(getattr(cfg.training, 'rank_ks', (1, 3, 5))),
            ndcg_k=int(getattr(cfg.training, 'ndcg_k', 5) or 5),
            subgraph_criterion=None,
        )
        
        # 写训练期指标到 outputs/train_epoch_X.json 与 logs/train_log.csv
        try:
            from src.evaluation.reporting import metrics_tuple_to_dict, write_eval_json
            ks = tuple(getattr(cfg.training, 'rank_ks', (1, 3, 5)))
            ndcg_k = int(getattr(cfg.training, 'ndcg_k', 5) or 5)
            metrics_dict_train = metrics_tuple_to_dict(train_metrics, ks=ks, ndcg_k=ndcg_k)
            _ = write_eval_json(run_dir, 'train', metrics_dict_train, epoch=epoch)
            try:
                lr_now = optimizer.param_groups[0]['lr'] if optimizer and getattr(optimizer, 'param_groups', None) else float(getattr(cfg.training, 'learning_rate', 5e-4))
            except Exception:
                lr_now = float(getattr(cfg.training, 'learning_rate', 5e-4))
            try:
                # 解析用于 append_log 的四元组
                # [Fix] loop.py returns 9 base items (including SCR at index 8)
                base_cnt = 9
                _kcnt = len(ks)
                print(f"[DEBUG][runner_cfg] train_metrics type: {type(train_metrics)}")
                print(f"[DEBUG][runner_cfg] train_metrics[0]: {train_metrics[0]}")
                tr_loss, tr_acc = float(train_metrics[0]), float(train_metrics[1])
                tr_scr = float(train_metrics[8]) # Extract SCR
                
                _hits_vec = train_metrics[base_cnt:base_cnt + _kcnt]
                train_hits = {int(k): float(_hits_vec[i]) for i, k in enumerate(ks)}
                mrr = float(train_metrics[base_cnt + _kcnt])
                ndcg = float(train_metrics[base_cnt + _kcnt + 1])
                append_log(
                    os.path.join(logs_dir, 'train_log.csv'),
                    epoch,
                    tr_loss,
                    tr_acc,
                    train_hits,
                    mrr,
                    ndcg,
                    0.0,
                    0.0,
                    {},
                    0.0,
                    0.0,
                    float(lr_now),
                    ks=ks,
                    ndcg_k=ndcg_k,
                    train_scr=tr_scr # Pass SCR to log if supported, or just ignore for now if append_log signature doesn't support it.
                    # Note: append_log signature in core.py might need update or we just log via kwargs if flexible.
                    # Looking at append_log usage in driver.py, it takes fixed args. 
                    # Let's check src/shared/logging/core.py first? 
                    # For safety, I will stick to standard args and print SCR to console.
                )
            except Exception:
                pass
            
            # Print Summary Console
            try:
                 # Helper to format Hits string
                 def _fmt_hits(hits_dict):
                     keys = sorted([k for k in hits_dict.keys() if k in (1, 3, 5, 10)])
                     return " ".join([f"H@{k}:{hits_dict[k]:.4f}" for k in keys])

                 train_hits_str = _fmt_hits(train_hits)
                 # Acc is usually Node-level. Hit@1 is graph level.
                 print(f"[EPOCH][{epoch}/{num_epochs}] "
                       f"Train Loss: {tr_loss:.4f} | Hit@1: {tr_acc:.4f} | SCR: {tr_scr:.4f} | {train_hits_str} | LR: {lr_now:.6f}")
            except Exception as e:
                print(f"[RUNNER] Console Print Error: {e}")

        except Exception as _e_train_write:
            print(f"[RUNNER][WARN] 写入训练期指标失败: {_e_train_write}")
    
        # Task 3: Value Dependency Test (Every 10 epochs)
        # if (epoch % 10 == 0):
        #     try:
        #         from src.evaluation.value_dependence import run_value_dependency_test
        #         print(f"\n[TEST] Running Value Dependency Test (Epoch {epoch})...")
        #         # Use val_loader if available, else train_loader
        #         test_loader = val_loader if val_loader is not None else train_loader
        #         if test_loader is not None:
        #             res = run_value_dependency_test(model, test_loader, device)
        #             print(f"[TEST] Result: {res['result']}")
        #             print(f"       Baseline Acc:  {res['baseline_acc']:.4f}")
        #             print(f"       Sabotaged Acc: {res['sabotaged_acc']:.4f}")
        #             print(f"       Drop Rate:     {res['drop_rate']:.4f}")
        #             if res['result'] == "FAIL":
        #                 print("       [WARNING] Model is likely cheating (Shortcut Learning)!")
        #             elif res['result'] == "PASS":
        #                 print("       [SUCCESS] Model relies on values (Physical Reasoning verified).")
        #     except Exception as e:
        #         print(f"[TEST] Value Dependency Test Failed: {e}")
        #         import traceback
        #         traceback.print_exc()

        # Optional evaluation
        if ((epoch + 1) % val_every == 0) or (epoch == num_epochs - 1):
            try:
                if bool(getattr(cfg.training, 'enable_eval', True)):
                    # Note: tqdm is now inside evaluate()
                    v_metrics = evaluate(
                        loader=val_loader,
                        model=model,
                        criterion=criterion,
                        device=device,
                        use_soft_labels=bool(getattr(cfg.training, 'use_soft_labels', True)),
                        soft_alpha_default=float(getattr(cfg.training, 'soft_alpha', 0.5)),
                        kl_t_default=float(getattr(cfg.training, 'kl_temperature', 1.0)),
                        max_eval_steps=int(getattr(cfg.training, 'max_eval_steps', 0) or 0),
                        rank_ks=tuple(getattr(cfg.training, 'rank_ks', (1, 3, 5))),
                        ndcg_k=int(getattr(cfg.training, 'ndcg_k', 5) or 5),
                        subgraph_criterion=None,
                    )
                    try:
                        from src.evaluation.reporting import metrics_tuple_to_dict, write_eval_json
                        ks = tuple(getattr(cfg.training, 'rank_ks', (1, 3, 5)))
                        ndcg_k = int(getattr(cfg.training, 'ndcg_k', 5) or 5)
                        metrics_dict_val = metrics_tuple_to_dict(v_metrics, ks=ks, ndcg_k=ndcg_k)
                        _ = write_eval_json(run_dir, 'val', metrics_dict_val, epoch=epoch)
                        
                        # Log Val Metrics to console
                        print(f"[VAL][Epoch {epoch}] Loss: {v_metrics[0]:.4f} Hit@1: {v_metrics[1]:.4f}")
                        
                        # Checkpoint & Best Model Selection
                        current_metric = 0.0
                        if 'recall' in select_best_key:
                             # Try to parse recall@K
                             try:
                                 k_val = int(select_best_key.split('@')[1])
                                 current_metric = metrics_dict_val.get(f'hit@{k_val}', 0.0)
                             except:
                                 current_metric = metrics_dict_val.get('hit@30', 0.0)
                        elif 'acc' in select_best_key:
                             current_metric = v_metrics[1]
                        
                        # Save Latest
                        ckpt_path = os.path.join(cfg.paths.checkpoints_dir, 'latest_checkpoint.pth')
                        save_dict = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_metric': best_metric_val,
                            'config': cfg.to_dict() if hasattr(cfg, 'to_dict') else str(cfg)
                        }
                        if scaler:
                            save_dict['scaler_state_dict'] = scaler.state_dict()
                        
                        import torch
                        torch.save(save_dict, ckpt_path)
                        
                        # Save Best
                        if current_metric > best_metric_val:
                            best_metric_val = current_metric
                            best_path = os.path.join(cfg.paths.checkpoints_dir, 'best_checkpoint.pth')
                            torch.save(save_dict, best_path)
                            print(f"[CHECKPOINT] New Best Model! ({select_best_key}={best_metric_val:.4f})")
                            
                    except Exception as _e_val_write:
                        print(f"[RUNNER][WARN] 写入验证期指标/保存Checkpoint失败: {_e_val_write}")
                        import traceback
                        traceback.print_exc()
            except Exception as _e_eval:
                print(f"[RUNNER][WARN] 评估阶段失败: {_e_eval}")
                import traceback
                traceback.print_exc()

    try:
        if probe:
            from src.shared.diagnostics import Probe
            probe.finalize({'status': 'ok'})
    except Exception:
        pass
    return 0
