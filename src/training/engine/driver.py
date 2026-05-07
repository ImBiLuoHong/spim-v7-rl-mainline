import os
from tqdm import tqdm


def run_ofb(
    cfg,
    model,
    optimizer,
    train_loader,
    criterion_ce_only,
    device,
    epochs,
    OFB_FIXED_BATCH_CPU,
    OVERFIT_STEPS,
    DIAG_PRINT_GRAD,
    DIAG_PRINT_TOP1,
    PRINT_CE_KL,
    SOFT_ALPHA,
    KL_T,
    CLIP_NORM,
    CLASS_W,
    LABEL_SMOOTH,
    cekl_csv_path,
    log_csv_path,
    RANK_KS,
    NDCG_K,
    lr,
    SELECT_BEST,
    best_val_loss,
    checkpoint_path,
    OFB_SAVE_EVERY,
    logs_dir,
):
    from src.shared.artifacts import save_checkpoint, cleanup_ofb_checkpoints
    from src.training.ofb import train_epoch_ofb
    from src.shared.logging.core import append_log, write_heartbeat, should_stop
    from src.evaluation.reporting import metrics_tuple_to_dict, write_eval_json

    print("[OFB] 单批过拟合路径启用，开始执行...")
    cleanup_ofb_checkpoints(cfg.paths.checkpoints_dir)
    for epoch in range(1, epochs + 1):
        avg_loss, avg_acc, avg_ce, avg_kl, avg_class, avg_total, avg_alpha, avg_T = train_epoch_ofb(
            epoch,
            model,
            optimizer,
            train_loader,
            criterion_ce_only,
            device,
            OFB_FIXED_BATCH_CPU,
            OVERFIT_STEPS,
            DIAG_PRINT_GRAD,
            DIAG_PRINT_TOP1,
            PRINT_CE_KL,
            SOFT_ALPHA,
            KL_T,
            CLIP_NORM,
            CLASS_W,
            LABEL_SMOOTH,
            cekl_csv_path,
        )
        try:
            append_log(
                log_csv_path,
                epoch,
                avg_loss,
                avg_acc,
                {int(k): 0.0 for k in RANK_KS},
                0.0,
                0.0,
                0.0,
                0.0,
                {int(k): 0.0 for k in RANK_KS},
                0.0,
                0.0,
                lr,
                ks=RANK_KS,
                ndcg_k=NDCG_K,
            )
            try:
                metrics_dict_train = metrics_tuple_to_dict((
                    avg_loss, avg_acc, avg_ce, avg_kl, avg_class, avg_total, avg_alpha, avg_T,
                    *([0.0] * len(RANK_KS)),
                    0.0, 0.0,
                ), ks=RANK_KS, ndcg_k=NDCG_K)
                _ = write_eval_json(cfg.paths.run_dir, 'train', metrics_dict_train, epoch=epoch)
            except Exception as _e_json:
                print(f"[EVAL][OFB] 写入训练报告失败: {_e_json}")
            if (epoch % max(1, OFB_SAVE_EVERY)) == 0:
                save_checkpoint(
                    cfg.paths.run_dir,
                    model,
                    optimizer,
                    epoch,
                    metrics={'train_loss': avg_loss, 'train_acc': avg_acc},
                    is_best_val=False,
                    is_best_train=True
                )
            write_heartbeat(logs_dir, epoch)

        except Exception as _e:
            print(f"[DRIVER][OFB] 记录日志失败: {_e}")
        try:
            lr_now = optimizer.param_groups[0]['lr'] if optimizer and getattr(optimizer, 'param_groups', None) else lr
            print(f"[EPOCH][{epoch}/{epochs}] OFB train_loss={avg_loss:.6f} train_acc={avg_acc:.4f} lr={float(lr_now):.6f}")
        except Exception:
            pass
        if should_stop(logs_dir):
            print("[DRIVER][OFB] 早停触发，结束OFB路径")
            break
    from src.shared.logging.core import write_finished_ok
    write_finished_ok(logs_dir)


from src.evaluation.value_dependence import run_value_dependency_test

def run_normal_training(
    cfg,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    criterion,
    scaler,
    subgraph_criterion,
    logs_dir,
    run_dir,
    epochs,
    start_epoch,
    WARMUP_EPOCHS,
    lr,
    USE_SOFT_LABELS,
    ALPHA_START,
    ALPHA_TARGET,
    ALPHA_RAMP,
    SOFT_ALPHA,
    KL_T,
    CLIP_NORM,
    MAX_EVAL_STEPS,
    RANK_KS,
    NDCG_K,
    PRINT_CE_KL,
):
    from src.shared.logging.core import append_ce_kl, append_log, write_heartbeat
    from src.training.loop import train_epoch
    from src.evaluation.loop import evaluate
    from src.evaluation.reporting import metrics_tuple_to_dict, write_eval_json, write_metrics_scope_json
    from src.shared.artifacts import save_checkpoint, cleanup_latest_best_keep_last

    print("[TRAIN] 正常训练路径启用")
    cleanup_latest_best_keep_last(cfg.paths.checkpoints_dir, keep_last=5)
    
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        epoch_offset = max(1, epoch - start_epoch)
        if WARMUP_EPOCHS > 0 and epoch_offset <= WARMUP_EPOCHS:
            eff_lr = float(lr) * (float(epoch_offset) / float(max(1, WARMUP_EPOCHS)))
            try:
                for g in optimizer.param_groups:
                    g['lr'] = eff_lr
                print(f"[SCHED][Warmup] epoch={epoch} offset={epoch_offset}/{WARMUP_EPOCHS} lr={eff_lr:.6f}")
            except Exception:
                pass
        else:
            try:
                for g in optimizer.param_groups:
                    g['lr'] = float(lr)
            except Exception:
                pass

        if USE_SOFT_LABELS:
            if ALPHA_RAMP > 0:
                ramp_ratio = min(1.0, float(epoch_offset) / float(max(1, ALPHA_RAMP)))
                alpha_now = float(ALPHA_START) + (float(ALPHA_TARGET) - float(ALPHA_START)) * ramp_ratio
            else:
                alpha_now = float(ALPHA_TARGET)
            try:
                setattr(cfg, 'alpha_now_debug', float(alpha_now))  # optional audit
            except Exception:
                pass
        else:
            alpha_now = float(SOFT_ALPHA)

        train_metrics = train_epoch(
            loader=tqdm(train_loader, total=len(train_loader), desc=f"Train {epoch}/{epochs}", unit="batch", dynamic_ncols=False, leave=False),
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            clip_norm=float(CLIP_NORM),
            grad_accum_steps=int(getattr(cfg.training, 'grad_accum_steps', 1) or 1),
            use_soft_labels=bool(USE_SOFT_LABELS),
            soft_alpha_default=float(alpha_now),
            kl_t_default=float(KL_T),
            max_eval_steps=int(MAX_EVAL_STEPS),
            grad_log_every_n_batches=int(getattr(cfg.training, 'grad_log_every_n_batches', 0) or 0),
            grad_log_csv_path=os.path.join(logs_dir, 'gradients.csv'),
            epoch_index=int(epoch),
            loader_len=len(train_loader),
            rank_ks=tuple(RANK_KS),
            ndcg_k=int(NDCG_K),
            subgraph_criterion=subgraph_criterion,
        )
        _base_cnt = 9
        _kcnt = len(RANK_KS)
        train_loss, train_node_acc, ce, kl, class_mix, total, alpha, T, scr = train_metrics[:_base_cnt]
        _hits_vec = train_metrics[_base_cnt:_base_cnt + _kcnt]
        mrr = train_metrics[_base_cnt + _kcnt]
        ndcg = train_metrics[_base_cnt + _kcnt + 1]
        train_hits = {int(k): float(_hits_vec[i]) for i, k in enumerate(RANK_KS)}

        # Override Acc to be Hit@1 as per user requirement "Acc = Hit@1"
        train_acc = train_hits.get(1, 0.0)

        # 验证与调度器步进
        val_loss = 0.0
        val_node_acc = 0.0
        val_acc = 0.0
        v_hits = {}
        v_mrr = 0.0
        v_ndcg = 0.0
        if getattr(cfg.training, 'enable_eval', True) and (epoch % max(1, cfg.training.val_every_n_epochs) == 0):
            v_metrics = evaluate(
                tqdm(val_loader, total=len(val_loader), desc=f"Val {epoch}/{epochs}", unit="batch", dynamic_ncols=False, leave=False),
                model,
                criterion,
                device,
                USE_SOFT_LABELS,
                SOFT_ALPHA,
                KL_T,
                MAX_EVAL_STEPS,
                rank_ks=RANK_KS,
                ndcg_k=NDCG_K,
                subgraph_criterion=subgraph_criterion,
            )
            _base_cnt = 8
            _kcnt = len(RANK_KS)
            v_loss, v_node_acc, v_ce, v_kl, v_class_mix, v_total, v_alpha, v_T = v_metrics[:_base_cnt]
            _v_hits_vec = v_metrics[_base_cnt:_base_cnt + _kcnt]
            v_mrr = v_metrics[_base_cnt + _kcnt]
            v_ndcg = v_metrics[_base_cnt + _kcnt + 1]
            val_loss = float(v_loss)
            val_node_acc = float(v_node_acc)
            v_hits = {int(k): float(_v_hits_vec[i]) for i, k in enumerate(RANK_KS)}
            
            # Override Acc to be Hit@1
            val_acc = v_hits.get(1, 0.0)

            try:
                metrics_dict = metrics_tuple_to_dict(v_metrics, ks=RANK_KS, ndcg_k=NDCG_K)
                # Ensure the dict also reflects Acc=Hit@1 if needed, or rely on explicit keys
                # metrics_tuple_to_dict likely uses the raw tuple, so 'acc' inside it is still NodeAcc.
                # We should update the dict to include our new definition if possible, but let's check reporting.py first.
                # For now, let's inject it.
                metrics_dict['acc'] = val_acc
                metrics_dict['node_acc'] = val_node_acc
                
                out_path = write_eval_json(run_dir, 'val', metrics_dict, epoch=epoch)
                print(f"[EVAL] 写入验证报告: {out_path}")
                scope = getattr(model, 'metrics_scope_last', None)
                if isinstance(scope, dict) and scope:
                    _ = write_metrics_scope_json(run_dir, scope)
            except Exception as _e:
                print(f"[EVAL] 写入验证报告失败: {_e}")
            # 调度器（验证驱动）
            try:
                scheduler.step(val_loss)
            except Exception:
                pass
        else:
            try:
                scheduler.step(train_loss)
            except Exception:
                pass

        # Task 3: Value Dependency Test (Every 10 epochs)
        if (epoch % 10 == 0):
            try:
                # from src.evaluation.value_dependence import run_value_dependency_test (Moved to top)
                print(f"\n[TEST] Running Value Dependency Test (Epoch {epoch})...")
                # Use val_loader if available, else train_loader
                test_loader = val_loader if val_loader is not None else train_loader
                if test_loader is not None:
                    res = run_value_dependency_test(model, test_loader, device)
                    print(f"[TEST] Result: {res['result']}")
                    print(f"       Baseline Acc:  {res['baseline_acc']:.4f}")
                    print(f"       Sabotaged Acc: {res['sabotaged_acc']:.4f}")
                    print(f"       Drop Rate:     {res['drop_rate']:.4f}")
                    if res['result'] == "FAIL":
                        print("       [WARNING] Model is likely cheating (Shortcut Learning)!")
                    elif res['result'] == "PASS":
                        print("       [SUCCESS] Model relies on values (Physical Reasoning verified).")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[TEST] Value Dependency Test Failed: {e}")

        try:
            _batches = int(len(train_loader))
        except Exception:
            _batches = 0
        append_ce_kl(os.path.join(logs_dir, 'ce_kl.csv'), int(epoch), int(_batches), int(0), float(ce), float(kl), float(alpha), float(T))
        lr_now = None
        try:
            lr_now = optimizer.param_groups[0]['lr'] if optimizer and getattr(optimizer, 'param_groups', None) else lr
        except Exception:
            lr_now = lr
        append_log(
            os.path.join(logs_dir, 'train_log.csv'),
            epoch,
            train_loss,
            train_acc,
            train_hits,
            mrr,
            ndcg,
            val_loss,
            val_acc,
            v_hits,
            v_mrr,
            v_ndcg,
            lr_now,
            train_node_acc=train_node_acc,
            val_node_acc=val_node_acc,
            ks=RANK_KS,
            ndcg_k=NDCG_K,
            train_scr=scr  # Add SCR to log
        )
        write_heartbeat(logs_dir, epoch)
        
        # Print Summary
        try:
             # Get Hit@1 if available
             h1_train = train_hits.get(1, 0.0)
             h1_val = v_hits.get(1, 0.0)
             
             # Helper to format Hits string
             def _fmt_hits(hits_dict):
                 # Filter only 1, 3, 5, 10 if they exist, or just print what we have
                 keys = sorted([k for k in hits_dict.keys() if k in (1, 3, 5, 10)])
                 return " ".join([f"H@{k}:{hits_dict[k]:.4f}" for k in keys])

             train_hits_str = _fmt_hits(train_hits)
             val_hits_str = _fmt_hits(v_hits)

             # Explicitly label as Top1 (Recall@1) and clarify Acc is Node-level
             # User Request: "我要正确的acc 和hit 1 3 5 10统计"
             # We treat Acc as Hit@1 (Top1)
             print(f"[EPOCH][{epoch}/{start_epoch + epochs}] "
                   f"Train Loss: {train_loss:.4f} | NodeAcc: {train_node_acc:.4f} | Acc(H@1): {train_acc:.4f} | SCR: {scr:.4f} | {train_hits_str} || "
                   f"Val Loss: {val_loss:.4f} | NodeAcc: {val_node_acc:.4f} | Acc(H@1): {val_acc:.4f} | {val_hits_str} | LR: {lr_now}")
        except Exception:
             pass

        # === 保存逻辑 ===
        try:
            is_best_train = False
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                is_best_train = True
            
            is_best_val = False
            # 仅当进行了评估且loss有效时更新
            if getattr(cfg.training, 'enable_eval', True) and (epoch % max(1, cfg.training.val_every_n_epochs) == 0):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best_val = True
            
            save_checkpoint(
                run_dir,
                model,
                optimizer,
                epoch,
                metrics={
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                },
                is_best_val=is_best_val,
                is_best_train=is_best_train
            )
        except Exception as _e_save:
            print(f"[WARN] Checkpoint save failed: {_e_save}")
