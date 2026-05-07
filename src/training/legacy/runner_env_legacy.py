import os
import sys
import subprocess
import time
from typing import Optional, Dict


def run_train_with_env(env_overrides: Dict[str, object], quiet: bool = False, log_path: Optional[str] = None) -> int:
    """Launch train.py with environment overrides and tee console to file.

    Responsibilities (SRP):
    - Merge and normalize environment variables.
    - Bind RUN_DIR, artifacts strategy/manifest, and logs snapshot before subprocess.
    - Launch train.py via PTY when possible to preserve TTY refresh; fallback to PIPE.
    - Tee stdout/stderr to log_path if provided.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    env = os.environ.copy()
    env.update({k: str(v) for k, v in (env_overrides or {}).items()})
    env.setdefault('WORKSPACE_ROOT', root)
    env.setdefault('FORCE_GPU', '1')
    try:
        pycache_prefix = os.path.join(root, '.pycache_unified')
        os.makedirs(pycache_prefix, exist_ok=True)
        env.setdefault('PYTHONPYCACHEPREFIX', pycache_prefix)
    except Exception:
        pass

    # Prefer file-backed Config to derive run_dir and paths
    try:
        from src.config.core import Config
        cfg = Config()
        run_dir = getattr(cfg.paths, 'run_dir', None)
        if not run_dir:
            exp_base = getattr(cfg.paths, 'experiments_dir', os.path.join(root, 'experiments'))
            os.makedirs(exp_base, exist_ok=True)
            rn = getattr(cfg.training, 'run_name', None)
            run_dir = os.path.join(exp_base, rn or time.strftime('exp-%Y%m%d-%H%M%S'))
        os.makedirs(run_dir, exist_ok=True)
        env['RUN_DIR'] = run_dir
        env['ARTIFACTS_DIR'] = getattr(cfg.paths, 'artifacts_dir', os.path.join(root, 'data', 'artifacts'))
        env['ARTIFACTS_STRATEGY'] = getattr(cfg.paths, 'artifacts_strategy', 'global')
        if getattr(cfg.paths, 'posenc_path', None):
            env['POSENC_PATH'] = cfg.paths.posenc_path
        if getattr(cfg.paths, 'soft_label_path', None):
            env['SOFT_LABEL_PATH'] = cfg.paths.soft_label_path
        if getattr(cfg.paths, 't0_artifact_path', None):
            env['T0_ARTIFACT_PATH'] = cfg.paths.t0_artifact_path
        if getattr(cfg.paths, 'data_split_dir', None):
            env['DATA_SPLIT_DIR'] = cfg.paths.data_split_dir
    except Exception as _e:
        if not quiet:
            print(f"[TRAIN_RUNNER][WARN] 读取文件化配置失败，回退到环境变量：{_e}", file=sys.stderr)
        experiments_base = env.get('EXPERIMENTS_DIR', os.path.join(root, 'experiments'))
        os.makedirs(experiments_base, exist_ok=True)
        run_name = env.get('RUN_NAME') or time.strftime('exp-%Y%m%d-%H%M%S')
        run_dir = os.path.join(experiments_base, run_name)
        os.makedirs(run_dir, exist_ok=True)
        env.setdefault('RUN_DIR', run_dir)

    # Artifacts manifest & strategy
    try:
        from src.shared.artifacts import build_manifest, snapshot_artifacts, write_manifest
        global_art_dir = env.get('ARTIFACTS_DIR', os.path.join(root, 'data', 'artifacts'))
        strategy = (env.get('ARTIFACTS_STRATEGY') or 'global').lower()
        snapshot_mode = (env.get('ARTIFACTS_SNAPSHOT_MODE') or 'hardlink').lower()
        if strategy == 'snapshot':
            manifest = snapshot_artifacts(global_art_dir, env['RUN_DIR'], mode=snapshot_mode)
            env['ARTIFACTS_DIR'] = os.path.join(env['RUN_DIR'], 'artifacts')
            write_manifest(manifest, env['RUN_DIR'])
            if not quiet:
                print(f"[TRAIN_RUNNER] artifacts strategy=snapshot, mode={manifest.get('snapshot_mode')}, dir={env['ARTIFACTS_DIR']}")
        elif strategy == 'isolated':
            run_art_dir = os.path.join(env['RUN_DIR'], 'artifacts')
            os.makedirs(run_art_dir, exist_ok=True)
            env['ARTIFACTS_DIR'] = run_art_dir
            manifest = build_manifest(run_art_dir, strategy='isolated', snapshot_mode=None)
            write_manifest(manifest, env['RUN_DIR'])
            if not manifest.get('files'):
                print("[TRAIN_RUNNER][ERROR] isolated 策略要求 RUN_DIR/artifacts 下存在产物文件，请先运行预处理或手动放置。", file=sys.stderr)
                print(f"[TRAIN_RUNNER] 期望目录: {run_art_dir}", file=sys.stderr)
                return 2
            if not quiet:
                print(f"[TRAIN_RUNNER] artifacts strategy=isolated, dir={run_art_dir}")
        else:
            manifest = build_manifest(global_art_dir, strategy='global', snapshot_mode=None)
            write_manifest(manifest, env['RUN_DIR'])
            if not quiet:
                print(f"[TRAIN_RUNNER] artifacts strategy=global, dir={global_art_dir}")
    except Exception as e:
        if not quiet:
            print(f"[TRAIN_RUNNER][WARN] Artifacts manifest handling failed: {e}", file=sys.stderr)

    # Pre-run snapshots (config/environment)
    try:
        from src.config.core import Config
        from src.shared.logging.core import write_config_snapshot, write_config_summary, write_env_snapshot, write_effective_config_snapshot
        changed = {}
        for k, v in env.items():
            prev = os.environ.get(k)
            if prev != v:
                changed[k] = prev
                os.environ[k] = v
        cfg = Config(root_dir=root)
        logs_dir = os.path.join(env['RUN_DIR'], 'logs') if not getattr(cfg.paths, 'logs_dir', None) else cfg.paths.logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        # Optional sys monitor
        try:
            from src.shared.sysmon import SystemMonitor
            if getattr(cfg, 'sys_monitor_enabled', False) or os.getenv('SYS_MONITOR', '0') in ('1','true','yes'):
                interval = float(getattr(cfg, 'sys_monitor_interval', 1.0))
                to_console = bool(getattr(cfg, 'sys_monitor_console', True))
                to_csv = bool(getattr(cfg, 'sys_monitor_log_csv', False))
                mon = SystemMonitor(logs_dir, interval=interval, to_console=to_console, to_log_jsonl=True, to_log_csv=to_csv)
                mon.start()
            else:
                mon = None
        except Exception:
            mon = None
        import torch
        runtime = {
            'python_version': sys.version.split()[0],
            'torch_version': getattr(torch, '__version__', ''),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cudnn_enabled': getattr(getattr(torch, 'backends', None), 'cudnn', None) and torch.backends.cudnn.enabled,
        }
        try:
            if torch.cuda.is_available():
                runtime['cuda_device_0'] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        selected_keys = [
            'NUM_WORKERS', 'PIN_MEMORY', 'AMP', 'BATCH_SIZE', 'FORCE_GPU',
            'PERSISTENT_WORKERS', 'PREFETCH_FACTOR', 'DL_V13_FEATURES',
            'WORKSPACE_ROOT', 'RUN_DIR', 'SYS_MONITOR', 'SYS_MONITOR_INTERVAL', 'SYS_MONITOR_CONSOLE', 'SYS_MONITOR_LOG_CSV',
            'OVERFIT_ONE_BATCH', 'OVERFIT_STEPS', 'OFB_BS', 'OFB_BATCH_SIZE', 'USE_SOFT_LABELS', 'SOFT_LABEL_ALPHA', 'KL_T', 'CLASS_W', 'LABEL_SMOOTH', 'OFB_LR',
            'LR_WARMUP_EPOCHS', 'SOFT_ALPHA_START', 'SOFT_ALPHA_TARGET', 'SOFT_ALPHA_RAMP_EPOCHS',
            'LR_SCHEDULER', 'COSINE_TMAX_EPOCHS', 'COSINE_ETA_MIN', 'LR_FACTOR', 'LR_PATIENCE'
        ]
        env_vars = {k: env.get(k) for k in selected_keys if k in env}
        write_config_snapshot(logs_dir, cfg, runtime=runtime, env_vars=env_vars)
        write_config_summary(logs_dir, cfg)
        write_env_snapshot(logs_dir, runtime, env_vars)
        try:
            _eff = write_effective_config_snapshot(env['RUN_DIR'], cfg)
            if not quiet:
                print(f"[TRAIN_RUNNER] 已写入有效配置快照: {_eff}")
        except Exception as _e_eff:
            if not quiet:
                print(f"[TRAIN_RUNNER][WARN] 写入有效配置快照失败: {_e_eff}", file=sys.stderr)
    except Exception as e:
        if not quiet:
            print(f"[TRAIN_RUNNER][WARN] Pre-run snapshot failed: {e}", file=sys.stderr)
    finally:
        try:
            for k, prev in list(changed.items()):
                if prev is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = prev
        except Exception:
            pass

    # Diagnostics probe (lazy)
    probe = None
    cmd = [sys.executable, '-m', 'src.scripts.run_training']

    # Build diagnostics probe from Config
    try:
        from src.config.core import Config
        from src.shared.diagnostics import Probe
        cfg_probe = Config(root_dir=root)
        logs_dir_probe = getattr(cfg_probe.paths, 'logs_dir', os.path.join(env.get('RUN_DIR', os.path.join(root, 'experiments', 'exp')), 'logs'))
        os.makedirs(logs_dir_probe, exist_ok=True)
        diag = getattr(cfg_probe, 'diagnostics', None)
        diag_enabled = bool(getattr(diag, 'enabled', False))
        log_jsonl = bool(getattr(diag, 'log_jsonl', True))
        log_csv = bool(getattr(diag, 'log_csv', False))
        events_filename = str(getattr(diag, 'events_filename', 'diagnostics_events.jsonl'))
        if diag_enabled:
            probe = Probe(enabled=True, logs_dir=logs_dir_probe, log_jsonl=log_jsonl, log_csv=log_csv, events_filename=events_filename)
            probe.start('runner.env.start', meta={'run_dir': env.get('RUN_DIR'), 'artifacts_strategy': env.get('ARTIFACTS_STRATEGY')})
    except Exception:
        pass

    if not quiet:
        print(f"[TRAIN_RUNNER] Launch: {' '.join(cmd)}")
        print(f"[TRAIN_RUNNER] Env overrides: {env_overrides}")
        print(f"[TRAIN_RUNNER] RUN_DIR: {env.get('RUN_DIR')}")
        try:
            print(f"[TRAIN_RUNNER] PYTHONPYCACHEPREFIX: {env.get('PYTHONPYCACHEPREFIX')}")
        except Exception:
            pass

    try:
        if not log_path:
            try:
                from src.config.core import Config
                cfg = Config(root_dir=root)
                log_path = os.path.join(getattr(cfg.paths, 'logs_dir', os.path.join(env['RUN_DIR'], 'logs')), 'train_console.log')
            except Exception:
                log_path = os.path.join(env['RUN_DIR'], 'logs', 'train_console.log')
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            env['PYTHONIOENCODING'] = 'utf-8'
            env['TEE_BY_RUNNER'] = '1'
            use_pty = True
            try:
                import pty
            except Exception:
                use_pty = False
            with open(log_path, 'a', encoding='utf-8-sig') as lf:
                lf.write("[TRAIN_RUNNER] === start ===\n")
                if use_pty:
                    import select
                    master_fd, slave_fd = pty.openpty()
                    try:
                        try:
                            if probe:
                                probe.log('runner.subprocess.start', mode='pty', log_path=str(log_path))
                        except Exception:
                            pass
                        _subproc_t0 = time.time()
                        proc = subprocess.Popen(
                            cmd,
                            stdout=slave_fd,
                            stderr=slave_fd,
                            env=env,
                            bufsize=0,
                            close_fds=False,
                        )
                        try:
                            os.close(slave_fd)
                        except Exception:
                            pass
                        while True:
                            try:
                                r, _, _ = select.select([master_fd], [], [], 0.1)
                                if master_fd in r:
                                    try:
                                        data = os.read(master_fd, 4096)
                                    except OSError:
                                        data = b''
                                    if not data:
                                        if proc.poll() is not None:
                                            break
                                        continue
                                    try:
                                        text = data.decode('utf-8', errors='replace')
                                    except Exception:
                                        text = data.decode('latin1', errors='replace')
                                    try:
                                        sys.stdout.write(text)
                                        sys.stdout.flush()
                                    except Exception:
                                        pass
                                    try:
                                        lf.write(text)
                                        lf.flush()
                                    except Exception:
                                        pass
                            except Exception:
                                if proc.poll() is not None:
                                    break
                        proc.wait()
                        try:
                            if probe:
                                probe.log('runner.subprocess.exit', returncode=int(proc.returncode), duration_s=float(time.time() - _subproc_t0))
                        except Exception:
                            pass
                    finally:
                        try:
                            os.close(master_fd)
                        except Exception:
                            pass
                else:
                    try:
                        if probe:
                            probe.log('runner.subprocess.start', mode='pipe', log_path=str(log_path))
                    except Exception:
                        pass
                    _subproc_t0 = time.time()
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                        bufsize=1,
                        text=True,
                    )
                    try:
                        assert proc.stdout is not None
                        for line in proc.stdout:
                            try:
                                print(line, end='')
                            except Exception:
                                pass
                            try:
                                lf.write(line)
                            except Exception:
                                pass
                    finally:
                        proc.wait()
                        try:
                            if probe:
                                probe.log('runner.subprocess.exit', returncode=int(proc.returncode), duration_s=float(time.time() - _subproc_t0))
                        except Exception:
                            pass
        else:
            try:
                from src.shared.diagnostics import Probe
                if probe:
                    probe.log('runner.subprocess.start', mode='inherit')
            except Exception:
                pass
            _subproc_t0 = time.time()
            proc = subprocess.Popen(cmd, env=env)
            proc.wait()
            try:
                if probe:
                    probe.log('runner.subprocess.exit', returncode=int(proc.returncode), duration_s=float(time.time() - _subproc_t0))
            except Exception:
                pass
    except Exception as e:
        if not quiet:
            print(f"[TRAIN_RUNNER][ERROR] Failed to launch train.py: {e}", file=sys.stderr)
        try:
            from src.shared.diagnostics import Probe
            if probe:
                probe.log('runner.subprocess.error', message=str(e))
                probe.finalize({'status': 'error'})
        except Exception:
            pass
        return 1

    rc = proc.returncode
    try:
        from src.shared.diagnostics import Probe
        if probe:
            probe.finalize({'status': 'ok', 'returncode': int(rc)})
    except Exception:
        pass
    return rc
