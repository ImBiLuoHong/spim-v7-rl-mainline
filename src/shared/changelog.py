import os
import time
import re
from typing import Optional


def _now_str() -> str:
    """返回本地时区的人类可读时间戳字符串。"""
    # 使用系统本地时区名称，符合变更记录格式要求
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def _ensure_parent(path: str) -> None:
    """确保父目录存在。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        # 父目录不可创建时静默忽略，由调用方处理回退
        pass


def _compose_line(timestamp: str,
                  purpose: str,
                  location: str,
                  change: str,
                  command: Optional[str] = None,
                  commit: Optional[str] = None,
                  status: Optional[str] = None) -> str:
    """按规范拼接单行变更记录。"""
    # 规范行示例：
    # - [YYYY-MM-DD HH:mm:ss Local Timezone] Purpose:<≤50 chars>; Location:<relative path>; Change:<key function/config key>; Command:<or N/A>; Commit:<git short hash or N/A>
    cmd = command if (command and command.strip()) else 'N/A'
    cmt = commit if (commit and commit.strip()) else 'N/A'
    prefix = f"- [{timestamp}]"
    if status:
        prefix = f"{prefix} [{status}]"
    return f"{prefix} Purpose:{purpose}; Location:{location}; Change:{change}; Command:{cmd}; Commit:{cmt}\n"


def ensure_repo_changelog_append(cfg, line: str) -> bool:
    """尝试将一行写入仓库内 docs/开发变更记录.md。

    返回 True 表示写入成功；False 表示写入失败（调用方可做回退）。
    """
    try:
        repo_md = getattr(cfg.paths, 'repo_changelog_path', None)
        if not repo_md:
            # 默认路径：<root>/docs/开发变更记录.md
            repo_md = os.path.join(cfg.paths.root_dir, 'docs', '开发变更记录.md')
        _ensure_parent(repo_md)
        with open(repo_md, 'a', encoding='utf-8') as f:
            f.write(line)
        return True
    except Exception:
        return False


def write_dual_changelog_line(cfg,
                              purpose: str,
                              location: str,
                              change: str,
                              command: Optional[str] = None,
                              commit: Optional[str] = None,
                              status: Optional[str] = None) -> str:
    """双写变更记录：
    1) 仓库 docs/开发变更记录.md（始终尝试写入）；
    2) Windows 路径（cfg.paths.windows_changelog_path，若不可达则回退到 RUN_DIR/logs/CHANGELOG_PENDING.txt）。

    - 纯配置驱动：不读取环境变量，也不依赖全局状态；
    - 返回最终写入的单行字符串便于审计；
    - 如 Windows 路径不可达，打印 [SYNC] 提示。
    """
    ts = _now_str()
    line = _compose_line(ts, purpose, location, change, command, commit, status)

    # 写入仓库 changelog（主记录）
    repo_ok = ensure_repo_changelog_append(cfg, line)

    # 写入 Windows 路径或回退
    win_path = getattr(cfg.paths, 'windows_changelog_path', None)
    wrote_windows = False
    if win_path:
        # 在非 Windows 系统上，如果路径形如 "X:\\..." 或 "X:/..."，认为是 Windows 盘符路径，不尝试写入，直接回退
        is_win_drive_path = bool(re.match(r'^[A-Za-z]:[\\/]', win_path))
        if os.name != 'nt' and is_win_drive_path:
            wrote_windows = False
        else:
            try:
                _ensure_parent(win_path)
                with open(win_path, 'a', encoding='utf-8') as f:
                    f.write(line)
                wrote_windows = True
            except Exception:
                wrote_windows = False

    if not wrote_windows:
        # 回退到 RUN_DIR/logs/CHANGELOG_PENDING.txt
        pending = os.path.join(cfg.paths.run_dir, 'logs', 'CHANGELOG_PENDING.txt')
        try:
            _ensure_parent(pending)
            with open(pending, 'a', encoding='utf-8') as f:
                f.write(line)
            print(f"[SYNC] Windows 变更记录不可达，已写入待同步文件: {pending}")
        except Exception:
            # 即便回退失败也要提示，调用方可自行处理
            print("[SYNC][ERROR] 写入待同步文件失败，可能磁盘不可写。")

    if not repo_ok:
        print("[SYNC][WARN] 仓库变更记录写入失败，已尝试 Windows/待同步文件回退。")

    return line