import os
import sys


def setup_dual_console_logging(logs_dir: str):
    """Tee stdout/stderr to logs_dir/train_console.log while preserving TTY behavior.
    Does not read environment variables; caller decides whether to enable.
    Returns the file handle or None on failure.
    """
    try:
        if not logs_dir:
            return None
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, 'train_console.log')

        class _Tee:
            def __init__(self, console, file):
                self.console = console
                self.file = file
                try:
                    self.encoding = getattr(console, 'encoding', 'utf-8')
                except Exception:
                    self.encoding = 'utf-8'
            def write(self, data):
                try:
                    self.console.write(data)
                except Exception:
                    pass
                try:
                    self.file.write(data)
                except Exception:
                    pass
            def flush(self):
                try:
                    self.console.flush()
                except Exception:
                    pass
                try:
                    self.file.flush()
                except Exception:
                    pass
            def isatty(self):
                try:
                    return bool(getattr(self.console, 'isatty', lambda: False)())
                except Exception:
                    return False
            def fileno(self):
                try:
                    return getattr(self.console, 'fileno')()
                except Exception:
                    raise OSError("fileno not available")
            def __getattr__(self, name):
                try:
                    return getattr(self.console, name)
                except Exception:
                    raise AttributeError(name)

        f = open(log_path, 'a', encoding='utf-8-sig')
        sys.stdout = _Tee(sys.stdout, f)
        sys.stderr = _Tee(sys.stderr, f)
        return f
    except Exception:
        return None

