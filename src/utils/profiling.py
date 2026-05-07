
import time
import torch
import logging

class ThroughputMonitor:
    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.reset()

    def reset(self):
        self.data_time_sum = 0.0
        self.gpu_time_sum = 0.0
        self.breakdown_sums = {'fwd': 0.0, 'loss': 0.0, 'bwd': 0.0}
        self.batch_count = 0
        self.start_time = time.time()
        self.last_report_time = time.time()

    def update(self, data_time, gpu_time, breakdown=None):
        self.data_time_sum += data_time
        self.gpu_time_sum += gpu_time
        if breakdown:
            for k, v in breakdown.items():
                if k in self.breakdown_sums:
                    self.breakdown_sums[k] += v
        self.batch_count += 1
        
    def report(self, step):
        if self.batch_count == 0:
            return None

        current_time = time.time()
        elapsed = current_time - self.last_report_time
        
        avg_data = self.data_time_sum / self.batch_count
        avg_gpu = self.gpu_time_sum / self.batch_count
        throughput = self.batch_count / elapsed # it/s
        
        # Breakdown stats
        avg_breakdown = {k: (v / self.batch_count) * 1000 for k, v in self.breakdown_sums.items()}
        
        # Reset counters for next interval
        self.data_time_sum = 0.0
        self.gpu_time_sum = 0.0
        self.breakdown_sums = {'fwd': 0.0, 'loss': 0.0, 'bwd': 0.0}
        self.batch_count = 0
        self.last_report_time = current_time

        # Get GPU stats
        try:
            gpu_util = torch.cuda.utilization()
            vram_used = torch.cuda.memory_allocated() / 1e9 # GB
        except:
            gpu_util = 0
            vram_used = 0

        # Diagnosis
        diagnosis = "OK"
        if avg_data > avg_gpu * 2:
            diagnosis = "CRITICAL: CPU Bottleneck (Data Wait >> Compute)"
        elif avg_gpu > avg_data * 5:
            diagnosis = "GPU Bound (Good)"
        elif avg_gpu < 0.001:
            diagnosis = "Suspiciously Fast Compute (Check Logic)"

        self.logger.info(
            f"[Step {step}] Throughput: {throughput:.2f} it/s | "
            f"Data: {avg_data*1000:.1f}ms | Compute: {avg_gpu*1000:.1f}ms "
            f"(Fwd: {avg_breakdown['fwd']:.1f}ms, Loss: {avg_breakdown['loss']:.1f}ms, Bwd: {avg_breakdown['bwd']:.1f}ms) | "
            f"GPU Util: {gpu_util}% | VRAM: {vram_used:.1f}GB | "
            f"Diagnosis: {diagnosis}"
        )
        
        return {
            'data_time': avg_data,
            'gpu_time': avg_gpu,
            'throughput': throughput,
            'gpu_util': gpu_util
        }

class Stopwatch:
    def __init__(self):
        self.start = 0.0
        
    def tic(self):
        self.start = time.time()
        
    def toc(self):
        return time.time() - self.start
