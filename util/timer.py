import time
# 简洁的高精度计时器
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
        print(f"耗时: {self.duration:.6f} 秒")