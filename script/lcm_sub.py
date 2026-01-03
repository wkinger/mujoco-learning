import lcm
import threading
import time
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/lcm_msg")
from exlcm.example_t import example_t
class LCMSub:
    def __init__(self):
        """ Step1: LCM通信实例化 """
        self.lcm = lcm.LCM()
        """ Step2: LCM接收的处理 """
        # LCM接收的句柄
        self.lcm.subscribe("EXAMPLE", self._handle_exmpale)
        # 线程锁，防止消息处理过程被修改
        self.sub_mutex = threading.Lock()
        # 用于判断是否是新的消息
        self.last_timestamp = 0
        self.new_msg = False
        # 接收到的信息
        self.sub_msg = None
        """ Step3: handle线程 """
        # 用于存储线程对象
        self.lcm_thread = None 
        # 控制线程是否运行
        self.running = False  

    def _lcm_thread_func(self):
        """LCM 消息处理线程"""
        while self.running:
            self.lcm.handle()  # 处理 LCM 消息

    def start_lcm_thread(self):
        """启动 LCM 线程"""
        if not self.running:
            self.running = True
            self.lcm_thread = threading.Thread(target=self._lcm_thread_func, daemon=True)
            self.lcm_thread.start()

    def stop_lcm_thread(self):
        """停止 LCM 线程"""
        if self.running:
            self.running = False
            if self.lcm_thread is not None:
                self.lcm_thread.join()  # 等待线程结束
            self.lcm_thread = None    

    def _handle_exmpale(self, channel, data):
        """ 处理接收到的EXAMPLE 信息"""
        msg = example_t.decode(data)
        with self.sub_mutex:
            if msg.timestamp > self.last_timestamp:
                self.sub_msg = msg
                self.last_timestamp = msg.timestamp
                self.new_msg = True
        
    def main(self):
        while True:
            if self.new_msg:
                print("-------------------------")
                print("print new messages: ")
                print(self.sub_msg.position)
                print(self.last_timestamp)
            time.sleep(1)

if __name__ == "__main__":
    lcm_sub = LCMSub()
    lcm_sub.start_lcm_thread()
    lcm_sub.main()
