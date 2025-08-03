import serial
import time

class ObjectTracker:
    def __init__(self, serial_port='/dev/ttyACM0', baudrate=9600, image_width=640):
        self.ser = serial.Serial(serial_port, baudrate, timeout=1)
        self.image_center = image_width // 2
        self.center_tolerance = 20  # 中心区间容忍度
        self.angle_step = 5         # 每次旋转角度

    def track(self, object_x):
        """
        根据物体中心X坐标，判断需要左转还是右转
        """
        offset = object_x - self.image_center

        if abs(offset) < self.center_tolerance:
            return  # 已居中，无需旋转

        direction = -1 if offset < 0 else 1
        angle_cmd = f"ROTATE:{direction * self.angle_step}"
        self.send_command(angle_cmd)

    def open_gripper(self):
        self.send_command("GRIP:OPEN")

    def close_gripper(self):
        self.send_command("GRIP:CLOSE")

    def send_command(self, command):
        print(f"[SEND] {command}")
        self.ser.write((command + '\n').encode())
        time.sleep(0.1)

# 用法示例
if __name__ == "__main__":
    tracker = ObjectTracker()

    # 模拟目标坐标不断变化
    while True:
        fake_x = int(input("Enter target X coordinate (0-640): "))
        tracker.track(fake_x)
