from queue import Queue
from PySide6 import QtCore

import keyboard

class ArduinoWorker2(QtCore.QObject):
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.arm_move = 0
        self.stop = False

        self.result_count = 0
        self.message_queue = Queue()       # cameraからのブザーやスピード情報

    def run(self):
        count_close = 0
        count_open = 0
        count_right = 0
        count_left = 0
        count_mortor = 0
        count_box = 0
        self.serialed_result = 0
        open = 120
        grip = "h"
        RL = "D"
        print("arduino")
        while not self.stop:
            key, message_data = self.message_queue.get(block=True)     # デフォがblock=True

            if key == "emg":
                print(message_data)

                if message_data == '4':
                    message_data = '2'
                    
                if message_data == '1' :  #closed
                    count_close += 1
                    if self.arm_move != 1 and count_close > 1:
                        self.serialed_result = 180
                        self.arm_move = 1
                        count_open = 0
                        count_left = 0
                        count_right = 0
                elif message_data == '2' :
                    count_open += 1
                    if self.arm_move != 2 and count_open > 1:
                        self.serialed_result = 140
                        self.arm_move = 2
                        count_close = 0
                        count_right = 0
                        count_left = 0
                elif message_data == '3':
                    count_left += 1
                    count_right = 0
                    if count_left == 3:
                        count_left = 0
                        self.serialed_result = -10
                elif message_data == '4':
                    count_right += 1
                    count_left = 0
                    if count_right == 4:
                        count_right = 0
                        self.serialed_result = 10
                else:
                    self.serialed_result = 0
                if self.serialed_result > 100:
                    message = f"{grip}{self.serialed_result}\n".encode()
                    self.ser.write(message)
                elif self.serialed_result != 0:
                    message = f"{RL}{self.serialed_result}\n".encode()
                    self.ser.write(message)
                    # pass

            elif key == "camera":
                # print(message_data)
                # if self.arm_move == 2 and count_box == 0:
                #     count_box = 1
                #     set_arm = 'S'
                #     set_move = 0
                #     message_data = f"{set_arm}{set_move}\n".encode()
                #     self.ser.write(message_data)
                # elif self.arm_move != 2:
                #     count_box = 0
                #     count_mortor += 1
                #     if count_mortor == 3:
                #         count_mortor = 0
                #     self.ser.write(message_data)
                #     print(self.ser.readline().decode().strip()) 
                self.ser.write(message_data)
                # print(self.ser.readline().decode().strip())
                
            if keyboard.is_pressed("o"):        #open
                message = f"{grip}{open}\n".encode()
                self.ser.write(message)
                print(self.ser.readline().decode().strip()) 
                print(message)
            if keyboard.is_pressed("c"):        #closed_hand
                message = f"{grip}{180}\n".encode()
                self.ser.write(message)
                     

    def stop_run(self):
        print("arduino worker stop__run")
        self.stop = True

if __name__ == "__main__":
    worker = ArduinoWorker2()