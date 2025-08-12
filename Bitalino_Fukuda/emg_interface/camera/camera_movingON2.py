import json
import time
from pathlib import Path
from queue import Queue

import cv2
import depthai as dai
import numpy as np

import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui

import sys
import keyboard  # 確認用キーボード入力検知
import serial


class MovingON2(QtWidgets.QWidget):
    cameraon = QtCore.Signal(object)

    def __init__(
        self,
        config_path = Path("emg_interface/camera/result/BS_640/epoch100-9000data/best.json"),
        model_path = Path("emg_interface/camera/result/BS_640/epoch100-9000data/best.blob"),
        # config_path=Path("emg_interface/camera/result/SL_640/data5560_epoch100-batch16-imgsz640/best.json"),
        # model_path=Path("emg_interface/camera/result/SL_640/data5560_epoch100-batch16-imgsz640/best.blob"),
    ):
        super().__init__()
        self.setWindowTitle("Camera plot")
        self.stop = False

        self.config_path = config_path
        self.model_path = model_path

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.foreground_color = self.palette().color(self.foregroundRole())

        pg.setConfigOptions(
            background=None,
            foreground=self.foreground_color,
            antialias=True,
        )

        self.load_config()
        self.create_pipeline()

        # 画像表示用ラベル
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # ラベルの内容を中央に配置
        self.main_layout.addWidget(self.image_label)

        self.count_buzzer = 0
        self.keyboard_print = 0
        self.targetobject_list = []
        self.object_list = []
        self.find_time = 0
        self.notfind_time = 0
        self.target = False
        self.diff_xnum = 0
        self.diff_ynum = 0
        self.not_screwCount = 0
        self.diff_x = 0
        self.past_diffx = 0
        self.count_diff = 0

        self.find_beep = 0
        self.buzzer = 0
        self.beep_xnum = 0
        self.beep_ynum = 0
        self.beep_on = 0
        self.beep_off = 0
        self.beep_off_reset = 0

        # self.result_queue = Queue()
        self.count_opened = 0
        self.count_closed = 0
        self.serialed_hand = 0
        self.arm_move = 0

        self.set_Gain(0.1)    # ad .42
        # self.set_gain(1)      # ad
        self.set_Pid(100)        # ad
        self.set_pId(150)        # ad
        # self.set_Gain(0.37)   # 追跡ON  ad=0.015,sp=0.33
        # self.set_gain(0.01)   # 追跡ON
        # self.set_Pos(180)     # ad
        # self.set_pos(230)     # ad
        # self.set_P()        

    def set_buzzer(self, buzz):  # 1のとき鳴る、0のとき無音
        self.send_command_to_arduino("b", buzz)
        # print("buzzer")
        # pass

    def set_Gain(self, gain):
        self.send_command_to_arduino("G", gain)     #id1

    def set_gain(self, gain):
        self.send_command_to_arduino("g", gain)     

    def set_vel_x(self, vel):
        self.send_command_to_arduino("S", vel)

    def set_vel_y(self, vel):
        self.send_command_to_arduino("s", vel)

    def set_Pos(self, pos):
        self.send_command_to_arduino("A", pos)

    def set_pos(self, pos):
        self.send_command_to_arduino("a", pos)

    def set_Move(self, diff):
        self.send_command_to_arduino("M", diff)

    def set_move(self, diff):
        self.send_command_to_arduino("m", diff)

    def set_Pid(self, Pid):
        self.send_command_to_arduino("P", Pid)

    def set_pId(self, Pid):
        self.send_command_to_arduino("I", Pid)


    def send_command_to_arduino(self, code, val):
        message = f"{code}{val}\n".encode()
        self.cameraon.emit(message)
        # print(message)
        # self.ser.write(message)
        # print(self.ser.readline().decode().strip())

    def load_config(self):
        with open(self.config_path) as f:
            config = json.load(f)
        self.nnConfig = config.get("nn_config", {})

        # parse input shape
        self.W, self.H = tuple(map(int, self.nnConfig.get("input_size").split("x")))

        # extract metadata
        metadata = self.nnConfig.get("NN_specific_metadata", {})
        self.classes = metadata.get("classes", {})
        self.coordinates = metadata.get("coordinates", {})
        self.anchors = metadata.get("anchors", {})
        self.anchorMasks = metadata.get("anchor_masks", {})
        self.iouThreshold = metadata.get("iou_threshold", {})
        self.confidenceThreshold = metadata.get("confidence_threshold", {})

        # parse labels
        nnMappings = config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        # sync outputs
        self.syncNN = True

    def create_pipeline(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.detectionNetwork = self.pipeline.create(dai.node.YoloDetectionNetwork)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.nnOut = self.pipeline.create(dai.node.XLinkOut)

        self.xoutRgb.setStreamName("rgb")
        self.nnOut.setStreamName("nn")

        # Properties
        self.camRgb.setPreviewSize(self.W, self.H)  # jsonの中身を416から640に変更
        print(self.W, " ", self.H)
        self.center_x = self.W / 2
        self.center_y = 500

        self.camRgb.setInterleaved(False)
        self.camRgb.setImageOrientation(
            dai.CameraImageOrientation.ROTATE_180_DEG
        )  # 上下左右反転
        self.camRgb.initialControl.setManualFocus(165)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(30)

        # Network specific settings
        self.detectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        self.detectionNetwork.setNumClasses(self.classes)
        self.detectionNetwork.setCoordinateSize(self.coordinates)
        self.detectionNetwork.setAnchors(self.anchors)
        self.detectionNetwork.setAnchorMasks(self.anchorMasks)
        self.detectionNetwork.setIouThreshold(self.iouThreshold)
        self.detectionNetwork.setBlobPath(self.model_path)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.input.setBlocking(False)

        # Linking
        self.camRgb.preview.link(self.detectionNetwork.input)
        self.detectionNetwork.passthrough.link(self.xoutRgb.input)
        self.detectionNetwork.out.link(self.nnOut.input)

    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(self, name, frame, detections):
        color = (255, 0, 0)

        # objects = {"Short_screw": [], "Long_screw": []}
        objects = {"Big_screw": [], "Small_screw": []}

        for detection in detections:
            curr_time = time.monotonic()
            object_X = "N"

            # バウンディングボックスの座標
            bbox = self.frameNorm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
            center_bbox_X = (detection.xmax + detection.xmin) / 2 * self.W
            center_bbox_Y = (detection.ymax + detection.ymin) / 2 * self.H

            objects[self.labels[detection.label]].append(
                np.array([center_bbox_X, center_bbox_Y])
            )

            cv2.putText(
                frame,
                self.labels[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (255, 255, 0),
            )
            # confidenceは検出の信頼度
            cv2.putText(
                frame,
                f"{int(detection.confidence * 100)}%",
                (bbox[0] + 10, bbox[1] + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (255, 255, 0),
            )

        if len(objects["Big_screw"]) > 0:

            self.beep_off_reset = 0
            self.not_screwCount = 0

            distances = np.zeros(len(objects["Big_screw"]))
            for i, screw_bbox in enumerate(objects["Big_screw"]):
                distance_to_centerX = (np.abs(self.center_x - screw_bbox[0]) /10) ** 2
                if screw_bbox[1] > 620:
                    distance_to_centerY = (np.abs(self.center_y - screw_bbox[1] + 100) /10) ** 2 * 4
                else:
                    distance_to_centerY = (np.abs(self.center_y - screw_bbox[1]) /10) ** 2 * 4
                distance_to_center = (distance_to_centerX/10) + (distance_to_centerY/10)
                distances[i] = distance_to_center

            # 选择最近的目标
            nearest = np.argmin(distances)
            nearest_object = objects["Big_screw"][nearest]
            cv2.circle(
                frame,
                (int(nearest_object[0]), int(nearest_object[1])),
                50,
                (255, 255, 255),
                thickness=3,
            )

            self.diff_x = self.center_x - nearest_object[0]  # 中央との差"X"
            diff_y = self.center_y - nearest_object[1]  # 中央との差"Y"

            distance_diffX = (detection.xmax - detection.xmin)

            if -100 <= self.diff_x <= 100:
                # diff_x = diff_x * 1.2
                if self.beep_xnum == 0:
                    self.beep_xnum = 1
            else:
                if self.beep_xnum == 1:
                    self.beep_xnum = 0
            if -50 <= diff_y <= 150:
                if self.beep_ynum == 0:
                    self.beep_ynum = 1
            else:
                if self.beep_ynum == 1:
                    self.beep_ynum = 0

            if self.beep_xnum == 1 and self.beep_ynum == 1:
                self.beep_on += 1
                if self.beep_on == 1:
                    self.beep_off = 0
                    self.set_buzzer(1)            
            else:
                self.beep_off += 1
                if self.beep_off == 1:
                    self.beep_on = 0
                    self.set_buzzer(0)


            # ad　ビジョンセンサあり　被験者用
            if abs(self.past_diffx - self.diff_x) <= 20:
                if abs(self.diff_x) < 30:
                    self.diff_x = self.diff_x * 0.04
                elif abs(self.diff_x) < 100:
                    self.diff_x = self.diff_x * 0.06
                elif abs(self.diff_x) < 140:
                    self.diff_x = self.diff_x * 0.09
                elif abs(self.diff_x) < 230:
                    self.diff_x = self.diff_x * 0.11
                if abs(self.diff_x) > 40:
                    if self.diff_x > 0:
                        self.diff_x = 40
                    else:
                        self.diff_x = -40
                self.set_Move(self.diff_x)
                print(self.diff_x)
                self.past_diffx = self.diff_x
            else:
                self.past_diffx = self.diff_x


        else:
            self.beep_off_reset += 1
            if self.beep_off_reset == 1:
                self.set_buzzer(0)
                # pass

            self.not_screwCount += 1
            if self.not_screwCount == 10:
                self.set_Pos(180)


        if keyboard.is_pressed("e"):  # end
            # self.set_vel(0)
            # self.ser.close()
            self.set_buzzer(0)
            self.set_vel_x(0)
            self.set_vel_y(0)
            print("終了")
            sys.exit()

        if keyboard.is_pressed("p"):
            self.set_Pos(180)     # ad


        # Show the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGRからRGBに色空間を変換
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width  # 画像の1行辺りのバイト数計算
        q_image = QtGui.QImage(
            frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        image_pixmap = QtGui.QPixmap.fromImage(q_image)
        # self.image_label.setPixmap(image_pixmap)
        # self.image_label.adjustSize()

        cv2.imshow(name, frame)  # rgbでの映像ウィジェット

    def run(self):
        try:
            with dai.Device(self.pipeline) as device:
                # Output queues will be used to get the rgb frames and nn data from the outputs defined above
                qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

                frame = None
                detections = []
                startTime = time.monotonic()
                counter = 0
                color2 = (255, 255, 255)

                while True:
                    inRgb = qRgb.get()
                    inDet = qDet.get()

                    if inRgb is not None:
                        frame = inRgb.getCvFrame()
                        cv2.putText(
                            frame,
                            "NN fps: {:.2f}".format(
                                counter / (time.monotonic() - startTime)
                            ),
                            (2, frame.shape[0] - 4),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.4,
                            color2,
                        )

                    if inDet is not None:
                        detections = inDet.detections
                        counter += 1

                    if frame is not None:
                        self.displayFrame(
                            "rgb", frame, detections
                        )  # 画像にバウンディングボックストラベルの描画
                        # self.displayFrame(frame, detections)

                    if cv2.waitKey(1) == ord("q"):
                        break

        except Exception as e:
            # self.camera_status_label.setText("カメラが接続されていません")
            print("デバイスなし")
            import traceback

            traceback.print_exc()

    def cam_stop(self):
        self.stop = True
                

if __name__ == "__main__":

    app = QtWidgets.QApplication([])

    yolo_depth_ai = MovingON2()
    yolo_depth_ai.run()  # ここだけで動かすとき必要

    yolo_depth_ai.show()
    app.exec()
