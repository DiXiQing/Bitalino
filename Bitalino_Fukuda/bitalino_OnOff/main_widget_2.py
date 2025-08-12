from PySide6 import QtWidgets, QtCore

from emg_interface.defs import SAMPLING_RATE, NUM_CHANNELS, CHUNK_SIZE, settings_file
from emg_interface.display_widget import LivePlotWidget
from emg_interface.docks import (
    ConnectDock,
    ArduinoDock,
    RecordDock,
    ProcessingDock,
    RecognitionDock,
)
from emg_interface.workers import RecognitionWorker, StreamWorker
from emg_interface.camera import MovingON2
from bitalino_OnOff.docks import CameraConnect
from bitalino_OnOff.worker import ArduinoWorker2
# from bitalino_OnOff.plot_widget import CameraPlotWidget
from pathlib import Path

class MainWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bitalino EMG")

        self.main_Widget = QtWidgets.QWidget(self)
        self.main_layout = QtWidgets.QHBoxLayout(self.main_Widget)
        self.setCentralWidget(self.main_Widget) 

        self.main_splitter = QtWidgets.QSplitter(self)
        self.main_layout.addWidget(self.main_splitter)

        self.live_plot_widget = LivePlotWidget(NUM_CHANNELS)
        self.main_splitter.addWidget(self.live_plot_widget)

        self.connect_dock = ConnectDock()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.connect_dock)
        self.connect_dock.connect_clicked.connect(self.connect_bitalino)

        self.connect_dockCamera = CameraConnect()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.connect_dockCamera)
        self.connect_dockCamera.button_click.connect(self.connect_camera)

        self.processing_dock = ProcessingDock(num_channels=NUM_CHANNELS)
        self.processing_dock.processing_updated.connect(self.update_processing)
        self.processing_dock.start_zero_offset_calibration.connect(
            self.start_zero_offset_calibration
        )
        self.processing_dock.finish_zero_offset_calibration.connect(
            self.finish_zero_offset_calibration
        )
        self.processing_dock.zero_offset_changed.connect(self.zero_offset_changed)
        self.processing_dock.start_max_value_calibration.connect(
            self.start_max_value_calibration
        )
        self.processing_dock.finish_max_value_calibration.connect(
            self.finish_max_value_calibration
        )
        self.processing_dock.zero_offset_changed.connect(self.zero_offset_changed)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.processing_dock)


        self.arduino_dock = ArduinoDock()
        self.arduino_dock.device_found.connect(self.connect_arduino)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.arduino_dock)
        self.lines = []

        self.recognition_dock = RecognitionDock()
        self.recognition_dock.model_updated.connect(self.update_model)
        self.recognition_dock.identity_updated.connect(self.update_identity)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.recognition_dock)

        self.record_dock = RecordDock()
        self.record_dock.start_record.connect(self.start_record)
        self.record_dock.end_record.connect(self.end_record)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.record_dock)

        # thread for streaming input
        self.stream_thread = QtCore.QThread()
        self.stream_worker = None

        # thread for recognition
        self.recognition_thread = QtCore.QThread()
        self.recognition_worker = RecognitionWorker()
        self.recognition_worker.moveToThread(self.recognition_thread)
        self.recognition_worker.serialed.connect(self.movement_serialed)    
        self.recognition_thread.started.connect(self.recognition_worker.run)
        self.recognition_thread.start()

        self.arduino_thread = QtCore.QThread()
        self.arduino_worker = None

        self.plot_update_timer = QtCore.QTimer()
        self.plot_update_timer.timeout.connect(self.update_plots)
        self.plot_update_timer.start(30)

        self.camera_thread = QtCore.QThread()
        self.camera_yolo = None
        self.image_label = QtWidgets.QLabel(self)

        self.settings_file = settings_file()
        if self.settings_file.is_file():
            settings = QtCore.QSettings(
                str(self.settings_file), QtCore.QSettings.IniFormat
            )
            self.gui_restore(settings)

    def connect_bitalino(self, pc_address):
        if self.stream_worker is not None:
            self.stream_worker.set_stop()

        self.stream_worker = StreamWorker(
            pc_address,
            window_length_seconds=5,
            sampling_rate=SAMPLING_RATE,
            num_channels=NUM_CHANNELS,
            chunk_size=CHUNK_SIZE,
        )
        self.stream_worker.moveToThread(self.stream_thread)
        self.stream_worker.finished.connect(self.stream_finished)
        self.stream_worker.stream_error.connect(self.handle_stream_error)
        self.stream_worker.downsampled_data.connect(self.gesture_recognition)
        self.stream_thread.started.connect(self.stream_worker.stream)
        self.stream_thread.start()

    def connect_arduino(self, ser):
        if self.arduino_worker is not None:
            self.arduino_worker.stop_run()
            print("stop_run")

        self.arduino_worker = ArduinoWorker2(ser)
        self.arduino_worker.moveToThread(self.arduino_thread)
        self.arduino_thread.started.connect(self.arduino_worker.run)
        self.arduino_thread.start()

    def connect_camera(self):
        if self.camera_yolo is not None:
            self.camera_yolo.cam_stop()

        self.camera_yolo = MovingON2(
            config_path = Path("emg_interface/camera/result/BS_640/epoch100-9000data/best.json"),
            model_path = Path("emg_interface/camera/result/BS_640/epoch100-9000data/best.blob"),
        )
        self.camera_yolo.moveToThread(self.camera_thread)
        self.camera_yolo.cameraon.connect(self.movement_speed)
        self.camera_thread.started.connect(self.camera_yolo.run)
        self.camera_thread.start()

    def movement_speed(self, speed):
        if self.arduino_worker is not None:
            self.arduino_worker.message_queue.put(("camera", speed))

    def stream_finished(self):
        self.stream_thread.exit()
        self.stream_worker = None

    def update_processing(self, new_processing_flags):
        if self.stream_worker is not None:
            self.stream_worker.update_processing(new_processing_flags)

    def start_zero_offset_calibration(self):
        if self.stream_worker is not None:
            self.stream_worker.start_collect_data()

    def finish_zero_offset_calibration(self):
        if self.stream_worker is not None:
            self.stream_worker.collection_complete.connect(
                self.zero_offset_collection_complete
            )
            self.stream_worker.finish_collect_data()
    
    def zero_offset_collection_complete(self, collected_data):
        self.stream_worker.collection_complete.disconnect(
            self.zero_offset_collection_complete
        )
        self.processing_dock.calc_zero_offsets(collected_data)

    def zero_offset_changed(self, offsets):
        if self.stream_worker is not None:
            self.stream_worker.set_zero_offsets(offsets)

    def start_max_value_calibration(self):
        if self.stream_worker is not None:
            self.stream_worker.start_collect_data()

    def finish_max_value_calibration(self):
        if self.stream_worker is not None:
            self.stream_worker.collection_complete.connect(
                self.max_value_collection_complete
            )
            self.stream_worker.finish_collect_data()

    def max_value_collection_complete(self, collected_data):
        self.stream_worker.collection_complete.disconnect(
            self.max_value_collection_complete
        )
        self.processing_dock.calc_max_values(collected_data)

    def max_values_changed(self, max_values):
        if self.stream_worker is not None:
            self.stream_worker.set_max_scaling(max_values)

    def handle_stream_error(self, msg):
        self.error_dialog(msg)

    def update_plots(self):
        if self.stream_worker is not None:
            self.live_plot_widget.set_data(self.stream_worker.processed_buffer)  

    def update_model(self, llgmn):
        self.recognition_worker.update_model(llgmn)

    def update_identity(self, identity):
        self.recognition_worker.update_identity(identity)

    def gesture_recognition(self, data):        # LLGMNに送る前のデータ
        self.recognition_worker.input_queue.put(data)

    def movement_recognised(self, result):
        self.radar_plot_widget.set_recognition_result(result)

    def movement_serialed(self, data):
        if self.arduino_worker is not None:
            
            self.arduino_worker.message_queue.put(("emg",data))

    def start_record(self, path):
        if self.stream_worker is not None:
            self.stream_worker.start_record(path)

    def end_record(self):
        if self.stream_worker is not None:
            self.stream_worker.finish_record()

    def gui_save(self, settings):
        self.connect_dock.gui_save(settings)
        self.record_dock.gui_save(settings)
        self.arduino_dock.gui_save(settings)
        self.processing_dock.gui_save(settings)
        self.recognition_dock.gui_save(settings)
        settings.setValue("Window/geometry", self.saveGeometry())
        settings.setValue("Window/state", self.saveState())
        settings.setValue("Window/splitter", self.main_splitter.saveState())

    def gui_restore(self, settings):
        try:
            if geometry := settings.value("Window/geometry"):
                self.restoreGeometry(geometry)
            if state := settings.value("Window/state"):
                self.restoreState(state)
            if state := settings.value("Window/splitter"):
                self.main_splitter.restoreState(state)
            self.connect_dock.gui_restore(settings)
            self.record_dock.gui_restore(settings)
            self.arduino_dock.gui_restore(settings)
            self.processing_dock.gui_restore(settings)
            self.recognition_dock.gui_restore(settings)
        except Exception as e:
            self.error_dialog(f"{self.settings_file} is corrupted!\n{str(e)}")

    def closeEvent(self, event):
        if self.stream_worker is not None:
            self.stream_worker.set_stop()
        settings = QtCore.QSettings(str(self.settings_file), QtCore.QSettings.IniFormat)
        self.gui_save(settings)
        event.accept()

    def error_dialog(self, error):      # QMessageBoxは警告や質問、エラーなどのメッセージボックスを作るためのモノ
        QtWidgets.QMessageBox.critical(self, "Error", error)
        
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWidget()
    # win.show()        # サイズ指定の時はshowのみ
    win.showMaximized()       # 全画面
    app.exec()