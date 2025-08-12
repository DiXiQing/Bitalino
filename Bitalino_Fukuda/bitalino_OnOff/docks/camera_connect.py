from PySide6 import QtWidgets, QtCore
from bitalino_OnOff.docks.dock_base_2 import BaseDock2

class CameraConnect(BaseDock2):
    button_click = QtCore.Signal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Connect to OAK-DS2")

        # main_layout = QtWidgets.QHBoxLayout()
        self.button = QtWidgets.QPushButton(self)
        self.button.setText("Connect")
        self.dock_layout.addWidget(self.button)

        self.button.clicked.connect(self.camera_button_clicked)

        #self.SetButton()

    def camera_button_clicked(self):
        self.button_click.emit()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = CameraConnect()
    widget.show()

    app.exec()