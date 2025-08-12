from PySide6 import QtWidgets, QtCore

from bitalino_OnOff.docks.dock_base_2 import BaseDock2
from emg_interface.groupboxes import CalibrationGroupBox

class ProcessingDock2(BaseDock2):
    processing_updated = QtCore.Signal(dict)
    start_zero_offset_calibration = QtCore.Signal()
    finish_zero_offset_calibration = QtCore.Signal()
    zero_offset_changed = QtCore.Signal(list)
    start_max_value_calibration = QtCore.Signal()
    finish_max_value_calibration = QtCore.Signal()
    max_value_changed = QtCore.Signal(list)

    def __init__(self, num_channels=4):
        super().__init__()

        self.num_channels = num_channels

        self.setWindowTitle("Processing")

        self.rectify_checkbox = QtWidgets.QCheckBox("Rectify", self)
        self.dock_layout.addWidget(self.rectify_checkbox)

        self.envelope_checkbox = QtWidgets.QCheckBox("Envelope", self)
        self.dock_layout.addWidget(self.envelope_checkbox)

        self.min_scaling_checkbox = QtWidgets.QCheckBox("Min_scaling", self)
        self.dock_layout.addWidget(self.min_scaling_checkbox)

        self.processes = {
            "rectify": False,
            "envelop": False,
            "min_scaling":False,
        }

        self.dock_layout.addStretch()

        self.zero_offset_groupbox = CalibrationGroupBox(  #normal 3秒間
            num_channels, duration=3, parent=self
        )
        self.zero_offset_groupbox.setTitle("Zero Offset Calibration")
        self.zero_offset_groupbox.set_button_tooltip("!! TO DO")
        self.zero_offset_groupbox.start_calibration.connect(
            self.zero_offset_calibration_started
        )
        self.zero_offset_groupbox.finish_calibration.connect(
            self.zero_offset_calibration_finished
        )
        self.zero_offset_groupbox.cal_value_changed.connect(
            self.zero_offset_changed.emit
        )
        self.dock_layout.addWidget(self.zero_offset_groupbox)

        self.rectify_checkbox.toggled.connect(self.checkbox_selection_changed)
        self.envelope_checkbox.toggled.connect(self.checkbox_selection_changed)
        self.min_scaling_checkbox.toggled.connect(self.checkbox_selection_changed)

    def checkbox_selection_changed(self):  # チェックボックス連鎖
        order = [
            self.rectify_checkbox,
            self.envelope_checkbox,
            self.min_scaling_checkbox,
        ]
        i = order.index(self.sender())

        for checkbox in order[:i]:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        for checkbox in order[i + 1:]:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

        self.processes = {
            "rectify": self.rectify_checkbox.isChecked(),
            "envelop": self.envelope_checkbox.isChecked(),
            "min_scaling": self.min_scaling_checkbox.isChecked(),
        }
        self.processing_updated.emit(self.processes)

    def zero_offset_calibration_started(self):
        self.start_zero_offset_calibration.emit()

    def zero_offset_calibration_finished(self):
        self.finish_zero_offset_calibration.emit()

    def calc_zero_offsets(self, collected_data):
        zero_offsets = collected_data.mean(axis=0)
        self.zero_offset_groupbox.set_cal_values(zero_offsets)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = ProcessingDock2(4)
    widget.show()

    app.exec()