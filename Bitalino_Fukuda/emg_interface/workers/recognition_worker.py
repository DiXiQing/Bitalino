from queue import Queue

import numpy as np
from PySide6 import QtCore

class RecognitionWorker(QtCore.QObject):
    recognised = QtCore.Signal(object) # 数値か文字
    serialed = QtCore.Signal(object)  # arduinoに送る用，数値のみ

    def __init__(self):
        super().__init__()

        self.llgmn = None
        
        self.identity = {}
        self.input_queue = Queue()
        self.stop = False

    def run(self):
        while not self.stop:
            data = self.input_queue.get(block=True)
            data = data[-1, :]
            if self.llgmn is not None:
                output = self.llgmn.forward(np.array([data]))
                movement = str(output.argmax())  # 数値のみ
                
                self.serialed.emit(movement)
                if movement in self.identity:
                    movement = self.identity[movement]
                self.recognised.emit(movement)

    def update_model(self, llgmn):
        self.llgmn = llgmn

    def update_identity(self, identity):
        self.identity = identity

    def set_stop(self):
        self.stop = True


if __name__ == "__main__":
    worker = RecognitionWorker()
    worker.run()
