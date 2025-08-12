import csv

import numpy as np
from PySide6 import QtCore
from bitalino import BITalino

import time     # 確認用

from emg_interface.funcs import (
    adc_to_mV,
    setup_realtime_envelop_filter,
    realtime_filter,
)


class StreamWorker(QtCore.QObject):
    finished = QtCore.Signal()
    stream_error = QtCore.Signal(str)
    collection_complete = QtCore.Signal(object)
    downsampled_data = QtCore.Signal(object)
    zero_data = QtCore.Signal(object)

    def __init__(
            self,
            mac_address,
            window_length_seconds=5,
            sampling_rate=1000,
            num_channels=4,
            chunk_size=10,
    ):
        super().__init__()

        self.mac_address = mac_address
        self.sampling_rate = sampling_rate
        self.window_length_seconds = window_length_seconds
        self.processing = {
            "rectify": False,
            "envelop": False,
            "max_min_scaling": False,
            "channel_normalise": False,
        }
        self.num_channels = num_channels
        self.chunk_size = chunk_size

        self.window_length_samples = self.window_length_seconds * self.sampling_rate

        self.raw_buffer = np.zeros((self.window_length_samples, self.num_channels))
        self.processed_buffer = np.zeros(
            (self.window_length_samples, self.num_channels)
        )

        self.b_i, self.a_i, self.z_i = self.set_envelop_cutoff_freq(1)  # カットオフ周波数1Hz

        self.stop = False
        self.zero_offset = np.zeros(self.num_channels)
        self.max_scaling = np.ones(self.num_channels)
        self.channel_scaling = np.ones(self.num_channels)

        self.collect_flag = False
        self.collected = []

        self.downsampling_prescaler = 10  # 10 Hz (100 Hz [Batch update rate] / 10)
        self.downsampling_counter = 0

        self.record_file = None
        self.record_writer = None

        self.acqChannels = [1, 2, 3, 4]

        self.mutex = QtCore.QMutex()

        self.power_size = 0
        self.start_Time = time.time()  # 確認用
        self.count_data = 0

    def stream(self):
        device = None

        print_count = 0

        try:
            device = BITalino(self.mac_address)
            device.start(self.sampling_rate, self.acqChannels)
        except Exception as e:
            self.stream_error.emit(str(e))

            if device is not None:
                device.close()
                device = None
            self.stop = True

        while not self.stop:
            try:
                raw_data = device.read(self.chunk_size)[:, 5:]
                raw_data = adc_to_mV(raw_data)
            except Exception as e:
                self.stream_error.emit(str(e))
                self.stop = True
                break

            processed_data = raw_data.copy()
            if self.processing["rectify"]:
                processed_data = np.abs(processed_data)  # 整流
            if self.processing["envelop"]:      # ローパスフィルタ
                processed_data = self.envelop_filter(processed_data)

            if self.processing["max_min_scaling"]:  # minを0、maxを1とする
                processed_data = processed_data - self.zero_offset  # 無力からの差分処理
                processed_data = processed_data / self.max_scaling
                zero = processed_data[-1,:]
                zero = sum(zero)
                # print("合計: ",zero)      # 力の強さを調べる用

                added_processed_data = processed_data
                

            if self.processing["channel_normalise"]:    # 正規化
                # max_min_scalingのzeroを用いて無力の閾値設定．閾値以下であれば無力と判定．
                muryoku = [0.17, 0.18, 0.35, 0.3]       # このmuryokuはfukutaの無力時のデータ平均を利用

                if zero < 0.1:           # 閾値、動作とみなすかどうか
                    processed_data = np.tile(muryoku, (10,1))            
                    self.power_size = 0

                for i in range(processed_data.shape[0]):
                    min_data = np.min(processed_data[i])
                    if min_data < 0:
                        processed_data += abs(min_data)

                sum_data = (processed_data.sum(axis=1)[:, None])
                processed_data = processed_data / sum_data

                # print(processed_data[-1,:])

            # roll data
            self.mutex.lock()
            self.raw_buffer[: -self.chunk_size, :] = self.raw_buffer[
                                                     self.chunk_size:, :
                                                     ]
            self.raw_buffer[-self.chunk_size:, :] = raw_data
            self.mutex.unlock()

            # processed_data
            self.mutex.lock()
            self.processed_buffer[: -self.chunk_size, :] = self.processed_buffer[
                                                           self.chunk_size:, :
                                                           ]
            self.processed_buffer[-self.chunk_size:, :] = processed_data
            self.mutex.unlock()

            if self.collect_flag:  # 押されたとき配列保存
                self.collected.append(processed_data)

            if self.downsampling_counter >= self.downsampling_prescaler:
                self.downsampled_data.emit(processed_data)
                self.downsampling_counter = 0
            else:
                self.downsampling_counter += 1

            if self.record_writer is not None:
                self.record_data(raw_data, added_processed_data)

        if device is not None:
            device.stop()
            device.close()
        self.downsampling_counter = 0
        self.stop = False
        self.finished.emit()

    def update_processing(self, new_processing):
        self.processing.update(new_processing)

    def set_envelop_cutoff_freq(self, freq):
        b_i, a_i, z_i = setup_realtime_envelop_filter(freq, fs=self.sampling_rate)
        return b_i, a_i, [z_i] * self.num_channels

    def envelop_filter(self, data):
        filtered_data = np.zeros((self.chunk_size, self.num_channels))
        for j in range(self.num_channels):
            filtered_data[:, j], self.z_i[j] = realtime_filter(
                data[:, j], self.b_i, self.z_i[j], self.a_i
            )  # フィルター処理
        return filtered_data

    def set_stop(self):
        self.stop = True

    def start_collect_data(self):
        self.collect_flag = True

    def set_zero_offsets(self, offsets):
        self.zero_offset = offsets

    def set_max_scaling(self, max_scaling):
        self.max_scaling = max_scaling

    def finish_collect_data(self):
        self.collect_flag = False
        self.collection_complete.emit(np.concatenate(self.collected, axis=0))
        self.collected.clear()

    def start_record(self, path):
        self.record_file = open(path, mode="w", encoding="utf-8", newline="")
        self.record_writer = csv.writer(self.record_file, delimiter=",")
        self.record_writer.writerow(
            [f"raw_ch{i + 1}" for i in range(self.num_channels)]
            + [f"add_processed_ch{i + 1}" for i in range(self.num_channels)]
        )

    def record_data(self, raw_data, processed_data):
        self.record_writer.writerows(np.concatenate([raw_data, processed_data], axis=1))

    def finish_record(self):
        self.record_file.close()
        self.record_writer = None
        self.record_file = None


if __name__ == "__main__":
    from time import sleep

    worker = StreamWorker("00:21:06:BE:18:0B")
    worker.stream()
    # print(worker.raw_buffer)

    sleep(1)
    worker.set_stop()
