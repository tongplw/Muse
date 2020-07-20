import joblib
import numpy as np
import pandas as pd

from threading import Thread
from scipy.optimize import fmin
from scipy.stats import normaltest
from pythonosc import dispatcher, osc_server
from multiprocessing import Process, Manager, Value
from src.utils import *
from src.RunningStats import RunningStats


class MuseMonitor():

    def __init__(self, server=None, port=None, debug=False):
        self.server = server
        self.port = port
        self.window_size = 1024
        self.sample_rate = 256
        self._buffer = []
        self._attention_buff = [.5, .5, .5, .5, .5]
        self._meditation_buff = [.5, .5, .5, .5, .5]
        self._running_stats = {'att': RunningStats(), 'med': RunningStats()}
        self.scaler = joblib.load('res/scaler')
        
        self.raw = Value('d', 0)
        self.attention = Value('d', 0)
        self.meditation = Value('d', 0)
        self.waves = Manager().dict()

        if not debug:
            process = Process(target=self._run)
            process.daemon = True
            process.start()

    def _get_dispatcher(self):
        d = dispatcher.Dispatcher()
        d.map("/debug", print)
        d.map("/muse/eeg", self._eeg_handler, "EEG")
        return d

    def _get_fft(self, raw_list):
        fft = np.abs(np.fft.rfft(raw_list - np.mean(raw_list))) * 2 / self.window_size
        freqs = np.fft.rfftfreq(self.window_size, 1 / self.sample_rate)
        return [freqs, fft]

    def _get_bands(self, raw_list):
        band_list = {b: [] for b in BAND_RANGE}
        freqs, fft = self._get_fft(raw_list)
        for freq, amps in zip(freqs, fft):
            for b in BAND_RANGE:
                low, high = BAND_RANGE[b]
                if low <= freq < high:
                    band_list[b] += [amps]
        for b in band_list:
            band_list[b] = np.mean(band_list[b]) ** 2
        return band_list

    def _reject_outliers(self, data, m=3):
        data = np.array(data)
        Q3 = np.percentile(data, 75)
        Q1 = np.percentile(data, 25)
        IQR = (Q3 - Q1) * m
        return data[(data > Q1 - IQR) & (data < Q3 + IQR)]
    
    def _is_wearing(self, key, val):
        return val < 10 and self._running_stats[key].get_std() < 0.5

    def _calibrate(self, key, val):
        self._running_stats[key].update(val)
        if not self._is_wearing(key, val):
            self._running_stats[key].clear()
            val = 0
        if self._running_stats[key].get_count() > 5:
            val = (val - self._running_stats[key].get_mean()) / self._running_stats[key].get_std() * 0.23 + 0.5
        return min(1, max(1e-5, val))

    def _convert_to_mindwave(self, band, value):
        mind_c, muse_c, mind_mean, muse_mean = CONVERT_MAP[band]
        value = value / 1.8 * 4096 * 2
        return ((value ** (1 / muse_c)) - muse_mean + mind_mean) ** mind_c

    def _attention(self, waves):
        waves = waves.copy()
        for band in waves:
            waves[band] = self._convert_to_mindwave(band, waves[band])
        for i in range(5):
            waves[f'attention-{i+1}'] = self._attention_buff[i]
        for i in list(waves):
            waves[f'log2-{i}'] = np.log2(waves[i])
        waves['log2-theta-alpha'] = np.log2(waves['theta'] + waves['low-alpha'] + waves['high-alpha'])

        wave_array = np.array([[val for val in waves.values()]])
        wave_transformed = self.scaler.transform(wave_array)
        att = np.sum(wave_transformed * ATT_COEF) + ATT_INTERCEPT

        att = self._calibrate('att', att)
        if 0 < att < 1:
            self._attention_buff = [att] + self._attention_buff[:-1]
        return att
    
    def _meditation(self, waves):
        waves = waves.copy()
        for band in waves:
            waves[band] = self._convert_to_mindwave(band, waves[band])
        for i in range(5):
            waves[f'meditation-{i+1}'] = self._meditation_buff[i]
        for i in list(waves):
            waves[f'log2-{i}'] = np.log2(waves[i])
        waves['log2-theta-alpha'] = np.log2(waves['theta'] + waves['low-alpha'] + waves['high-alpha'])

        wave_array = np.array([[val for val in waves.values()]])
        wave_transformed = self.scaler.transform(wave_array)
        med = np.sum(wave_transformed * MED_COEF) + MED_INTERCEPT

        med = self._calibrate('med', med)
        if 0 < med < 1:
            self._meditation_buff = [med] + self._meditation_buff[:-1]
        return med

    def _eeg_handler(self, unused_addr, args, TP9, AF7, AF8, TP10, AUX):
        self.raw.acquire()
        self.raw.value = AF7 - TP9 # TP9 หลังหู-ซ้าย, AF7 หน้าผาก-ซ้าย
        self.raw.release()
        self._buffer.append(self.raw.value)
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[1:]
            self.waves.update(self._get_bands(self._buffer))
            self._buffer = self._buffer[self.sample_rate:]

            new_attention = self._attention(self.waves)
            self.attention.acquire()
            self.attention.value = np.round(new_attention, 2) * 100
            self.attention.release()
            new_meditation = self._meditation(self.waves)
            self.meditation.acquire()
            self.meditation.value = np.round(new_meditation, 2) * 100
            self.meditation.release()

    def _run(self):
        server = osc_server.BlockingOSCUDPServer((self.server, self.port), self._get_dispatcher())
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()