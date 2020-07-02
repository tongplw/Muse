import joblib
import numpy as np
import pandas as pd

from threading import Thread
from scipy.optimize import fmin
from scipy.stats import normaltest
from multiprocessing import Process, Manager, Value
from pythonosc import dispatcher, osc_server


class MuseMonitor():

    def __init__(self, server, port):
        self.server = server
        self.port = port
        self.window_size = 1024
        self.sample_rate = 256
        self._buffer = []
        self._attention_buff = [.5, .5, .5, .5, .5]
        self.scaler = joblib.load('scaler')
        
        self.raw = Value('d', 0)
        self.waves = Manager().dict()
        self.attention = Value('d', 0)

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
        bands = {'delta': (1, 3), 'theta': (4, 7), 'low-alpha': (8, 9), 'high-alpha': (10, 12),
                'low-beta': (13, 17), 'high-beta': (18, 30), 'low-gamma': (30, 40), 'mid-gamma': (41, 50)}
        band_list = {b: [] for b in bands}
        freqs, fft = self._get_fft(raw_list)
        for freq, amps in zip(freqs, fft):
            for b in bands:
                low, high = bands[b]
                if low <= freq < high:
                    band_list[b] += [amps]
        for b in band_list:
            band_list[b] = np.mean(band_list[b]) ** 2
        return band_list

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _convert_to_mindwave(self, band, value):
        d_map = {'delta': [6.22949219, 3.34765625, 5.782872, 2.108653],
                'theta': [5.60664063, 2.18398438, 5.908993, 1.616526],
                'low-alpha': [5.10400391, 1.18662109, 5.842178, 1.166655],
                'high-alpha': [5.58320313, 0.61787109, 5.449491, 1.045655],
                'low-beta': [4.77851563, 0.20209961, 6.226439, 0.109606],
                'high-beta': [4.32861328, 1.23632813, 7.679877, 0.276243],
                'low-gamma': [4.13066406, 1.72353516, 6.415543, 0.269904],
                'mid-gamma': [6.01787109, 2.26328125, 4.476359, 0.306473]}
        mind_c, muse_c, mind_mean, muse_mean = d_map[band]
        return ((value ** (1/muse_c)) - muse_mean + mind_mean) ** mind_c

    def _attention(self, waves):
        waves = waves.copy()
        for band in waves:
            waves[band] = self._convert_to_mindwave(band, waves[band])
        index = ['delta', 'theta', 'low-alpha', 'high-alpha', 'low-beta', 'high-beta',
                'low-gamma', 'mid-gamma', 'attention-1', 'attention-2', 'attention-3',
                'attention-4', 'attention-5', 'log2-delta', 'log2-theta',
                'log2-low-alpha', 'log2-high-alpha', 'log2-low-beta', 'log2-high-beta',
                'log2-low-gamma', 'log2-mid-gamma', 'log2-attention-1',
                'log2-attention-2', 'log2-attention-3', 'log2-attention-4',
                'log2-attention-5', 'log2-theta-alpha']
        coef = [3.53705769e-03, -4.23337846e-02, -2.69203699e-02, -1.34591328e-02,
                7.62320160e-02, -1.88313044e-02, -7.54789000e-03,  6.47782749e-02,
                8.54250997e-01,  1.73868814e-02, -1.09176174e-01, -2.62815125e-01,
                2.48854082e-01,  1.00844863e-02,  7.98661639e-03,  1.04763222e-02,
                8.79394322e-03, -7.04150397e-03,  7.52820760e-01,  4.04370665e-04,
                2.35751280e-03, -1.27259062e-03, -4.07376421e-03,  2.82478619e-02,
                -3.90226910e-03, -4.99784235e-03, -7.18270207e-01]

        for i in range(5):
            waves[f'attention-{i+1}'] = self._attention_buff[i]
        for i in list(waves):
            waves[f'log2-{i}'] = np.log2(waves[i])
        waves['log2-theta-alpha'] = np.log2(waves['theta'] + waves['low-alpha'] + waves['high-alpha'])

        wave_array = np.array([[val for val in waves.values()]])
        wave_transformed = self.scaler.transform(wave_array)
        att = np.sum(wave_transformed * coef)
        if att < 50:
            att = self._sigmoid((att-0.25) * 2.5) + 1e-5
            self._attention_buff = [att] + self._attention_buff[:-1]
            return att
        else:
            return 1e-5 # self._attention_buff[0]

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

    def _run(self):
        server = osc_server.BlockingOSCUDPServer((self.server, self.port), self._get_dispatcher())
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()