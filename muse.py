import joblib
import numpy as np
import pandas as pd

from threading import Thread
from scipy.optimize import fmin
from scipy.stats import normaltest
from multiprocessing import Process, Manager, Value
from pythonosc import dispatcher, osc_server


class MuseMonitor():

    def __init__(self, server=None, port=None, debug=False):
        self.server = server
        self.port = port
        self.window_size = 1024
        self.sample_rate = 256
        self._buffer = []
        self._attention_buff = [.5, .5, .5, .5, .5]
        self._attention_history = []
        self.scaler = joblib.load('scaler')
        
        self.raw = Value('d', 0)
        self.attention = Value('d', 0)
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

    def _reject_outliers(self, data, m=3):
        data = np.array(data)
        Q3 = np.percentile(data, 75)
        Q1 = np.percentile(data, 25)
        IQR = (Q3 - Q1) * m
        return data[(data > Q1 - IQR) & (data < Q3 + IQR)]

    def _adjust_attention(self, att):
        if len(self._attention_history) < 10:
            return att
        atts = self._reject_outliers(self._attention_history)
        return (att - np.mean(atts)) / np.std(atts) * 0.2 + np.mean(atts)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _convert_to_mindwave(self, band, value):
        d_map = {'delta':       [7.32900391, 7.47392578, 5.576955, 5.687801],
                'theta':        [6.39179688, 6.41220703, 5.832594, 6.030335],
                'low-alpha':    [5.95166016, 5.65654297, 5.188705, 5.819261],
                'high-alpha':   [5.67324219, 5.06787109, 5.440118, 6.053460],
                'low-beta':     [5.53769531, 4.69228516, 5.509398, 5.892162],
                'high-beta':    [5.55654297, 4.56396484, 5.261583, 5.753560],
                'low-gamma':    [5.14970703, 4.81093750, 5.088524, 5.302043],
                'mid-gamma':    [7.08144531, 4.92177734, 4.860328, 5.343249]}
        mind_c, muse_c, mind_mean, muse_mean = d_map[band]
        value = value / 1.8 * 4096 * 2
        return ((value ** (1 / muse_c)) - muse_mean + mind_mean) ** mind_c

    def _attention(self, waves):
        waves = waves.copy()
        for band in waves:
            waves[band] = self._convert_to_mindwave(band, waves[band])
        index = ['delta', 'theta', 'low-alpha', 'high-alpha', 
                'low-beta', 'high-beta', 'low-gamma', 'mid-gamma', 
                'attention-1', 'attention-2', 'attention-3', 'attention-4', 
                'attention-5', 'log2-delta', 'log2-theta', 'log2-low-alpha', 
                'log2-high-alpha', 'log2-low-beta', 'log2-high-beta', 'log2-low-gamma', 
                'log2-mid-gamma', 'log2-attention-1', 'log2-attention-2', 'log2-attention-3', 
                'log2-attention-4', 'log2-attention-5', 'log2-theta-alpha']
        coef_ = [1.85597993e-03, -3.89405744e-02, -2.17976458e-02,
                -4.94719580e-03,  5.92689481e-02, -2.66903157e-02,
                2.30846084e-02,  6.82606511e-02,  8.54525920e-01,
                2.10894178e-02, -1.06262949e-01, -2.69200545e-01,
                2.50926910e-01,  6.62901487e-03,  5.85212860e-03,
                6.45113734e-03,  1.61475866e-04, -1.34012739e-03,
                7.47788932e-01, -8.24321394e-04, -1.39632993e-03,
                -9.23905598e-03, -9.27560591e-03,  2.64911164e-02,
                2.01285851e-03, -2.64492016e-03, -7.13587632e-01]
        intercept_ = 0.18241877

        for i in range(5):
            waves[f'attention-{i+1}'] = self._attention_buff[i]
        for i in list(waves):
            waves[f'log2-{i}'] = np.log2(waves[i])
        waves['log2-theta-alpha'] = np.log2(waves['theta'] + waves['low-alpha'] + waves['high-alpha'])

        wave_array = np.array([[val for val in waves.values()]])
        wave_transformed = self.scaler.transform(wave_array)
        att = np.sum(wave_transformed * coef_) + intercept_

        self._attention_history += [att]
        att = self._adjust_attention(att)
        att = min(1, max(1e-5, att))
        self._attention_buff = [att] + self._attention_buff[:-1]
        return att

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