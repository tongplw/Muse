import numpy as np
import pandas as pd

from threading import Thread
from datetime import datetime
from pythonosc import dispatcher, osc_server


class MuseMonitor():

    def __init__(self, server, port):
        self.server = server
        self.port = port
        self.fft_window_size = 1024
        self.sample_rate = 256
        self._buffer = []
        self._attention_buff = [50, 50, 50, 50, 50]
        self.raw = 0
        self.waves = {}
        self.attention = 0

        thread = Thread(target=self._run)
        thread.daemon = True
        thread.start()

    def _get_dispatcher(self):
        d = dispatcher.Dispatcher()
        d.map("/debug", print)
        d.map("/muse/eeg", self._eeg_handler, "EEG")
        return d

    def _get_fft(self, raw_list):
        fft = np.abs(np.fft.rfft(raw_list))
        freqs = np.fft.rfftfreq(self.fft_window_size, 1 / self.sample_rate)
        return [freqs, fft]

    def _get_bands(self, raw_list):
        bands = {'delta': (1, 4), 'theta': (4, 8), 'low-alpha': (8, 10), 'high-alpha': (10, 13),
                'low-beta': (13, 18), 'high-beta': (18, 31), 'low-gamma': (30, 41), 'mid-gamma': (41, 50)}
        band_list = {b: [] for b in bands}
        freqs, fft = self._get_fft(raw_list)
        for freq, amps in zip(freqs, fft):
            for b in bands:
                low, high = bands[b]
                if low <= freq < high:
                    band_list[b] += [amps]
        for b in band_list:
            band_list[b] = np.mean(band_list[b])
        return band_list

    def _attention(self, waves):
        waves = waves.copy()
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

        att = 0
        for ind, coe in zip(index, coef):
            att += waves[ind] * coe
        self._attention_buff = [max(1e-5, att)] + self._attention_buff[:-1]
        return att

    def _eeg_handler(self, unused_addr, args, TP9, AF7, AF8, TP10, AUX):
        self.raw = AF7 - TP9 # TP9 หลังหู-ซ้าย, AF7 หน้าผาก-ซ้าย
        self._buffer.append(self.raw)
        if len(self._buffer) > self.fft_window_size:
            self._buffer = self._buffer[1:]

            self.waves = self._get_bands(self._buffer)
            self.attention = self._attention(self.waves)
            self._buffer = self._buffer[self.sample_rate:]

    def _run(self):
        server = osc_server.BlockingOSCUDPServer((self.server, self.port), self._get_dispatcher())
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()