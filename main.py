import time
import pandas as pd
from src.muse import MuseMonitor
from datetime import datetime


if __name__ == "__main__":
    
    headset = MuseMonitor("192.168.1.122", 5000)
    # headset = MuseMonitor("192.168.43.126", 5000)
    time.sleep(10)
    starttime = time.time()
    values = []

    while True:
        time.sleep(1)
        wave = headset.waves
        values += [[headset.attention.value] + list(wave.values())]
        # values += [[headset.attention.value]]
        print(headset.attention.value)

        # save data every 10 lines
        if len(values) % 10 == 0:
            df = pd.DataFrame(values)
            df.to_csv('data.csv', mode='a', index=False, header=False)
            values = []