import time
import pandas as pd
from muse import MuseMonitor
from datetime import datetime


if __name__ == "__main__":
    
    headset = MuseMonitor("192.168.1.122", 5000)
    time.sleep(10)
    starttime = time.time()
    values = []

    while True:
        time.sleep(1/256 - ((time.time() - starttime) % (1/256)))
        wave = headset.waves
        values += [[datetime.now()] + [headset.raw.value] + list(wave.values())]
        # save data every 10 lines
        if len(values) % 1024 == 0:
            df = pd.DataFrame(values)
            df.to_csv('raw.csv', mode='a', index=False, header=False)
            values = []