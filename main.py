import time
import pandas as pd

from datetime import datetime
from src.muse import MuseMonitor
from src.pylive import plot_attention


if __name__ == "__main__":
    
    headset = MuseMonitor("192.168.1.122", 5000)
    time.sleep(5)
    starttime = time.time()
    values = []

    while True:
        time.sleep(1)
        attention = headset.attention.value
        plot_attention(attention)