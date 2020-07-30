import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.muse import MuseMonitor
from src.pylive import live_plot

if __name__ == "__main__":
    
    headset = MuseMonitor(debug=True)
    df = pd.read_csv('res/raw.csv')
    
    atts = []
    meds = []

    for i in tqdm(range(len(df))):
        AF7, TP9 = df.iloc[i]
        headset._eeg_handler(AF7=AF7, TP9=0)
        attention = headset.attention.value
        meditation = headset.meditation.value
        atts += [attention]
        meds += [meditation]

    # plt.hist(atts, bins=200)
    # plt.hist(meds, bins=200, alpha=0.5)
    plt.plot(atts)
    plt.plot(meds)
    plt.show() 