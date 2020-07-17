import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.muse import MuseMonitor
from src.pylive import live_plot

if __name__ == "__main__":
    
    headset = MuseMonitor(debug=True)
    df = pd.read_csv('res/data.csv', header=None).iloc[:, 1:]
    df.columns = ['delta', 'theta', 'low-alpha', 'high-alpha', 'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']

    atts = []
    meds = []

    for i in tqdm(range(len(df))):
        waves = df.iloc[i].to_dict()
        attention = headset._attention(waves) * 100
        atts += [attention]
        meditation = headset._meditation(waves) * 100
        meds += [meditation]

    # plt.hist(atts, bins=200)
    # plt.hist(meds, bins=200, alpha=0.5)
    plt.plot(atts)
    plt.plot(meds)
    plt.show() 