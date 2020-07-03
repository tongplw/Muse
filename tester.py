import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from muse import MuseMonitor


if __name__ == "__main__":
    
    headset = MuseMonitor(debug=True)
    df = pd.read_csv('data.csv', header=None).iloc[:, 1:]
    df.columns = ['delta', 'theta', 'low-alpha', 'high-alpha', 'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']

    atts = []

    for i in range(len(df)):
        waves = df.iloc[i].to_dict()
        attention = headset._attention(waves)
        if attention < 1:
            atts += [attention]

    plt.hist(atts, bins=200)
    plt.show()