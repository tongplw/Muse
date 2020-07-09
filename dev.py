import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.muse import MuseMonitor
from src.pylive import plot_attention

if __name__ == "__main__":
    
    headset = MuseMonitor(debug=True)
    df = pd.read_csv('res/data.csv', header=None).iloc[:, 1:]
    df.columns = ['delta', 'theta', 'low-alpha', 'high-alpha', 'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']

    for i in tqdm(range(len(df))):
        waves = df.iloc[i].to_dict()
        attention = headset._attention(waves)

        attention = int(np.round(float(attention), 2) * 100)
        plot_attention(attention)
        time.sleep(0.5)