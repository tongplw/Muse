import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


df = pd.read_csv('data.csv', header=None)
df = df.round(2)
df = df[df[0] % 1 != 0]
df.hist(bins=200)
plt.show()