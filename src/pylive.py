import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

size = 100
x_vec = np.linspace(0, 1, size + 1)[:-1]
y_vec = np.zeros(size)
line1 = []

def live_plotter(x_vec, y1_data, line1, title='', pause_time=1e-2):
    if line1 == []:
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        line1, = ax.plot(x_vec, y1_data,'-o',alpha=0.8)        
        plt.ylabel('Y Label')
        plt.title(title)
        plt.show()
    line1.set_ydata(y1_data)
    plt.ylim([-5, 105])
    plt.pause(pause_time)
    return line1

def plot_attention(att):
    global x_vec, y_vec, line1
    y_vec = np.append(y_vec[1:], att)
    try:
        line1 = live_plotter(x_vec, y_vec, line1)
    except:
        pass