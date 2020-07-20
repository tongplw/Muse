import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

size = 100
x_vec = np.linspace(0, 1, size + 1)[:-1]
y_vecs = []
lines = []

def live_plotter(x_vec, y1_data, line1, title='', pause_time=1e-2):
    if line1 == []:
        plt.ion()
        fig = plt.figure(figsize=(15, 3))
        ax = fig.add_subplot(111)
        line1, = ax.plot(x_vec, y1_data,'-o',alpha=0.8)        
        plt.ylabel('Y Label')
        plt.title(title)
        plt.show()
    line1.set_ydata(y1_data)
    plt.ylim([-5, 105])
    plt.pause(pause_time)
    return line1

def live_plot(*args):
    global x_vec, y_vecs, lines, size
    while len(y_vecs) < len(args):
        y_vecs += [np.zeros(size)]
        lines += [[]]
    for i, v in enumerate(args):
        y_vecs[i] = np.append(y_vecs[i][1:], v)
        try:
            lines[i] = live_plotter(x_vec, y_vecs[i], lines[i])
        except:
            pass