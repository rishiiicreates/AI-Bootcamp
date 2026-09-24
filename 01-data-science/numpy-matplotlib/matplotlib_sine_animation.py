import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()
x = np.linspace(0, 4 * np.pi, 500)
line, = ax.plot(x, np.sin(x), color='crimson', lw=2)
ax.set_ylim(-1.2, 1.2)


def update(frame):
    line.set_ydata(np.sin(x + frame / 10.0))
    return line,


ani = FuncAnimation(fig, update, frames=100, interval=30, blit=True)
plt.show()

