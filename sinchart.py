import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)

y = np.sin(x)

plt.plot(x, y)


plt.xlabel("time")
plt.ylabel("sin(x)")
plt.title("Sine Wave")

plt.grid(True)

plt.show()
