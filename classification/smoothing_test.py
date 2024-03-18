import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import time


f = lambda x: np.sin(x)

x = np.linspace(0, 2 * np.pi, 100)
random_points = np.random.random(100) * 0.8
y = f(x) + random_points
true_y = f(x) + np.mean(random_points)

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

MSE = {}


total = len(range(1, 100, 5))
with alive_bar(total) as bar:  # declare your expected total
    for i in range(1, 100, 5):
        temp_y = smooth(y, i)
        MSE[i] = np.mean(np.square(true_y - temp_y))

        plt.plot(x, temp_y, lw=0.5, label=str(i) + " points")
        bar()

plt.plot(x, y, 'o')
plt.plot(x, true_y, 'b-', lw=2)
plt.text(0, -0.1, "Minimum MSE, box_pts = " + str(min(MSE, key=MSE.get)))
plt.plot(x, smooth(y, min(MSE, key=MSE.get)), 'r-', lw=2)


plt.legend()
plt.show()

plt.plot(MSE.keys(), MSE.values())
plt.plot(min(MSE, key=MSE.get), min(MSE.values()), 'ro')
plt.xlabel("Box points")
plt.ylabel("MSE")
plt.show()