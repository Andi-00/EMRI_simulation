import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

colormap = cm.get_cmap("jet")

x = np.arange(0, 10)

fig, ax = plt.subplots()

for i in range(1, len(x)):
    col = cm(i / len(x))

    ax.plot(x[(i - 1) * len - i], x,)