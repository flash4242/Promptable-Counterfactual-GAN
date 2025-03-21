#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))  # Két alábra (2 sor, 1 oszlop)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax1.plot(x, y1, label="sin(x)", color='blue')  # Első ábra
ax1.set_title("Sinus függvény")
ax1.legend()

ax2.plot(x, y2, label="cos(x)", color='red')  # Második ábra
ax2.set_title("Cosinus függvény")
ax2.legend()

plt.tight_layout()  # Hogy ne csússzanak egymásra az ábrák
plt.savefig("plotolas_pelda.png")
