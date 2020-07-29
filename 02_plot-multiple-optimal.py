#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,20)

z = 0
y1 = x + 1
y2 = 0.5*x + 2
Z = lambda x,z: 0.5*x + z

plt.plot(x,y1, color='red')
plt.plot(x,y2, color='y')
for z in range(-1,3):
    plt.plot(x, Z(x, z), 'c--')
plt.axvline(color='gray')
plt.axhline(color='gray')
plt.xlim(-1,6)
plt.ylim(-1,6)
plt.show()