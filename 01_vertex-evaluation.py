#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
import numpy as np

points = {'A' : (20, 0),
'B' : (0, 25),
'C' : (0, 30),
'D' : (30.7692307692308, 11.5384615384615),
'E' : (40, 0)}

objective = lambda x, y: 3*x + 4*y

maximum = 0
for label,point in points.items():
    z = objective(*point)
    print(label, z)
    if z > maximum:
        maximum = z
        result = label, z
print(result)
    
    