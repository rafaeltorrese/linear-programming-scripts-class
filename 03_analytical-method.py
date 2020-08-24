#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
import numpy as np
Z = np.array([3, 4])
A = np.array([[1,1,1,0],
              [2,1,0,1]])
b = np.array([450, 600])

print(A.shape)
solution1 = np.zeros(A.shape[1])
print(solution1)
Ainv1 = np.linalg.inv(A[:, [0,1]])
Ainv2 = np.linalg.inv(A[:, [0,2]])
Ainv3 = np.linalg.inv(A[:, [0,3]])
Ainv4 = np.linalg.inv(A[:, [1,2]])
Ainv5 = np.linalg.inv(A[:, [1,3]])
Ainv6 = np.linalg.inv(A[:, [2,3]])
basic1 = Ainv1.dot(b)
basic2 = Ainv2.dot(b)
basic3 = Ainv3.dot(b)
basic4 = Ainv4.dot(b)
basic5 = Ainv5.dot(b)
basic6 = Ainv6.dot(b)
solution1[[0,1]] = basic1
print(solution1)
print(basic1)
print(basic2)
print(basic3)
print(basic4)
print(basic5)
print(basic6)



print(list({0,1,2,3} - {0,1}))
print({0,1,2,3} - {0,2})
# for i in range(1,7):
#     print("print(basic{})".format(i))
# for i in range(1,7):
#     print("print(z{})".format(i,i))