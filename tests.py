import numpy as np

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([5, 4, 3])

print("x1: ")
print(x1)

# x1_new = x1[:, np.newaxis]
x1_new = x1[np.newaxis, :]
# now the shape of s1_new is (5, 1)

print("x1_new: ")
print(x1_new)
# print("x1_new + x2: ")
# print(x1_new + x2)
