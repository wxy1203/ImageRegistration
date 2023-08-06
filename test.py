import numpy as np
a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8]
result = np.cross(np.array(a), np.array(b))
print(result)

m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
n = [[9, 0, 0], [0, 11, 0]]
r = np.dot(m, a)
s = np.dot(n, m)
# t = np.dot(m, n)
# print(r)
print(s)
#print(t)