# Filename: part1.py
# Date Created: 19-Mar-2019 11:16:08 pm
# Description:
# ----- 1D Example ------------------------------------------------
n = 100

x = mgrid[-1:1:complex(0,n)].reshape(n, 1)
# set y and add random noise
y = sin(3*(x+0.5)**3 - 1)
# y += random.normal(0, 0.1, y.shape)

# rbf regression
rbf = RBF(1, 10, 1)
rbf.train(x, y)
z = rbf.test(x)

# plot original data
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'k-')

# plot learned model
plt.plot(x, z, 'r-', linewidth=2)

# plot rbfs
plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')

for c in rbf.centers:
    # RF prediction lines
    cx = arange(c-0.7, c+0.7, 0.01)
    cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
    plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

plt.xlim(-1.2, 1.2)
plt.show()

random.permutation(X.shape[0])[:self.numCenters]

import scipy
scipy.random.permutation(1)[:4]
