import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data = np.loadtxt('svmData.csv', delimiter=',')

X_data = np.delete(data, 2, axis=1)
y = np.delete(data, [0, 1], axis=1)

X = StandardScaler().fit_transform(X_data)

const_C = np.linspace(1, 100, 100)
loss = [0] * len(const_C)
br = [0] * len(const_C)

for i in range(0, len(const_C)):
    clf = SVC(C=const_C[i], kernel='linear')
    clf.fit(X, y.ravel())

    w = clf.coef_
    b = clf.intercept_
    supp_vec = clf.support_vectors_

    for j in range(0, len(X)):
        gama_kapa = (w @ X[j] + b) * y[j]

        if (gama_kapa < 1):
            br[i] += 1
            loss[i] +=  1 - gama_kapa


plt.plot(const_C, loss)
plt.xlabel('C')
plt.ylabel('Loss')
plt.show()

plt.scatter(const_C, br)
plt.xlabel('C')
plt.ylabel('Loss count')
plt.show()

C_opt = const_C[br.index(min(br))]

clf = SVC(C=C_opt, kernel='linear')
clf.fit(X, y.ravel())

print('w = ', clf.coef_)
print('b = ', clf.intercept_)
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))

w = clf.coef_
b = clf.intercept_
indeksi = clf.support_
supp_vec = clf.support_vectors_

colors = []
boja_kruga = []

for i in range(len(X)):
    is_support_vector = False
    for j in range(len(supp_vec)):
        if np.array_equal(X[i], supp_vec[j]):
            is_support_vector = True
            break

    if is_support_vector:
        if y[i][0] == 1:
            colors.append('red')
            boja_kruga.append('black')
        elif y[i][0] == -1:
            colors.append('cyan')
            boja_kruga.append('black')
    else:
        if y[i][0] == 1:
            colors.append('red')
            boja_kruga.append('red')
        elif y[i][0] == -1:
            colors.append('cyan')
            boja_kruga.append('cyan')


xmin = -2
xmax = 2
ymin = -2
ymax = 2

koef = -w[0][0] / w[0][1]
odsecak = -b[0] / w[0][1]

osa_x = np.linspace(xmin, xmax)
osa_y = koef * osa_x + odsecak
plt.plot(osa_x, osa_y)

plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), alpha=0.9, cmap=matplotlib.colors.ListedColormap(colors), edgecolors=boja_kruga)


gubici = []
for i in range(0, len(supp_vec), 3):
    gama = (w @ supp_vec[i] + b) * y[indeksi[i]][0]
    gubici.append(1 - gama)

brr = 0
for i in range(0, len(supp_vec), 3):
    plt.text(supp_vec[i][0], supp_vec[i][1], round(gubici[brr][0], 2))
    brr += 1

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()