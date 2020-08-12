import  numpy as np
from  matplotlib import pyplot as plt

values = [-1393.5, -1419.1]
errs = [13, 12.4]
weights = []

for e in errs:
    weights.append(1 / (e ** 2))

mean = np.average(values, weights=weights)
print(mean)
std = np.std(values)
print(std)

plt.errorbar([1, 2], values, yerr=errs, fmt='b.')
plt.plot([0, 3], [mean, mean], 'r-')
plt.fill_between([0,3], mean + std, mean - std, alpha=0.2, linewidth=0, color='b')
plt.title('Isotope shift for neg (1) and pos (2) B values')
plt.ylabel('Isotope shift in MHz')
plt.show()