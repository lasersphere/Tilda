import  numpy as np
from  matplotlib import pyplot as plt

values = [-121.1, -121.6]
errs = [4.5, 4.7]
weights = []

for e in errs:
    weights.append(1 / (e ** 2))

mean = np.average(values, weights=weights)
print(mean)
std = np.std(values)
print(std)

plt.errorbar([1, 2], values, yerr=errs, fmt='b.')
plt.plot([0, 3], [mean, mean], 'r-')
plt.fill_between([0, 3], mean + std, mean - std, alpha=0.2, linewidth=0, color='b')
plt.title('lower A for pos (1) and neg (2) B values')
plt.ylabel('lower A in MHz')
plt.show()