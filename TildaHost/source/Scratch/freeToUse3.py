"""

Created on '21.08.2015'

@author:'simkaufm'

"""
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.widgets import CheckButtons
#
# t = np.arange(0.0, 2.0, 0.01)
# s0 = np.sin(2*np.pi*t)
# s1 = np.sin(4*np.pi*t)
# s2 = np.sin(6*np.pi*t)
#
# fig, ax = plt.subplots()
# l0, = ax.plot(t, s0, visible=False, lw=2)
# l1, = ax.plot(t, s1, lw=2)
# l2, = ax.plot(t, s2, lw=2)
# plt.subplots_adjust(left=0.2)
#
# rax = plt.axes([0.05, 0.4, 0.1, 0.15])
# check = CheckButtons(rax, ('2 Hz', '4 Hz', '6 Hz'), (False, True, True))
#
#
# def func(label):
#     print(label)
#     if label == '2 Hz':
#         l0.set_visible(not l0.get_visible())
#     elif label == '4 Hz':
#         l1.set_visible(not l1.get_visible())
#     elif label == '6 Hz':
#         l2.set_visible(not l2.get_visible())
#     plt.draw()
# check.on_clicked(func)
#
# plt.show()
#
# x = np.arange(-2, 2, 0.02)
# y = np.arange(-2, 2, 0.01)
# X, Y = np.meshgrid(x, y)
# print(X)
# ellipses = X * X / 9 + Y * Y / 4 -1
# print(type(ellipses))
# print(len(ellipses[0]))
# plt.imshow(Y, origin='lower', extent=[0, 10, 0, 10])
# plt.colorbar()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=2)
print(axes)
for ax in axes.flat:
    print(ax)
    im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()