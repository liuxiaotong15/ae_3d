import functools
import itertools as IT
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
seed = 1234
random.seed(seed)
np.random.seed(seed)

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

detail_cnt = 50

cnt = np.random.rand(detail_cnt, detail_cnt)

# @profile  # used with `python -m memory_profiler script.py` to measure memory usage
def main():
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = plt.subplot(111)
    cm = plt.cm.get_cmap('viridis_r')
    im = ax.imshow(cnt, cmap=cm, vmin=0, vmax=1)
    # im = ax.imshow(cnt)
    xticks = range(0, detail_cnt+1, 5)
    xlabels = [el for el in range(0, detail_cnt+1, 5)]
    # xlabels[len(xlabels)-1] = '10+'
    ylabels = [detail_cnt - el for el in range(0, detail_cnt+1, 5)]
    # ylabels[0] = '10+'
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xticklabels(xlabels, font_axis)
    ax.set_yticklabels(ylabels, font_axis)
    ax.set_ylabel('Prediction Order', font_axis)
    ax.set_xlabel('Actual Order', font_axis)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # Loop over data dimensions and create text annotations.
    # for i in range(detail_cnt):
    #     for j in range(detail_cnt):
    #         if(cnt[i][j] > 1):
    #             text = ax.text(j, i, int(cnt[i][j]),
    #                        ha="center", va="center", color="w")
    #         elif(cnt[i][j] > 0):
    #             text = ax.text(j, i, cnt[i][j],
    #                        ha="center", va="center", color="w")


    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Percent in Column, %', fontdict=font_axis)
    cbar.ax.tick_params(labelsize=font_axis['size']) 
    plt.subplots_adjust(bottom=0.11, right=0.96, left=0.00, top=0.97)

    plt.show()

if __name__ == '__main__':
    size = 50
    volume = np.random.rand(size, size, size)
    # print(volume.shape, volume[0:size][1][0:size].shape, volume[0].shape)
    # print(np.sum(volume[0]), np.sum(volume[0:size][1][0:size]), np.sum(volume[0:size][0:size][1]))
    # print(volume, volume[0:size][0][0:size], volume[0][0:size][0:size])
    # print(volume)
    # print(volume[0, 0, 0:size], volume[0, 0:size, 0])
    # print(volume[0][0][0:size], volume[0][0:size][0], volume[0][0][0])
    max_sum = 0
    for i in range(size):
        if np.sum(volume[0:size, i, 0:size]) > max_sum:
            cnt[0:size][0:size] = volume[0:size, i, 0:size]
            max_sum = np.sum(volume[0:size, i, 0:size])
            print(1, i, max_sum, np.sum(cnt))
        if np.sum(volume[i, 0:size, 0:size]) > max_sum:
            cnt[0:size][0:size] = volume[i, 0:size, 0:size]
            max_sum = np.sum(volume[i, 0:size, 0:size])
            print(0, i, max_sum, np.sum(cnt))
        if np.sum(volume[0:size, 0:size, i]) > max_sum:
            cnt[0:size][0:size] = volume[0:size, 0:size, i]
            max_sum = np.sum(volume[0:size, 0:size, i])
            print(2, i, max_sum, np.sum(cnt))
    print(max_sum)
    main()
