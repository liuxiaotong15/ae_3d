import numpy as np
import matplotlib.pyplot as plt
import time
import re
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-f", "--filename", help="log name", required=True)

args = vars(parse.parse_args())

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

file_name = args['filename']

epoch = []
r_ma = []
r_cur = []

import re
f = open(file_name, "r") #, encoding='utf-8')     #打开test.txt文件，以只读得方式，注意编码格式，含中文
data = f.readlines()                            #循环文本中得每一行，得到得是一个列表的格式<class 'list'>
f.close()                                       #关闭test.txt文件
for line in data:
    # result = re.match('.*INFO: Episode:(.*) Reward: (.*) 2body d is: (.*)', line)
    result = re.match('.*Ep: (.*) \| Ep_r_ma: (.*) \| Ep_r_cur: (.*) \| Ep_r_max: .*', line)
    if(result):
        print(result.group(0))
        print(result.group(1))
        print(result.group(2))
        print(result.group(3))
        epoch.append(float(result.group(1)))
        r_ma.append(float(result.group(2)))
        r_cur.append(float(result.group(3)))
    else:
        pass

########################################################################3

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.scatter(epoch, r_cur, c='tab:blue', label='Reward_SGL',
          alpha=1.0, edgecolors='none')
ax.plot(epoch, r_ma, c='tab:red', label='Reward_MA',
          alpha=1.0)
# ax.scatter(x, y, c='tab:red', label='nn_output',
#            alpha=1.0, edgecolors='none')
ax1 = ax.twinx()
# ax1.scatter(episodes, distances, c='tab:red', label='distances',
#            alpha=1.0, edgecolors='none')

# ax1.scatter(episodes, sess_run, c='tab:red', label='nn_output',
#            alpha=1.0, edgecolors='none')

valid_reward = 0

# xx = [x for x in rewards if x>0]
# print('valid data is(>0 reward):', len(xx))

ymin, ymax = ax.get_ylim()
# ax.set_ylim(min(rewards)*1.2, max(rewards)*1.2)
# ax.set_ylim(0, max(rewards)*1.2)
# ax.spines['left'].set_color('blue')
# ax.tick_params(axis='y', colors='blue')
# ax.yaxis.label.set_color('blue')
# ymin, ymax = ax1.get_ylim()
# ax1.set_ylim(min(rewards)*1.2, max(rewards)*1.2)
ax1.set_ylim(ymin, ymax)
# ax1.spines['right'].set_color('blue')
# ax1.tick_params(axis='y', colors='blue')
# ax1.yaxis.label.set_color('blue')

# ax.scatter(x, y1, c='tab:green', label='success',
#            alpha=1.0, edgecolors='none')

#
#for color in ['tab:blue', 'tab:orange', 'tab:green']:
#    n = 750
#    x, y = np.random.rand(2, n)
#    scale = 200.0 * np.random.rand(n)
#    ax.scatter(x, y, c=color, s=scale, label=color,
#               alpha=0.3, edgecolors='none')

ax.tick_params(labelsize=15)
ax1.tick_params(labelsize=15)
ax.legend(loc='fit', prop=font_legend)
ax.grid(True)

ax.set_ylabel('Reward', font_axis)
ax1.set_ylabel('Reward', font_axis)
ax.set_xlabel('Episodes', font_axis)
# fig.suptitle(str(len(episodes)-min_idx) + ' total data.' )


plt.subplots_adjust(bottom=0.12, right=0.92, left=0.09, top=0.97)
plt.show() 
