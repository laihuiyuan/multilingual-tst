# -*- coding:utf-8 _*-

import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus']=False
plt.rc('font', family='Times New Roman', size=16)

fig, ax = plt.subplots(figsize=(8, 4.8))
plt.grid(True, ls='--')
plt.ylabel('BLEU Score')
plt.xlim(0.0, 9.5)
plt.ylim(0.0, 0.6)
ax1=ax.twinx()
plt.ylim(0.0, 0.6)
plt.ylabel('COMET Score')

data = [
    [0.299, 0.466, 0.455, 0.329, 0.557, 0.521],
    [0.158, 0.276, 0.270, 0.044, 0.259, 0.241],
    [0.158, 0.237, 0.221, 0.231, 0.305, 0.276],
    [0.298, 0.345, 0.351, 0.479, 0.505, 0.493]
]

x=[0.5+1.4*i if i< 3 else 0.5+1.55*i for i in range(6)]

b1 = plt.bar(x, data[0], width=0.25, label='informal-to-formal-(a)')
b2 = plt.bar([i + 0.25 for i in x], data[1], width=0.25, label='informal-to-formal-(b)')
b3 = plt.bar([i + 0.5 for i in x], data[2], width=0.25, label='formal-to-informal-(a)')
b4 = plt.bar([i + 0.75 for i in x], data[3], width=0.25, label='formal-to-informal-(b)')

plt.xticks([i + 0.375 for i in x],
           ['INPUT', 'BART', 'mBART', 'INPUT', 'BART', 'mBART'])

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }
plt.subplots_adjust(left=0.085, right=0.92, top=0.97, bottom=0.08)

plt.xticks(rotation=40)

plt.legend(loc='upper left', ncol=2, borderpad=0.1, handlelength=1.5, handleheight=0.5, prop=font1)
plt.savefig('a.png', dpi=500)

