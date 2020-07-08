import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import sys

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % (height),
                    xy=(rect.get_x() + rect.get_width() / 2, 0.01),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#'results_hitball_v3_81','results_hitball_v3_82',
dirnames = ['results_hitball_v3_83','results_hitball_v3_84', 'results_hitball_v3_85']

tnames = ['50']#['50', '100', '150']
fnames = ['original_mdn', 'entropy_mdn']#['baselines', 'original_mdn', 'entropy_mdn', 'eub_mdn']

fig, ax = plt.subplots()
width = 0.35
blank = 0.2
npos = (len(tnames)-1) * (len(fnames) * 0.35 + blank)
rects = []

for i in range(len(fnames)):
    ds = []
    for j in range(len(dirnames)):
        cdata = np.loadtxt(dirnames[j] + '/' + fnames[i], delimiter=',')
        if cdata.ndim == 1:
            cdata = np.expand_dims(cdata, axis=1)

        ds.append(cdata)

    data = np.concatenate(ds, axis=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = np.linspace(i*width, npos+i*width, len(tnames))
    for j in range(len(tnames)):
        rect = ax.bar(x[j], mean[j], width, yerr=std[j], label=fnames[i])
        autolabel(rect)

x = np.linspace(0, npos, len(tnames))
ax.set_ylabel('Success Rate')
ax.set_xlabel('Training Data Size')
ax.set_xticks(x)
ax.set_xticklabels(tnames)
fig.tight_layout()
plt.show()
