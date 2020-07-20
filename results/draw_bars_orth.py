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


parser = OptionParser()
parser.add_option("-d", "--dir", dest='dirname', type='string')
(options, args) = parser.parse_args(sys.argv)

dirname = options.dirname + '/'


tnames = ['50']
#tnames = ['50', '100', '150']
#fnames =['baseline', 'omdn', 'mce', 'omce', 'oelk']
fnames = ['omdn', 'mce', 'omce']

fig, ax = plt.subplots()
width = 0.35
blank = 0.2
npos = (len(tnames)-1) * (len(fnames) * 0.35 + blank)
rects = []
for i in range(len(fnames)):
    fname = fnames[i]
    rawdata = np.loadtxt(dirname+fname, delimiter=',')
    if rawdata.ndim == 1:
        rawdata = np.expand_dims(rawdata, axis=1)

    rawdata = rawdata[np.all(rawdata,axis=1),:]
    rawdata = np.sort(rawdata)
    data = rawdata
#    data = rawdata[1:-2,:]
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
