import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')
from models.mdgan import cMDGAN
from models.mdnmp import MDNMP
import sys
from optparse import OptionParser
import numpy as np
from sklearn.model_selection import train_test_split
from experiments.evaluate_exps import evaluate_docking
import matplotlib.pyplot as plt
from run_mdgan_for_docking import train_evaluate_mdgan_for_docking
from run_mdnmp_for_docking import train_evaluate_mdnmp_for_docking

parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=3)
parser.add_option("-n", "--num_exp", dest="expnum", type="int", default=1)
(options, args) = parser.parse_args(sys.argv)

queries = np.loadtxt('data/docking_queries.csv', delimiter=',')
vmps = np.loadtxt('data/docking_weights.csv', delimiter=',')
starts = np.loadtxt('data/docking_starts.csv', delimiter=',')
goals = np.loadtxt('data/docking_goals.csv', delimiter=',')

wtest = np.expand_dims(vmps, axis=1)
cc, successId = evaluate_docking(wtest, queries, starts,goals)
data = np.concatenate([queries, starts, goals], axis=1)
data = data[successId, :]
vmps = vmps[successId,:]
knum = np.shape(vmps)[1]
rstates = np.random.randint(0, 100, size=options.expnum)

# create mdgan
mdgan_struct = {'generator': [40], 'discriminator': [20], 'lambda': [10], 'd_response': [40,5], 'd_context': [10,5]}
mdgan = cMDGAN(n_comps=options.nmodel, context_dim=6, response_dim=knum, noise_dim=1, nn_structure=mdgan_struct)
mdgan.gen_lrate = 0.00025
mdgan.dis_lrate = 0.0002

# create mdnmp
mdnmp_struct = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [60],
                'scale_layers': [60],
                'mixing_layers': [60]}
mdnmp = MDNMP(n_comps=options.nmodel, d_input=6, d_output=knum, nn_structure=mdnmp_struct, scaling=1)


# start experiment
tsize = [0.9, 0.7, 0.5, 0.3, 0.1]
num_train_data = []

mdgan_res = np.zeros(shape=(options.expnum, len(tsize)))
omdnmp_res = np.zeros(shape=(options.expnum, len(tsize)))
emdnmp_res = np.zeros(shape=(options.expnum, len(tsize)))

for i in range(len(tsize)):
    tratio = tsize[i]
    for expId in range(options.expnum):
        trdata, tdata, trvmps, tvmps = train_test_split(data, vmps, test_size=tratio, random_state=rstates[expId])
        num_train_data.append(np.shape(trdata)[0])
        print("======== exp: %1d for training dataset: %1d =======" % (expId, np.shape(trdata)[0]))

        trqueries = trdata[:, 0:6]
        print(">>>> train MD-GAN ")
        # train and test mdgan
        mdgan_res[expId, i] = train_evaluate_mdgan_for_docking(mdgan, trqueries, trvmps, tdata, max_epochs=50000)

        print(">>>> train original MDN")
        # train and test original mdnmp
        omdnmp_res[expId, i] = train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata, False, max_epochs=20000)

        print(">>>> train entropy MDN")
        # train and test entropy mdnmp
        emdnmp_res[expId, i] = train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata, True, max_epochs=20000)

mdgan = np.mean(mdgan_res, axis=0)
omdnmp = np.mean(omdnmp_res, axis=0)
emdnmp = np.mean(emdnmp_res, axis=0)

model_names = ['MD-GAN', 'Orig MDN', 'Entropy MDN']
x = np.arange(len(num_train_data))
fig, ax = plt.subplots()
width = 0.35
rects1 = ax.bar(x - width, mdgan, width, label=model_names[0])
rects2 = ax.bar(x, omdnmp, width, label=model_names[1])
rects3 = ax.bar(x + width, emdnmp, width, label=model_names[2])

ax.set_ylabel('Success Rate - Docking')
ax.set_title('Training Data Size')
ax.set_xticks(x)
ax.set_xticklabels(num_train_data)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % (height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
fig.tight_layout()
plt.show()