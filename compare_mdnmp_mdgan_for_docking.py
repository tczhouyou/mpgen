import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')
from models.mdgan import cMDGAN
from models.mdnmp import MDNMP
from models.gmgan import GMGAN
import sys
from optparse import OptionParser
import numpy as np
from sklearn.model_selection import train_test_split
from experiments.evaluate_exps import evaluate_docking
import matplotlib.pyplot as plt
from run_mdgan_for_docking import train_evaluate_mdgan_for_docking
from run_mdnmp_for_docking import train_evaluate_mdnmp_for_docking
from run_gmgan_for_docking import train_evaluate_gmgan_for_docking

parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=3)
parser.add_option("-n", "--num_exp", dest="expnum", type="int", default=1)
parser.add_option("-d", "--result_dir", dest="result_dir", type="string", default="results_compare_docking")
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
mdgan_struct = {'generator': [40], 'discriminator': [20], 'lambda': [20], 'd_response': [40,10], 'd_context': [20]}
mdgan = cMDGAN(n_comps=options.nmodel, context_dim=6, response_dim=knum, noise_dim=1, nn_structure=mdgan_struct)
mdgan.gen_lrate = 0.0005
mdgan.dis_lrate = 0.0005


# create mdnmp
mdnmp_struct = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [60],
                'scale_layers': [60],
                'mixing_layers': [60]}
mdnmp = MDNMP(n_comps=options.nmodel, d_input=6, d_output=knum, nn_structure=mdnmp_struct, scaling=1)

# create gmgan
nn_structure = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [60],
                'scale_layers': [60],
                'mixing_layers': [60],
                'discriminator': [20],
                'lambda': [10],
                'd_response': [40,5],
                'd_context': [10,5]}
gmgan = GMGAN(n_comps=options.nmodel, context_dim=6, response_dim=knum, nn_structure=nn_structure)


# start experiment
num_train_data = np.array([50, 100, 150])
tsize = (np.shape(data)[0] -num_train_data)/np.shape(data)[0]
print(tsize)

result_dir = options.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for expId in range(options.expnum):
    # omdgan_res = np.zeros(shape=(1, len(tsize)))
    # emdgan_res = np.zeros(shape=(1, len(tsize)))
    omdnmp_res = np.zeros(shape=(1, len(tsize)))
    emdnmp_res = np.zeros(shape=(1, len(tsize)))
    ogmgan_res = np.zeros(shape=(1, len(tsize)))
    egmgan_res = np.zeros(shape=(1, len(tsize)))

    for i in range(len(tsize)):
        tratio = tsize[i]
        trdata, tdata, trvmps, tvmps = train_test_split(data, vmps, test_size=tratio, random_state=rstates[expId])
        print("======== exp: %1d for training dataset: %1d =======" % (expId, np.shape(trdata)[0]))

        trqueries = trdata[:, 0:6]

        # print(">>>> train entropy MD-GAN ")
        # # train and test mdgan
        # emdgan_res[0, i] = train_evaluate_mdgan_for_docking(mdgan, trqueries, trvmps, tdata, True, max_epochs=300000)
        #
        
        # print(">>>> train original MD-GAN ")
        # # train and test mdgan
        # omdgan_res[0, i] = train_evaluate_mdgan_for_docking(mdgan, trqueries, trvmps, tdata, False, max_epochs=300000)
        print(">>>> train entropy GMGANs")
        egmgan_res[0, i] = train_evaluate_gmgan_for_docking(gmgan, trqueries, trvmps, tdata, True, max_epochs=20000, sup_max_epoch=20001)

        print(">>>> train normal GMGANs")
        ogmgan_res[0, i] = train_evaluate_gmgan_for_docking(gmgan, trqueries, trvmps, tdata, False, max_epochs=20000, sup_max_epoch=20001)

        print(">>>> train entropy MDN")
        emdnmp_res[0, i] = train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata, True, max_epochs=20000)

        print(">>>> train original MDN")
        omdnmp_res[0, i] = train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata, False, max_epochs=20000)

    # with open("results_compare_docking/original_mdgan", "a") as f:
    #     np.savetxt(f, np.array(omdgan_res), delimiter=',', fmt='%.3f')
    # with open("results_compare_docking/entropy_mdgan", "a") as f:
    #     np.savetxt(f, np.array(emdgan_res), delimiter=',', fmt='%.3f')
    with open(result_dir + "/normal_gmgan", "a") as f:
        np.savetxt(f, np.array(ogmgan_res), delimiter=',', fmt='%.3f')
    with open(result_dir + "/entropy_gmgan", "a") as f:
        np.savetxt(f, np.array(egmgan_res), delimiter=',', fmt='%.3f')
    with open(result_dir + "/original_mdn", "a") as f:
        np.savetxt(f, np.array(omdnmp_res), delimiter=',', fmt='%.3f')
    with open(result_dir + "/entropy_mdn", "a") as f:
        np.savetxt(f, np.array(emdnmp_res), delimiter=',', fmt='%.3f')


# model_names = ['Orig MD-GAN','Entropy MD-GAN', 'Orig MDN', 'Entropy MDN']
# x = np.arange(len(num_train_data))
# fig, ax = plt.subplots()
# width = 0.35
# rects1 = ax.bar(x - 2 * width, omdgan, width, label=model_names[0])
# rects2 = ax.bar(x - width, emdgan, width, label=model_names[1])
# rects3 = ax.bar(x + width, omdnmp, width, label=model_names[2])
# rects4 = ax.bar(x + 2 * width, emdnmp, width, label=model_names[3])
#
# ax.set_ylabel('Success Rate - Docking')
# ax.set_title('Training Data Size')
# ax.set_xticks(x)
# ax.set_xticklabels(num_train_data)
# ax.legend()
#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('%.2f' % (height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# fig.tight_layout()
# plt.show()