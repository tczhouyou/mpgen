import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from models.mdgan import cMDGAN
from mp.vmp import VMP
import sys
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from experiments.obs_avoid_envs import ObsExp

parser = OptionParser()
parser.add_option("-s", "--save", action="store_true", dest="save_image", default=False)
(options, args) = parser.parse_args(sys.argv)


goals = np.loadtxt('data/modelCollapse_goals.csv', delimiter=',')
vmps = np.loadtxt('data/modelCollapse_weights.csv', delimiter=',')

n_data = np.shape(vmps)[0]

# generate exp
num_test = 9
obsExp = ObsExp(exp_name='goAroundObs')
nmodel = obsExp.nmodel
start = obsExp.start
colors = obsExp.colors
obs = obsExp.obs
testgoals = obsExp.gen_test(num_test=num_test)
knum = np.shape(vmps)[1]
dim = np.shape(goals)[1]
vmp = VMP(2, kernel_num=int(knum / dim), use_outrange_kernel=False)

train_goals = np.copy(goals)
train_ws = np.copy(vmps)

fig, axe = plt.subplots(nrows=1, ncols=1)

nn_structure = {'generator': [40,40],
                'discriminator': [10],
                'lambda': [10], 'd_response': [40,5], 'd_context': [10,5]}

mdgan = cMDGAN(n_comps=2, context_dim=2, response_dim=20, noise_dim=1, nn_structure=nn_structure)

mdgan.gen_lrate = 0.0002
mdgan.dis_lrate = 0.0002
mdgan.entropy_ratio = 0.0

mdgan.create_network(num_real_data=n_data)
mdgan.init_train()

train_input = np.random.uniform(low=np.min(train_goals, axis=0), high=np.max(train_goals, axis=0),
                                size=(1000, np.shape(train_goals)[1]))
mdgan.train(train_context=train_input, real_context=train_goals, real_response=train_ws, max_epochs=50000, is_load=False, is_save=False)

wout = mdgan.generate(testgoals)
for k in range(np.shape(wout)[0]):
    axe.set_ylim([-2, 2])
    axe.set_xlim([-2.5, 2.5])
    obs.plot(axe)
    cgoals = testgoals[k, :]
    vmp.set_weights(wout[k,:])
    traj = vmp.roll(start, cgoals)
    axe.plot(traj[:, 1], traj[:, 2], '-', color='r', linewidth=2)
    axe.plot(cgoals[0], cgoals[1], 'bo')

plt.draw()
plt.show()

