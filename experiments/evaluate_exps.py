import os, inspect
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')

from mp.vmp import VMP
import matplotlib.pyplot as plt
from obs_avoid_envs import ObsExp
import numpy as np


def evaluate_circular_obstacle_avoidance(mpweights, testqueries, teststarts, testgoals, knum=10, isdraw=False, ndraw=4, onlySuccess=False):
    # mpweights: N x dim, N: number of experiments, dim: dimension of MP
    # testqueries: N x qdim, N: number of experiments, qdim: dimension of queries
    # teststarts: N x 2
    # testgoals: N x 2

    success_num = 0
    ndata = np.shape(mpweights)[0]

    vmp = VMP(2, kernel_num=knum)

    obsExp = ObsExp(exp_name="goAroundObsR3V2")
    envs = obsExp.get_envs(testqueries)

    k = 0
    if isdraw:
        if not onlySuccess:
            ids = np.random.choice(ndata, ndraw * ndraw, replace=False)
        else:
            ids = np.linspace(0,ndata,ndata)

        fig, axes = plt.subplots(nrows=ndraw, ncols=ndraw)

    for i in range(ndata):
        w = mpweights[i,:]
        vmp.set_weights(w)
        start = teststarts[i,:]
        goal = testgoals[i,:]
        env = envs[i]
        traj = vmp.roll(start, goal)

        iscollision = False
        for j in range(np.shape(traj)[0]):
            iscollision = env.isCollision([traj[j, 1], traj[j, 2]])
            if iscollision:
                break

        if not iscollision:
            success_num = success_num + 1

        if isdraw:
            if i in ids:
                if not onlySuccess or not iscollision:
                    ax = int(np.floor(k / ndraw))
                    ay = int(k % ndraw)
                    axes[ax, ay].clear()
                    axes[ax, ay].set_ylim([-10, 10])
                    axes[ax, ay].set_xlim([-10, 10])
                    axes[ax, ay].set_yticklabels([])
                    axes[ax, ay].set_xticklabels([])
                    env.plot(axes[ax, ay])
                    k = k + 1

                    if iscollision:
                        axes[ax, ay].plot(traj[:, 1], traj[:, 2], '-.', color='r')
                    else:
                        axes[ax, ay].plot(traj[:, 1], traj[:, 2], '-', color='b')

                    axes[ax,ay].plot(start[0], start[1], 'ro')
                    axes[ax,ay].plot(goal[0], goal[1], 'bo')


    print('success_num: {}, ndata: {}'.format(success_num, ndata))
    if isdraw:
        plt.show()

    return success_num / ndata


def evaluate_docking_for_all_models(mpweights, mfs, testqueries, teststarts, testgoals, knum=10, ndraw = 4):
    ndata = np.shape(mpweights)[0]
    vmp = VMP(2, kernel_num=knum)
    obsExp = ObsExp(exp_name="Docking")
    envs = obsExp.get_envs(testqueries)

    fig, axes = plt.subplots(nrows=ndraw, ncols=ndraw)
    colors = [(232/255,82/255,88/255,1), (1/255,129/255,1/255,1), (4/255,129/255,205/255,1)]#cmap(np.linspace(0,1,nmodel))

    k = 0
    for i in range(ndata):
        ws = mpweights[i,:,:]
        iscollision = False
        for j in range(np.shape(ws)[1]):
            start = teststarts[i, :]
            goal = testgoals[i, :]
            env = envs[i]
            mf = mfs[i,:]
            vmp.set_weights(ws[:,np.argmax(mf)])
            traj = vmp.roll(start, goal)

            if np.linalg.norm(traj[0, 1:] - start) > 1 or np.linalg.norm(traj[-1, 1:] - goal) > 1:
                iscollision = True
                continue

            if np.linalg.norm(testqueries[i, 0:2] - start) >1 or np.linalg.norm(testqueries[i, 3:5] - goal) > 1:
                iscollision = True
                continue

            for j in range(np.shape(traj)[0]):
                iscollision = env.isCollision([traj[j, 1], traj[j, 2]])
                if iscollision:
                    break

        if not iscollision and k < ndraw*ndraw:
            ax = int(np.floor(k / ndraw))
            ay = int(k % ndraw)
            axes[ax, ay].clear()
            axes[ax, ay].set_ylim([-5, 25])
            axes[ax, ay].set_xlim([-5, 25])
            axes[ax, ay].set_yticklabels([])
            axes[ax, ay].set_xticklabels([])
            axes[ax, ay].set_aspect('equal')
            axes[ax, ay].tick_params(length=0)
            env.plot(axes[ax, ay])

            axes[ax, ay].plot(traj[0, 1], traj[0, 2], 'k.')
            axes[ax, ay].plot(traj[-1, 1], traj[-1, 2], 'k.')

            for mid in range(len(mf)):
                vmp.set_weights(ws[:,mid])
                traj = vmp.roll(start, goal)
                if mid == np.argmax(mf):
                    axes[ax, ay].plot(traj[:, 1], traj[:, 2], '-', color=colors[mid], linewidth=2)
                else:
                    axes[ax, ay].plot(traj[:, 1], traj[:, 2], '-.', color=colors[mid], linewidth=2)

            k = k + 1


    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.draw()
    plt.pause(0.001)
    plt.savefig('./docking_img.png')



def evaluate_docking(mpweights, testqueries, teststarts, testgoals, knum=10,mua=None, isdraw=False, ndraw=3, onlySuccess=True):
    # mpweights: N x S x dim, N: number of experiments, S: number of samples, dim: dimension of MP
    # testqueries: N x qdim, N: number of experiments, qdim: dimension of queries
    # teststarts: N x 2
    # testgoals: N x 2
    success_num = 0
    ndata = np.shape(mpweights)[0]

    vmp = VMP(2, kernel_num=knum)

    obsExp = ObsExp(exp_name="Docking")
    envs = obsExp.get_envs(testqueries)

    k = 0
    if isdraw:
        if not onlySuccess:
            ids = np.random.choice(ndata, ndraw * ndraw, replace=False)
        else:
            ids = np.linspace(0,ndata,ndata)

        fig, axes = plt.subplots(nrows=ndraw, ncols=ndraw)

    successId = []
    colors = [(232/255,82/255,88/255,0.5), (1/255,129/255,1/255,0.5), (4/255,129/255,205/255,0.5)]

    for i in range(ndata):
        w = mpweights[i,:]

        for j in range(np.shape(w)[0]):
            vmp.set_weights(w[j,:])
            start = teststarts[i,:]
            goal = testgoals[i,:]
            env = envs[i]
            traj = vmp.roll(start, goal)
            iscollision = False

            if np.linalg.norm(traj[0,1:] - start) > 1 or np.linalg.norm(traj[-1,1:] - goal) >1:
                iscollision = True
                continue

            for j in range(np.shape(traj)[0]):
                iscollision = env.isCollision([traj[j, 1], traj[j, 2]])
                if iscollision:
                    break

            if not iscollision:
                success_num = success_num + 1
                successId.append(i)
                break

        if isdraw:
            if k < ndraw * ndraw:
                if not onlySuccess or not iscollision:
                    ax = int(np.floor(k / ndraw))
                    ay = int(k % ndraw)
                    axes[ax, ay].clear()
                    axes[ax, ay].set_ylim([-5, 25])
                    axes[ax, ay].set_xlim([-5, 25])
                    axes[ax, ay].set_yticklabels([])
                    axes[ax, ay].set_xticklabels([])
                    axes[ax, ay].set_aspect('equal')
                    axes[ax, ay].tick_params(length=0)
                    # axes[ax, ay].axhline(linewidth=2)
                    # axes[ax, ay].axvline(linewidth=2)
                    env.plot(axes[ax, ay])

                    axes[ax, ay].plot(start[0], start[1], 'r.')
                    axes[ax, ay].plot(goal[0], goal[1], 'b.')
                    axes[ax, ay].plot(traj[:, 1], traj[:, 2], 'k-', linewidth=2)

                    if mua is not None:
                        for mid in range(np.shape(mua)[2]):
                            vmp.set_weights(mua[i, :, mid])
                            traj = vmp.roll(start, goal)
                            axes[ax, ay].plot(traj[:, 1], traj[:, 2], '-.', color=colors[mid], linewidth=2)

                    k = k + 1

    print('success_num: %1d, ndata: %1d, success_rate: %.3f' % (success_num, ndata, success_num/ndata))
    if isdraw:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.draw()
        plt.pause(0.001)
        plt.savefig('./docking.png', format='png', dpi=1200)

    return success_num / ndata, successId


