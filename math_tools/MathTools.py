import math
import numpy as np
from sklearn import mixture
from scipy.stats import multivariate_normal


def mat2rpy(R):
    a = ((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2
    a = max(a,-1)
    a = min(a,1)

    theta = np.arccos(a)
    if theta == 0:
        return np.zeros(3)
    else:
        multi = 1 / (2 * math.sin(theta))

        rpy = np.zeros(3)

        rpy[0] = multi * (R[2, 1] - R[1, 2]) * theta
        rpy[1] = multi * (R[0, 2] - R[2, 0]) * theta
        rpy[2] = multi * (R[1, 0] - R[0, 1]) * theta
        return rpy.copy()


def rpy2mat(rpy):
    m = np.zeros((3, 3))
    sgamma = np.sin(rpy[0])
    cgamma = np.cos(rpy[0])
    sbeta = np.sin(rpy[1])
    cbeta = np.cos(rpy[1])
    salpha = np.sin(rpy[2])
    calpha = np.cos(rpy[2])

    m[0, 0] = calpha * cbeta
    m[0, 1] = calpha * sbeta * sgamma - salpha * cgamma
    m[0, 2] = calpha * sbeta * cgamma + salpha * sgamma

    m[1, 0] = salpha * cbeta
    m[1, 1] = salpha * sbeta * sgamma + calpha * cgamma
    m[1, 2] = salpha * sbeta * cgamma - calpha * sgamma

    m[2, 0] = - sbeta
    m[2, 1] = cbeta * sgamma
    m[2, 2] = cbeta * cgamma
    return m


def draw_contour(ax, gmlist, X = np.linspace(-5, 15, 100), Y = np.linspace(-5, 15, 100), color=None):
    mean = gmlist['mean'][0]
    scale = gmlist['scale'][0]
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    i = 0
    while i < len(mean):
        mu = np.array([mean[i], mean[i+1]])
        sig = np.array([[scale[i],0], [0,scale[i+1]]])
        F = multivariate_normal(mu, sig)
        Z = F.pdf(pos)
        Zmax = np.amax(Z)
        if color is not None:
            ax.contour(X,Y,Z, levels=[Zmax/4,Zmax/3,Zmax/2], colors=[color])
        else:
            ax.contour(X,Y,Z, levels=[Zmax/4,Zmax/3,Zmax/2])

        i = i + 2


def draw_contour_gmm(ax, gmm, X=np.linspace(-5,15,60), Y=np.linspace(-5,15,60), color=None, linestyle='solid', linewidths=1.0):
    means = gmm.means_
    scales = gmm.covariances_

    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    for i in range(np.shape(means)[0]):
        mu = means[i,:]
        scale = scales[i,:]
        sig = np.array([[scale[0],0], [0,scale[1]]])
        F = multivariate_normal(mu, sig)
        Z = F.pdf(pos)
        Zmax = np.amax(Z)
        if color is not None:
            ax.contour(X,Y,Z, levels=[Zmax/5,Zmax/2,Zmax/1.2], colors=[color], linestyles=linestyle, linewidths=linewidths)
        else:
            ax.contour(X,Y,Z, levels=[Zmax/5,Zmax/2,Zmax/1.2], linestyles=linestyle, linewidths=linewidths)


def mdn_to_gmm(outdict, fdim=2, ind=0):
    if ind >= np.shape(outdict['mean'])[0]:
        return None

    mc = outdict['mc'][ind]
    n_comp = len(mc)
    mean = np.reshape(outdict['mean'][ind], newshape=(-1,fdim))
    scale = np.reshape(outdict['scale'][ind], newshape=(-1,fdim))

    gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type='diag')
    gmm.fit(np.random.uniform(-1.0, 1.0, (10, fdim)))
    gmm.means_ = mean
    gmm.covariances_ = np.square(scale)
    gmm.weights_ = mc
    return gmm


def calc_kl_mc(gmm0, gmm1, n_data=1e1):
    samples, _ = gmm0.sample(n_data)
    logprob0 = gmm0.score_samples(samples)
    logprob1 = gmm1.score_samples(samples)
    kl = np.mean(logprob0 - logprob1)
    return kl


