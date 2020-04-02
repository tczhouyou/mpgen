import numpy as np


def sample_gmm(n_samples, means, scales, mixing_coeffs):
    ws = np.zeros(shape=(n_samples, np.shape(means)[1]))
    n_components = np.shape(mixing_coeffs)[0]
    np.random.seed(seed=10)
    idx = np.random.choice(n_components, size=n_samples, p=mixing_coeffs)

    for i in range(n_samples):
        comp_id = idx[i]
        mu = means[comp_id, :]
        cov = np.diag(np.square(scales[comp_id,:]))
        cw = np.random.multivariate_normal(mu, cov)
        ws[i, :] = cw

    idx[0] = np.argmax(mixing_coeffs)
    ws[0,:] = means[idx[0],:]

    return ws, idx

