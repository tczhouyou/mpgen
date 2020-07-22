import numpy as np


def get_training_data_from_2d_grid(ndata_for_each_dim, queries):
    minq = np.min(queries, axis=0)
    maxq = np.max(queries, axis=0)

    x = np.linspace(minq[0], maxq[0], ndata_for_each_dim)
    y = np.linspace(minq[1], maxq[1], ndata_for_each_dim)

    xv, yv = np.meshgrid(x, y)

    results = []
    ids = []
    for i in range(np.shape(xv)[0]):
        for j in range(np.shape(yv)[0]):
            deq = np.array([xv[i,j], yv[i,j]])
            ndata = np.shape(queries)[0]
            deqmat = np.tile(deq, (ndata, 1))
            idx = np.argmin(np.linalg.norm(deqmat - queries, axis=1))
            results.append(queries[idx,:])
            ids.append(idx)
            queries = np.delete(queries, idx, 0)

    results = np.stack(results)
    return results, ids


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    queries = np.loadtxt('mujoco/hitball/hitball_mpdata_v2/hitball_queries.csv', delimiter=',')
    weights = np.loadtxt('mujoco/hitball/hitball_mpdata_v2/hitball_weights.csv', delimiter=',')
    _, ids = get_training_data_from_2d_grid(7, queries)
    trqueries = queries[ids,:]
    trweights = weights[ids,:]
    plt.plot(queries[:,0], queries[:,1], 'ro')
    for i in range(np.shape(trqueries)[0]):
        if np.min(trweights[i,0]) < 0:
            plt.plot(trqueries[i,0], trqueries[i,1], 'bo')
        else:
            plt.plot(trqueries[i,0], trqueries[i,1], 'go')


    plt.show()