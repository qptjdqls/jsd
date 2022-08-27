import numpy as np
import scipy.stats as st
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from tqdm import trange


def js_divergence(*args):
    if len(args) == 3:
        return distributions_js(*args)
    else:
        return distributions_js_(*args)


def distributions_js(distribution_p, distribution_q, n_samples):
    '''
    original code:
    https://stats.stackexchange.com/questions/345915/trying-to-implement-the-jensen-shannon-divergence-for-multivariate-gaussians/419421#419421
    '''
    # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
    # all the logarithms are defined as log2 (because of information entrophy)
    X = distribution_p.rvs(n_samples)
    p_X = distribution_p.pdf(X)
    q_X = distribution_q.pdf(X)
    dim = q_X.shape[1]
    q_X = np.prod(q_X, axis=1)
    log_mix_X = np.log2(p_X + q_X)

    Y = distribution_q.rvs(size=(n_samples, dim))
    p_Y = distribution_p.pdf(Y)
    q_Y = distribution_q.pdf(Y)
    q_Y = np.prod(q_Y, axis=1)
    log_mix_Y = np.log2(p_Y + q_Y)

    return (np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
            + np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))) / 2


def distributions_js_(distribution_p, distribution_q, distribution_q_, dim, n_samples):
    '''
    original code:
    https://stats.stackexchange.com/questions/345915/trying-to-implement-the-jensen-shannon-divergence-for-multivariate-gaussians/419421#419421
    '''
    # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
    # all the logarithms are defined as log2 (because of information entrophy)
    X = distribution_p.rvs(n_samples)
    p_X = distribution_p.pdf(X)
    q_X = distribution_q.pdf(X[:,:dim])
    q_X_ = distribution_q_.pdf(X[:,dim])
    # dim = q_X.shape[1] + 1
    q_X = np.prod([q_X, q_X_], axis=0)
    log_mix_X = np.log2(p_X + q_X)

    Y = distribution_q.rvs(size=n_samples)
    Y_ = distribution_q_.rvs(size=n_samples)
    if Y.ndim == 1:
        Y = Y[...,np.newaxis]
    Y = np.concatenate([Y, Y_[...,np.newaxis]], axis=1)
    p_Y = distribution_p.pdf(Y)
    q_Y = distribution_q.pdf(Y[:,:dim])
    q_Y_ = distribution_q_.pdf(Y[:,dim])
    q_Y = np.prod([q_Y, q_Y_], axis=0)
    log_mix_Y = np.log2(p_Y + q_Y)

    return (np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
            + np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))) / 2

'''
parameters
'''
MIN = 0
MAX = 1
# # input dimension
# dim = 4
dim_start = 3
dim_end = 4
n_samples = 10**5
n_trials = 100


mean1_list = []
mean2_list = []
var1_list = []
var2_list = []
for dim in range(dim_start, dim_end):
    d2_min_list = []
    d2_max_list = []
    d1_list = []
    d2_list = []
    d2_mean_list = []
    diff1_list = []
    diff2_list = []
    triangle_count = 0
    mean_count = 0
    max_count = 0
    order_count = 0
    for i in trange(n_trials):
        d2_min = MAX
        d2_max = MIN
        # generate mean/covariance matrix
        mean = np.random.randint(-dim, dim, size=dim)
        # mean = np.random.randint(-100, 100, size=dim)
        # mean = np.random.randn(dim) * 0.01
        cov = make_spd_matrix(dim)

        # p(x,y,z) vs p(x)p(y)p(z)
        p_joint = st.multivariate_normal(mean, cov)
        p_marginal = st.norm(loc=mean, scale=np.diag(cov))
        d1 = js_divergence(p_joint, p_marginal, n_samples) ** (0.5)        # sqrt

        # p(x,y,z) vs p(x,y)p(z) ...
        d2 = 0
        d2_prev = 0
        d2_cur = 0
        p_prev = p_joint
        for j in range(dim-1, 0, -1):
            pa = st.multivariate_normal(mean[:j], cov[:j,:j])
            pb = st.norm(loc=mean[j], scale=np.diag(cov)[j])
            # d2 += distributions_js_(p_prev, pa, pb, j)
            d2_ = js_divergence(p_prev, pa, pb, j, n_samples) ** (0.5)        # sqrt

            d2_cur = d2_
            if d2_cur < d2_prev:
                # print("order count: " + str(order_count))
                order_count += 1
            d2_prev = d2_cur

            d2_max = max(d2_max, d2_)
            d2_min = min(d2_min, d2_)
            d2 += d2_
            p_prev = pa

        d2_max_list.append(d2_max)
        d2_min_list.append(d2_min)
        d1_list.append(d1)
        d2_list.append(d2)
        d2_mean_list.append(d2/(dim-1))
        diff1_list.append(d2-d1)
        diff2_list.append(d1-d2/(dim-1))

        if d1 > d2:
            triangle_count += 1
            # print("triangle count: " + str(triangle_count))
        if d1 < d2/(dim-1):
            mean_count += 1
            # print("mean count: " + str(mean_count))
        if d2_max > d1:
            max_count += 1
            # print("max count: " + str(max_count))

    mean1, var1 = np.mean(diff1_list), np.var(diff1_list)
    mean2, var2 = np.mean(diff2_list), np.var(diff2_list)
    mean1_list.append(mean1)
    mean2_list.append(mean2)
    var1_list.append(var1)
    var2_list.append(var2)

    print("dim: %d, order_count: %d, triangle_count: %d, mean_count: %d, max_count: %d, mean1: %f, var1: %f, mean2: %f, var2: %f \n"
        % (dim, order_count, triangle_count, mean_count, max_count, mean1, var1, mean2, var2))

    t = np.arange(0, n_trials, 1)
    plt.plot(t, d2_mean_list, t, d1_list, t, d2_list, t, d2_max_list, t, d2_min_list)
    plt.legend(['d2_mean', 'd1', 'd2', 'd2_max', 'd2_min'])
    # plt.show()
    plt.tight_layout()
    plt.savefig('jsd'+str(dim)+'.png')
    plt.cla()

d = np.arange(dim_start, dim_end)
plt.plot(d, mean1_list, d, var1_list, d, mean2_list, d, var2_list)
plt.legend(['mean1', 'var1', 'mean2', 'var2'])
# plt.show()
plt.tight_layout()
plt.savefig('mean+var.png')
plt.cla()