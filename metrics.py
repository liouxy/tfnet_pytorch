import numpy as np
from scipy.ndimage import sobel
from numpy.linalg import norm

def sam(ms, ps):

    assert ms.ndim == 3 and ms.shape == ps.shape
    dot_sum = np.sum(ms * ps, axis=2)
    norm_true = norm(ms, axis=2)
    norm_pred = norm(ps, axis=2)

    res = np.arccos(dot_sum/norm_pred/norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x,y) in zip(is_nan[0], is_nan[1]):
        res[x,y]=0

    sam = np.mean(res)
    return sam

def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')


    return  (np.sum(ps_sobel*ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel)))


