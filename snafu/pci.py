# This file is used to implement the 95% CI Clopper-Pearson method used by Conceptual Network method.
# Originally, it relied on statsmodels/scipy, which is a very large package
# These functions are not intended for the end user of SNAFU

from . import *

# copied from https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
def _contfractbeta(a,b,x, ITMAX = 200):
    EPS = 3.0e-7
    bm = az = am = 1.0
    qab = a+b
    qap = a+1.0
    qam = a-1.0
    bz = 1.0-qab*x/qap
    for i in range(ITMAX+1):
        em = float(i+1)
        tem = em + em
        d = em*(b-em)*x/((qam+tem)*(a+tem))
        ap = az + d*am
        bp = bz+d*bm
        d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem))
        app = ap+d*az
        bpp = bp+d*bz
        aold = az
        am = ap/bpp
        bm = bp/bpp
        az = app/bpp
        bz = 1.0
        if (abs(az-aold)<(EPS*abs(az))):
            return az
    print ('a or b too large or given ITMAX too small for computing incomplete beta function.')

# copied from https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
# same as scipy.special.betainc within rounding
# normalized incomplete beta is same as beta cdf
def _incomplete_beta(a, b, x):
    if (x == 0):
        return 0;
    elif (x == 1):
        return 1;
    else:
        lbeta = math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1-x)
        if (x < (a+1) / (a+b+2)):
            return math.exp(lbeta) * _contfractbeta(a, b, x) / a;
        else:
            return 1 - math.exp(lbeta) * _contfractbeta(b, a, 1-x) / b;

# implements beta ppf
# same result as stats.beta.ppf(alpha_2, a, b)
def _ppf(alpha_2, a, b, lower=0.0, upper=1.0, span=11, maxiter=20):
    if alpha_2 == 1.0:
        return 1.0
    elif alpha_2 == 0.0:
        return 0.0
    nprange = np.linspace(lower, upper, span)
    highlow = [_incomplete_beta(a, b, x) > alpha_2 for x in nprange]
    idx_of_true = [idx for idx, x in enumerate(highlow) if x == True]
    if len(idx_of_true) == span:
        return lower
    elif len(idx_of_true) == 0:
        return upper
    else:
        if maxiter == 0:
            return nprange[idx_of_true[0]]
        else:
            return _ppf(alpha_2, a, b, lower=nprange[idx_of_true[0]-1], upper=nprange[idx_of_true[0]], maxiter=(maxiter-1))

# same result as stats.beta.ppf(alpha_2, count, nobs - count + 1) (from statsmodels)
def _pci_lowerbound(cooccur, total, alpha):
    alpha_2 = alpha * 0.5
    return _ppf(alpha_2, cooccur, total - cooccur + 1)
