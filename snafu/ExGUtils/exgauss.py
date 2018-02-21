#    Copyright (C) 2012 Daniel Gamermann <gamermann@gmail.com>
#
#    This file is part of ExGUtils
#
#    ExGUtils is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    ExGUtils is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with ExGUtils.  If not, see <http://www.gnu.org/licenses/>.
#
#    
#    Please, cite us in your reasearch!
#


from math import log, pi, sin, cos, acos, atan, sqrt
from numpy import exp
from scipy.special import erf
from ExGUtils.stats import histogram, stats
from ExGUtils.nummath import zero, fitter, integral



##################################
## EX Gaussian Distr
##################################


def exgauss(x,mu,sig,tau):
    """
       Exgaussian function from parameters mu sig and tau 
    """
    arg1=2.*mu+(sig**2)/tau-2.*x
    arg2=mu+(sig**2)/tau-x
    sq2=sqrt(2.)
    bla1=(0.5/tau)*exp((0.5/tau)*arg1)
    bla2=1.-erf(arg2/(sq2*sig))
    return bla1*bla2

def exg_lamb(z,lamb):
    """
       Exgaussian function from asymmetry parameter lambda
    """
    arg1 = -2.*z*lamb-3*lamb**2+1
    arg1 /= (2*lamb**2)
    arg2 = -z + 1./lamb - 2.*lamb
    arg2 /= (1.-lamb**2)**.5
    sq2 = 2.**.5
    bla1 = exp(arg1)
    bla2 = 1.-erf(arg2/sq2)
    return bla1*bla2/(2.*lamb)

def pars_to_stats(mu,sig,tau):
    """
    converts mu sig and tau into M (average), S (standard deviation) and
                                                                  t (skewness)
    """
    M = mu+tau
    S = (sig**2+tau**2)**.5
    lamb = tau/S
    return [M, S, lamb]

def stats_to_pars(M, S, lamb):
    """
    converts M S and lambda  into mu sig and tau.
    """
    mu = M - S*lamb
    sig = S*(1-lamb**2)**.5
    tau = S*lamb
    return [mu, sig, tau]


def fit_exgauss(lista, Nint=None):
    """
    Fits an exgauss distribution to the values in lista.
    """
    AA = len(lista)
    [M, S] = stats(lista)
    lamb = 0.9
    [mu, sig, tau] = stats_to_pars(M, S, lamb)
    if not Nint:
        [XX, YY] = histogram(lista)
    else:
        [XX, YY] = histogram(lista, Nint=Nint)
    tofit = lambda p, x: p[3]*exgauss(x, p[0], p[1], p[2])
    [mu, sig, tau, AA] = fitter(tofit, XX, YY, [mu, sig, tau, AA])
    return [mu, sig, tau, AA]


def exgauss_lt(z, mu, sig, tau, eps=1.e-9):
    """
    Left tail for the exgaussian distribution given in terms
    of the mu, sig and tau parameters.
    """
    inf = -abs(z)*4
    toint = lambda x: exgauss(x, mu, sig, tau)
    while abs(toint(inf))>eps/10.:
        inf *= 2
    NN = 20
    inte = integral(toint, inf, z, Nints=NN)
    ninte = integral(toint, inf, z, Nints=NN*2)
    while abs(ninte-inte)>eps:
        inte = ninte
        NN *= 2
        ninte = integral(toint, inf, z, Nints=NN*2)
    return ninte


def exg_lamb_lt(z, lamb, eps=1.e-9):
    """
    Left tail for the exgaussian distribution with zero average, 
    standard deviation equal to one and lamb assymetry.
    """
    inf = -4.
    while abs(exg_lamb(inf, lamb))>eps/10:
        inf *= 2
    NN = 20
    toint = lambda x: exg_lamb(x, lamb)
    inte = integral(toint, inf, z, Nints=NN)
    ninte = integral(toint, inf, z, Nints=NN*2)
    while abs(ninte-inte)>eps:
        inte = ninte
        NN *= 2
        ninte = integral(toint, inf, z, Nints=NN*2)
    return ninte

def zalp_exgauss(alp, mu, sig, tau, eps=1.e-9):
    """
    Finds the value x_alpha which has a right tail equal to alpha
    """
    inf = -4.
    while abs(exgauss(inf, mu, sig, tau))>eps/10:
        inf *= 2
    tozeroth = lambda x, NN: integral(lambda y: exgauss(y, mu, sig, tau), inf, x, Nints=NN) - 1.+alp
    Ni = 20
    z1 = zero(lambda x: tozeroth(x, Ni), mu+tau, eps=eps)
    z2 = zero(lambda x: tozeroth(x, 2*Ni), mu+tau, eps=eps)
    while abs(z1-z2)>eps:
        z1 = z2
        Ni *= 2
        z2 = zero(lambda x: tozeroth(x, 2*Ni), mu+tau, eps)
    return z2
    
def zalp_exg_lamb(alp, lamb, eps=1.e-9):
    """
    Finds the value z_alpha which has a right tail equal to alpha
    """
    inf = -4.
    while abs(exg_lamb(inf, lamb))>eps/10:
        inf *= 2
    tozeroth = lambda x, NN: integral(lambda y: exg_lamb(y, lamb), inf, x, Nints=NN) - 1.+alp
    Ni = 20
    z1 = zero(lambda x: tozeroth(x, Ni), 0., eps=eps)
    z2 = zero(lambda x: tozeroth(x, 2*Ni), 0., eps=eps)
    while abs(z1-z2)>eps:
        z1 = z2
        Ni *= 2
        z2 = zero(lambda x: tozeroth(x, 2*Ni), 0., eps)
    return z2
    




