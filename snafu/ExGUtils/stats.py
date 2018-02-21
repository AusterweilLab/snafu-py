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


from random import random as rand
from random import gauss
from scipy.stats.distributions import f as fsnede
from math import log



####################################
# Statistical stuff ################
####################################

def stats(lista):
    """
        Calculates the average and dispersion of the values in lista.
    """
    N=float(len(lista))
    xmed=sum(lista)/N
    if N == 1:
        return [xmed, -1]
    s2=[(xmed-ele)**2 for ele in lista]
    s=(sum(s2)/(N-1.))**.5
    return [xmed,s]    

def stats_int(xi,ni):
    """
        Calculates the average and dispersion of the values in intervals
            (value xi[i] with absolute frequency ni[i]).
    """
    N=float(sum(ni))
    xn = [xi[ii]*ni[ii] for ii in xrange(len(xi))]
    xmed = sum(xn)/N
    s2=[ni[ii]*(xmed-ele)**2 for ii, ele in enumerate(xi)]
    s=(sum(s2)/(N-1.))**.5
    return [xmed,s]    


def rand_exp(tau):
    """ Generates random number with exponential distribution."""
    nrand = rand()
    return -tau*log(1.-nrand)


def rand_exg(mu, sig, tau):
    """ Generates random number with ex-gaussian distribution."""
    nexp = rand_exp(tau)
    ngau = gauss(mu, sig)
    return nexp + ngau


def histogram(lista,ini=None,fin=None,Nint=None, dell=.5, accu=0):
    """
        Returns a list with the histigram of the elements in lista. ini is the
        lowest value, fin the highest and Nint the number of intervals in the 
        histogram.
        dell is a shift for the values in the xxx list.
        accu=0 means histogram made with frequencies, accu=+-1 means accumulated
           frequencies to the right or left.
    """
    if ini==None:
        ini=min(lista)
    if fin==None:
        fin=max(lista)
    fin+=1.e-10
    if Nint==None:
        Nint=2*int(len(lista)**.5)
    fin=float(fin)
    ini=float(ini)
    anch=1.0*(fin-ini)/Nint
    hist=[0. for ele in xrange(Nint)]
    for ele in lista:
        if ele >= ini and ele < fin:
            Int=int((ele-ini)//anch)
            hist[Int]+=1.
    dx=(fin-ini)/Nint
    if accu==-1:
        hist = [sum(hist[ii:]) for ii in xrange(Nint)]
    if accu==1:
        hist = [sum(hist[:ii+1]) for ii in xrange(Nint)]
    xxx=[ini+dx*(ii+dell) for ii in xrange(Nint)]
    return [xxx,hist]



def ANOVA(tab):
    """
    ANOVA test for table tab (tab should be a list of lists).
     Values returned are in order:
       Fs: Value for the variable F (F of snedecor).
       glentre: degrees of freedom in between.
       gldentro: degrees of fredom inside.
       1-fsnede: left tail (p-value for the test).
    """
    r = len(tab)
    ni = [len(ele) for ele in tab]
    xbi = [stats(ele)[0] for ele in tab]
    N = sum(ni)
    XB = sum([ni[ii]*xbi[ii] for ii in xrange(r)])/N
    ssi = [sum([(ele - xbi[ii])**2 for ele in ele2]) for ii, ele2 in enumerate(tab)]
    SSdentro = sum(ssi)
    gldentro = N-r
    MSdentro = SSdentro/gldentro
    SSentre = sum([ni[ii]*(ele-XB)**2 for ii, ele in enumerate(xbi)])
    glentre = r-1
    MSentre = SSentre/glentre
    Fs = MSentre/MSdentro
    return [Fs, glentre, gldentro, 1.-fsnede.cdf(Fs,glentre,gldentro)]



