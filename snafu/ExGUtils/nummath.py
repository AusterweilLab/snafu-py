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


from numpy import matrix
from numpy import array
from scipy.optimize import leastsq


####################################
# Mathematical stuff ################
####################################

def minsquare(X,Y,G=1):
    """
       Calculates the coefficients of the best polinomial to fil the points in X,Y list. G is the degree of the polinomial.
       
       y=alpha[0]+alpha[1]*x+alpha[2]*x**2+...
    """
    np=len(X)
    csi=[sum([x**i for x in X]) for i in range(2*G+1)]
    phi=matrix([sum([Y[j]*X[j]**i for j in range(np)]) for i in range(G+1)])
    lamb=matrix([[csi[i+j] for j in range(G+1)]for i in range(G+1)])
    alpha=lamb.I*phi.T
    return alpha.T.tolist()[0]




def integral(func, ini, fin, Nints=20):
    """ Calculates the integral of func between ini and fin
    with 20 points gaussian method dividing the interval [ini; fin] 
    in Nint intervals."""
    Y=[.993128599185094924786,.963971927277913791268,.912234428251325905868,
       .839116971822218823395,.74633190646015079614,.636053680726515025453,
       .510867001950827098004,.373706088715419560673,.227785851141645078080,
                     .076526521133497333755]
    W=[.017614007139152118312,.040601429800386941331,.062672048334109063570,
      .083276741576704748725,.101930119817240435037,.118194531961518417312,
      .131688638449176626898,.142096109318382051329,.149172986472603746788,
                   .152753387130725850698]
    stepint = (fin-ini)/float(Nints)
    suma = 0.
    for ii in xrange(Nints):
        A = ini + ii*stepint
        B = ini + (ii+1)*stepint
        bma = (B-A)*.5
        apb = (A+B)*.5
        tosum = [W[jj]*(func(bma*Y[jj]+apb)+func(-bma*Y[jj]+apb)) for jj in xrange(10)]
        suma += bma*sum(tosum)
    return suma

def fitter(func,xx,yy,p0, suc=False):
    """
       Fits the points in xx,yy according to func with parameters in p0.
       func = func(p, x)
    """
    errfunc=lambda p, x, y : (func(p, x) - y)
    p1, succ = leastsq(errfunc, p0[:], args=(array(xx),array(yy)))
    if suc:
        return p1.tolist(), succ
    else:
        return p1.tolist()

def zero(func, x0, eps=1.e-9, delt=.01):
    """
       Finds the zero of function func starting at x0 with precision eps.
       delt is the dx for calculation the derivative (func(x+delt)-func(x))/delt)
    """
    diff = func(x0)
    xx = x0
    while abs(diff) > eps:
        der = (func(xx+delt)-func(xx))/delt
        ddd = -diff/der
        xx += ddd
        diff = func(xx)
    return xx




