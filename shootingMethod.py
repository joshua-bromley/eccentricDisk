import numpy as np
from scipy import integrate as nint
from scipy import differentiate as nderiv
from tqdm import tqdm
import cgs
import kepUnits as kep
from matplotlib import pyplot as plt

#def density(r, rPeak, rhoPeak, p0, p1):
def density(x, law, *args):
    """
    One-dimensional double power law density profile

    Rho = {rhoPeak*(r/rPeak)^p0; r < rPeak, rhoPeak*(r/rPeak)^p1, r > rPeak}

    Ideally p0 > 0, p1 < 0

    Parameters
    ----------
    r : double
        radius from central binary
    rPeak : double
        radius of peak density
    rhoPeak : double
        peak density (g/cm^3)
    p0 : double
        index of inner power law
    p1 : double
        index of outer power law

    

    Reutrns
    -------
    double 
        density at r in g/cm^3
    """


    #p = np.where(r < rPeak, p0, p1)
    #return rhoPeak*(r/rPeak)**p
    if law == "VSS":
        xPeak = args[0]
        rho0 = args[1]
        return (rho0*np.sqrt(1/x))*(1 - 0.7*np.sqrt(1/x))*np.exp(-(xPeak/x)**12)
    if law == "SinglePower":
        x0 = args[0]
        rho0 = args[1]
        p = args[2]
        return rho0*(x/x0)**p
    if law == "tanh":
        x0 = args[0]
        xIn = args[1]
        xOut = args[2]
        rho0 = args[3]
        wOut = 0.01*xOut
        return rho0* np.sqrt(x0/x)*(1 - np.sqrt(xIn/x))**(5/9)*np.tanh((xOut - x)/(wOut))

def densityGradient(x, law, *args):
    if law == "VSS":
        xPeak = args[0]
        rho0 = args[1]
        return np.exp(-(xPeak/x)**12)*rho0*np.sqrt(1/x)*((1-0.7*np.sqrt(1/x))*(-0.5*np.sqrt(1/x) + 12/xPeak * (xPeak/x)**13) + 0.5*0.7*np.sqrt(1/(x*x*x)))
    if law == "SinglePower":
        x0  = args[0]
        rho0 = args[1]
        p = args[2]
        return (p*rho0*(x/x0)**(p-1))/x0
    if law == "tanh":
        x0 = args[0]
        xIn = args[1]
        xOut = args[2]
        rho0 = args[3]
        wOut = 0.01*xOut
        return rho0* (np.sqrt(x0/x)*(18*x*(np.sqrt(x/xIn) -1)*(np.cosh((xOut - x)/wOut)**(-2)) - 5*wOut*np.tanh(xOut - x)/wOut))/(18*wOut*(-np.sqrt(x0/x)*(np.sqrt(x/xIn)-1)**(4/9)))
     
    


def soundSpeed2(x, law,*args):
    if law == "constH0":
        a = args[0]
        M = args[1]
        h0 = args[2]
        omegaKep = np.sqrt(cgs.G*M*cgs.MSun/(a**3))
        return h0*h0*omegaKep*omegaKep*a*a/x
    if law == "const":
        T = args[0]
        mu = args[1]
        return cgs.kB*T/mu/cgs.mp
    if law == "tanh":
        a = args[0]
        M = args[1]
        h0 = args[2]
        xIn = args[3]
        omegaKep = np.sqrt(cgs.G*M*cgs.MSun/((x*a)**3))
        return h0*a*x*(1-np.sqrt(xIn/x))**(2/9)*omegaKep


def soundSpeedGradient(x, law,*args):
    if law == "constH0":
        a = args[0]
        M = args[1]
        h0 = args[2]
        omegaKep = np.sqrt(cgs.G*M*cgs.MSun/(a**3))
        return -h0*h0*omegaKep*omegaKep*a/x/x
    if law == "const":
        return 0
    if law == "tanh":
        a = args[0]
        M = args[1]
        h0 = args[2]
        xIn = args[3]
        omegaKep = np.sqrt(cgs.G*M*cgs.MSun/((x*a)**3))
        return h0*a*np.sqrt(xIn/x)*(25*xIn - 23*x*np.sqrt(xIn/x))*omegaKep/(18*xIn*(-a*h0*x*(np.sqrt(xIn/x)-1))**(7/9))


def soundSpeedGradient2(x, law,*args):
    if law == "constH0":
        a = args[0]
        M = args[1]
        h0 = args[2]
        omegaKep = np.sqrt(cgs.G*M*cgs.MSun/(a**3))
        return 2*h0*h0*omegaKep*omegaKep/x/x/x
    if law == "const":
        return 0
    if law == "tanh":
        a = args[0]
        M = args[1]
        h0 = args[2]
        xIn = args[3]
        omegaKep = np.sqrt(cgs.G*M*cgs.MSun/((x*a)**3))
        return h0*h0*a*a*np.sqrt(xIn/x)*(xIn*(1075*np.sqrt(xIn/x) - 2032) + 943*x*np.sqrt(xIn/x))*omegaKep/(324*xIn*(-a*h0*x*(np.sqrt(xIn/x)-1))**(16/9))

def laplaceCoeff(a,j,s):
    """
    Compute laplace coefficient b(j,s/2,a) = 1/pi int_0^2pi cos(j phi)d phi / (1 + a^2 - 2acos phi)^(s/2)
    Parameters
    ---------
    a : double
    j : double
    s : double

    Returns
    ---------
    double
        Laplace Coefficient b(j,s/2,a)
    """
    b = (1/np.pi)*nint.quad(lambda phi, a, j, s : np.cos(j*phi)/((1 + a*a - 2*a*np.cos(phi))**(s/2)), 0, 2*np.pi, args=(a,j,s))[0]
    return b

def eccentricityEvol(x, E, law, omega, M, a, q, h0, rhoParams, csParams):
    """
    Evolution of eccentricity in a circumbinary disk, see eq 26 in Goodchild and Ogilvie (2006), uses ansatz E(r,t) = E(r)e^i omega t

    Parameters
    ---------

    x : double
        radius from central binary in a_b
    E : ndarray
        length 2 ndarray where the first entry is the scaled eccentricity and the second entry is dE[0]/dr
    omega : double
        frequency (in time) of the procession of Eccentricity in omega_b
    M : double
        central binary mass (MSun)
    a : double
        central binary semi-major axis (cm)
    q : double
        central binary mass ratio
    h0 : double
        disk aspect ratio
    rhoParams : list
        parameters for density function

    Returns
    -------

        ndarray(double)
            dE/dx [dE/dx, dE^2/dx^2]
    """ 
    if law == "2DIso":
        densityLaw = "VSS"
        csLaw = "constH0"
    if law == "3DIso":
        densityLaw = "SinglePower"
        csLaw = "const"


    rho = density(x, densityLaw, *rhoParams)
    drhodr = densityGradient(x, densityLaw, *rhoParams)

    cs2 = soundSpeed2(x, csLaw, *csParams)
    dcs2dr = soundSpeedGradient(x, csLaw, *csParams)
    d2cs2dr2 = soundSpeedGradient2(x, csLaw, *csParams)

    omegaKep0 = np.sqrt(cgs.G*M*cgs.MSun/((a)**3))
    omegaKep = np.sqrt(cgs.G*M*cgs.MSun/((x*a)**3))
      

    E0prime = E[1]
    if law == "2DIso":
        E1prime = -E[1]*(3/x + drhodr/rho) - E[0]*(drhodr/(x*rho) + dcs2dr*(-2/x - drhodr/rho)/cs2 - d2cs2dr2/cs2 - 2*a*a*omegaKep*omegaKep0*omega/cs2  + 3*q*omegaKep*omegaKep*a*a/(4*cs2*x*x*(1+q)*(1+q)))
        #print( drhodr/(x*rho) + dcs2dr*(-2/x - drhodr/rho)/cs2 - d2cs2dr2/cs2, 2*a*a*omegaKep*omegaKep0*omega/cs2, 3*q*omegaKep*omegaKep*a*a/(4*cs2*x*x*(1+q)*(1+q)))
        #print(-E[1]*(3/x + drhodr/rho), E[0]*(drhodr/(x*rho) + dcs2dr*(-2/x - drhodr/rho)/cs2 - d2cs2dr2/cs2 - 2*a*a*omegaKep*omegaKep0*omega/cs2  + 3*q*omegaKep*omegaKep*a*a/(4*cs2*x*x*(1+q)*(1+q))) )
        #print(E1prime)
    if law == "3DIso":
        E1prime = -E[1]*(3/x + drhodr/rho) - E[0]*(drhodr/(x*rho) + dcs2dr*(1/x - drhodr/rho)/cs2 - d2cs2dr2/cs2 + 6/x/x - 2*a*a*omegaKep*omegaKep0*omega/cs2 + 6*x*x*a*a*a*a*(omegaKep**4)/(cgs.c*cgs.c*cs2) + 0*3*q*omegaKep*omegaKep*a*a/(4*cs2*x*x*(1+q)*(1+q)))
        #print(drhodr/(x*rho) + dcs2dr*(1/x - drhodr/rho)/cs2 - d2cs2dr2/cs2 + 6/x/x, 2*a*a*omegaKep*omegaKep0*omega/cs2, 6*x*x*a*a*a*a*(omegaKep**4)/(cgs.c*cgs.c*cs2))


    return np.array((E0prime,E1prime))

def rk4(function, y0, span, dt, params):
    """
    Numerically solves coupled differential equations using a 4th order Runge Kutta sovler

    Parameters 
    ----------

    Returns
    ----------
    """

    N = int((span[-1]-span[0])/dt)
    x = np.zeros(N)
    u = np.zeros(N)
    t = np.zeros(N)
    t[0] = span[0]
    x[0] = y0[0]
    u[0] = y0[-1]
    for i in tqdm(range(1, N)):
        k1x, k1u = function(t[i-1], (x[i-1], u[i-1]), *params)
        k1x *= dt
        k1u *= dt
        k2x, k2u = function(t[i-1], (x[i-1] + k1x/2, u[i-1] + k1u/2), *params)
        k2x *= dt
        k2u *= dt
        k3x, k3u = function(t[i-1], (x[i-1] + k2x/2, u[i-1] + k2u/2), *params)
        k3x *= dt
        k3u *= dt
        k4x, k4u = function(t[i-1], (x[i-1] + k3x, u[i-1] + k3u), *params)
        k4x *= dt
        k4u *= dt

        x[i] = x[i-1] + (k1x + 2*k2x + 2*k3x + k4x)/6
        u[i] = u[i-1] + (k1u + 2*k2u + 2*k3u + k4u)/6
        t[i] = t[i-1]+dt

    return x, u, t

def shootingMethod(function, bcI, bcF, span, bounds, threshold, params):
    sqrtSgn = 1
    if bounds[0] < 0:
        sqrtSgn = -1
    params[1] = bounds[1]
    guessZero = nint.solve_ivp(function, span, y0 = bcI, method = "Radau", args = params)
    print(guessZero["y"][1][-1], guessZero["y"][0][-1], bcF)
    if guessZero["y"][1][-1] - guessZero["y"][0][-1]*bcF < 0:
        sign = -1
    else:
        sign = 1
    print(sign)
    p0 = sqrtSgn*np.sqrt(bounds[0]*bounds[1])
    params[1] = p0
    initialGuess = nint.solve_ivp(function, span, y0 = bcI, method = "Radau", args = params)
    dif = initialGuess['y'][1][-1] - initialGuess["y"][0][-1]*bcF
    while np.abs(dif) > threshold:
        print(p0, dif)
        if dif*sign < 0:
            bounds[0] = p0
        else:
            bounds[1] = p0
        p0 = sqrtSgn*np.sqrt(bounds[0]*bounds[1])
        params[1] = p0
        guess = nint.solve_ivp(function, span, y0 = bcI, method = "Radau", args = params)
        dif = guess['y'][1][-1] - guess["y"][0][-1]*bcF
    params[1] = p0
    finalGuess = nint.solve_ivp(function, span, y0 = bcI, method = "Radau", args = params)
    return p0, finalGuess



a = cgs.RSun #0.1*cgs.AU
xCav = 2.5
#x0 = 0.75*xCav*a
#xf = 10*a
x0 = 0.4*cgs.RSun
xf = 1*cgs.RSun

M = 0.6
q = 0.01
h0 = 0.1

#rhoParams = (xCav, 1000)
#csParams = (a,M,h0)

rhoParams = (x0,1000,-2)
csParams = (5000,28)

#rhoParams = (1, x0/a, xf/a, 1000)
#csParams = (a, M, h0, x0/a)

omegaKep = np.sqrt(cgs.G*M*cgs.MSun/(a**3))
omegaQ = (3/4) * (1/(1+q)/(1+q)) * (1/xCav/xCav) * (xCav**(-3/2))
omega = 3e-4 #*(0.1*cgs.AU/cgs.RSun)**(3/2)




#res = nint.solve_ivp(eccentricityEvol, (x0/a, xf/a), (1,0), method="BDF", args = ["3DIso" ,omega, M, a, q, h0, rhoParams, csParams])
omega, res = shootingMethod(eccentricityEvol, (1,0), 0 ,(x0/a, xf/a), [1e-5,1e-3], 1e-3*a/xf, params = ["3DIso",omega, M, a, q, h0, rhoParams, csParams])
res2 = nint.solve_ivp(eccentricityEvol, (x0/a, xf/a), (1,0), method="Radau", args = ["3DIso" ,omega, M, a, q, h0, rhoParams, csParams])

fig, ax = plt.subplots(2,1)
#ax[0].plot(res["t"], res["y"][0])
#ax[1].plot(res["t"], res["y"][1])
ax[0].plot(res2["t"], res2["y"][0], ls = "--")
ax[1].plot(res2["t"], res2["y"][1], ls = "--")
#ax[0].set_ylim(-0.1,1)
#ax[0].set_xscale("log")
#ax[0].set_yscale("log")
#ax[1].set_xscale("log")
#ax[1].set_yscale("log")


ax[1].set_xlabel("r")


plt.show()

print(omega, omega/omegaQ, h0*h0/omegaQ)
print(omega)
#print(omega*omegaKep*3.155e7)
"""
omega = np.logspace(-4,-3)
err = np.zeros_like(omega)

for i in range(len(omega)):
    res = nint.solve_ivp(eccentricityEvol, (x0/a, xf/a), (1,-a/x0), method="Radau", args = ["2DIso" ,omega[i], M, a, q, h0, rhoParams, csParams])
    err[i] = res["y"][1][-1] - res["y"][0][-1]/res["t"][-1]

fig, ax = plt.subplots(1,1)
ax.plot(omega, np.abs(err))
ax.plot(omega,err)
ax.set_xlabel("omega")
ax.set_ylabel("dE/dr - E/r")
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()
"""
