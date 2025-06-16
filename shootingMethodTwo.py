import numpy as np
from scipy import integrate as nint
from matplotlib import pyplot as plt
import kepUnits as kep
import cgs
from tqdm import tqdm


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

def dEdr(r, E, omega, T, p0, p1, rPeak, q, r0, a, M = 1):

    omegaKep = np.sqrt(cgs.G*M/((r*a)**3))
    cs2 = cgs.kB*T/(28*cgs.mp) * (r*a/r0)**(-q)
    p = np.where(r < rPeak, p0, p1)

    Eprime = E[1]
    #E1prime = -E[1]*(3-p)/r - E[0]*(p/(r*r) + r*r*r*(omegaKep**2)/(2*a*a*cs2) - 2*omegaKep*omega/(cs2))
    E1prime = (-E[1]*(3-p)/(r) - E[0]*((6 - q*(q+2) - p*(q+1))/(r*r)  - 2*a*a*omegaKep*omega/(cs2) + 6*r*r*a*a*a*a*(omegaKep**4)/(cgs.c*cgs.c*cs2)))
    print(omegaKep, omega, cs2)

    return np.array((Eprime, E1prime))

def shootingMethod(function, bcI, bcF, span, bounds, threshold, params):
    params[0] = bounds[0]
    guessZero = nint.solve_ivp(function, span, y0 = bcI, method = "Radau", args = params, max_step = 1e9)
    print(guessZero["y"][1][-1])
    if guessZero["y"][1][-1] - guessZero["y"][0][-1]*bcF < 0:
        sign = 1
    else:
        sign = -1
    print(sign)
    p0 = np.sqrt(bounds[0]*bounds[1])
    params[0] = p0
    print(p0)
    initialGuess = nint.solve_ivp(function, span, y0 = bcI, method = "Radau", args = params, max_step = 1e9)
    dif = initialGuess["y"][1][-1] - bcF
    print(dif)
    while np.abs(dif) > threshold:
        print(p0, dif)
        if dif*sign  < 0:
            bounds[0] = p0
        else:
            bounds[1] = p0
        p0 = np.sqrt(bounds[0]*bounds[1])
        params[0] = p0
        guess = nint.solve_ivp(function, span, y0 = bcI, method = "Radau", args = params, max_step = 1e-2)
        dif = guess['y'][1][-1] - bcF
    finalGuess = nint.solve_ivp(function, span, y0 = bcI, method = "Radau", args = params, max_step = 1e-2)
    return p0, finalGuess

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

    return {"t" : t, "y" : np.vstack((x,u))}


a = cgs.RSun
r0 = 0.1*cgs.RSun
rf = 1*cgs.RSun

omegaKep = np.sqrt(cgs.G*0.6*cgs.MSun/(a**3))
omega = 0.0008289007775435815*omegaKep

res = nint.solve_ivp(dEdr, (r0/a,rf/a), y0 = (1,0), method = "Radau", args = [omega, 3000, -2, 2, 0, 0, r0, a, 0.6*cgs.MSun], max_step = 1e9)
#omega, res = shootingMethod(dEdr, (1,0), 0, (r0/a, rf/a), [1e-6*omegaKep,1e-4*omegaKep], 1e-4*a/rf, params = [30, 3000, -2, 2, 0, 0, r0, a, 0.6*cgs.MSun])
#res = rk4(dEdr, (1,0), (r0, 10*r0), dt = 1e5, params = (omega, 5000, 2, 0, r0, 0.6*cgs.MSun))

#r = res[2]
r = res["t"]
#E = res[0]
E = res["y"][0]
#Eprime = res[1]
Eprime = res["y"][1]

fig, ax = plt.subplots(2,1)
ax[0].plot(r, E)
ax[1].plot(r, Eprime)
plt.show()

print(omega/omegaKep)

omegaKep = np.sqrt(cgs.G*0.6*cgs.MSun/(r0**3))


#omega = np.linspace(0,100 ,100)
#dif = np.zeros_like(omega)
#for i in range(len(omega)):
#    res = nint.solve_ivp(dEdr, (0.0004,0.004), (1,0), method = "RK45", args = (omega[i], 3000, 3, 0, 0.0004, 1))
#    dif[i] = res["y"][1][-1]
#fig, ax = plt.subplots(1,1)
#ax.plot(omega, dif)
#plt.show()