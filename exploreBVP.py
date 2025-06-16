import numpy as np
from scipy import integrate as nint
from matplotlib import pyplot as plt

def waveOnString(x, y, omega, T):
    yPrime = y[-1]
    yDPrime = -omega*omega*y[0]/T
    return yPrime, yDPrime

def rk4(function, y0, span, dt, params):
    """
    Numerically solves coupled differential equations using a 4th order Runge Kutta sovler

    Parameters 
    ----------

    Returns
    ----------
    """
    t = span[0]
    N = int((span[-1]-span[0])/dt)
    x = np.zeros(N)
    u = np.zeros(N)
    x[0] = y0[0]
    u[0] = y0[-1]
    for i in range(1, N):
        k1x, k1u = function(t, (x[i-1], u[i-1]), *params)
        k1x *= dt
        k1u *= dt
        k2x, k2u = function(t, (x[i-1] + k1x/2, u[i-1] + k1u/2), *params)
        k2x *= dt
        k2u *= dt
        k3x, k3u = function(t, (x[i-1] + k2x/2, u[i-1] + k2u/2), *params)
        k3x *= dt
        k3u *= dt
        k4x, k4u = function(t, (x[i-1] + k3x, u[i-1] + k3u), *params)
        k4x *= dt
        k4u *= dt

        x[i] = x[i-1] + (k1x + 2*k2x + 2*k3x + k4x)/6
        u[i] = u[i-1] + (k1u + 2*k2u + 2*k3u + k4u)/6

    return x, u

spRes = nint.solve_ivp(waveOnString, t_span = (0,1), y0 = (1,0), args = (1,1))
print(spRes["y"][0])


def shootingMethod(function, bcI, bcF, p0, threshold, span):
    
    initialGuess = nint.solve_ivp(function, t_span = span, y0 = bcI, args = (p0, 1))
    p = [p0]
    dif = [bcF - initialGuess["y"][1][-1]]
    guessTwo = nint.solve_ivp(function, t_span = span, y0 = bcI, args = (p0 - 0.1, 1))
    p.append(p0 - 0.1)
    dif.append(bcF - guessTwo["y"][1][-1])
    while np.abs(dif[-1]) > threshold:
        p0 = p[-1] - dif[-1] * (p[-2] - p[-1]) / (dif[-2] - dif[-1])
        p.append(p0)
        guess = nint.solve_ivp(function, t_span = span, y0 = bcI, args = (p0, 1))
        dif0 = bcF - guess["y"][1][-1]
        dif.append(dif0)
        fig, ax = plt.subplots(1,1)
        ax.plot(guess["t"], guess["y"][0])
        ax.plot(guess["t"], guess["y"][1])
        plt.show()
    return p[-1]

omega = shootingMethod(waveOnString, (1,0), 0, 1.5, 1e-5, (0,1))
print(omega)