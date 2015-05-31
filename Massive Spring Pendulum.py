import numpy as np
from scipy.integrate import odeint
from numpy import sin, cos, pi, array

spring = {
    'm'     :   1.0,      # Mass of spring in kg
    'k'     :   1.0e3,  # Spring constant Nm^-1
    'l'     :   3.0e-2  # Rest length in m
    }

mass = 1.0  # Mass of attachment in kg

init = array([pi/2, 0, mass*9.8/spring['k'], 0]) # initial values
      #array([theta, theta_dot, x, x_dot])

def deriv(z, t, spring, mass): # return derivatives of the array y
    m = spring['m']
    k = spring['k']
    l = spring['l']
    M = mass
    g = 9.8
    
    return array([
        z[1],
        -1.0/(l+z[2])*(2*z[1]*z[3]+g*(m/2+M)/(m/3+M)*sin(z[0])),
        z[3],
        (l+z[2])*z[1]**2+(m/2+M)/(m/3+M)*g*cos(z[0])-1.0/(m/3+M)*k*z[2]
        ])


time = np.linspace(0.0,10.0,1000)
y = odeint(deriv,init,time, args = (spring, mass))

##############################################################################
#                                                                            #
#  Animate                                                                   #
#                                                                            #
##############################################################################
import matplotlib.pyplot as plt
import matplotlib.animation as animation

l = spring['l']
r = l+y[:,2]
theta = y[:,0]
dt = np.mean(np.diff(time))

x = r*sin(theta)
y = -r*cos(theta)

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, 
                     xlim=(-1.2*r.max(), 1.2*r.max()),
                     ylim=(-1.2*r.max(), 0.2*r.max()), aspect = 1.0)
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x[i]]
    thisy = [0, y[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
    interval=25, blit=True, init_func=init)

#ani.save('double_pendulum.mp4', fps=15)
plt.show()