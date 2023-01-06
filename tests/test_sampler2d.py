import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm
import matplotlib as mpl

import os
import scipy.interpolate

from adaptsamp.sampler2d import AdaptiveSampler2D, Triangle, Interval2D

#width_range = [120, 230]
#wvl_range = [600, 780]

x_min = 600.
x_max = 780.
x_fine = np.linspace(x_min, x_max, 101)

y_min = 120.
y_max = 230.
y_fine = np.linspace(y_min, y_max, 101)

X,Y = np.meshgrid(x_fine, y_fine, indexing='ij')


def lorentzian(x, x0, gamma, a):
    return a*(0.5*gamma)/( (x-x0)**2 + (0.5*gamma)**2)

def lorentzian_peak_val(gamma, a):
    return a*0.5*gamma/(0.5*gamma)**2

def lorentzian_a_factor(gamma, peak_val):
    return (2*peak_val/gamma)*(0.5*gamma)**2

def test_function(wavelength, width, height):
    peak_pos = peak_pos_function(width, height)
    peak_int = peak_intensity_function(width, height)
    width = peak_width_function(width, height)
    return lorentzian(wavelength, peak_pos, width, peak_int)
    
def peak_pos_function(width, height):
    m1 = 1.5 + 1e-2*(height-250)
    m2 = -2e-3 - 1e-5*(height-250)
    #1e-2*(height-250)
    return 1.5*(250-height)+480 + m1*width + m2*width**2

def peak_width_function(width, height):
    m1 = -3e-2 
    m2 = 0.
    return 10 + m1*(width-140) - m2*(width-120)**2


def peak_intensity_function(width, height):
    m1 = 1e-8 -  1e-9*(height-250)
    m2 = 1e-5 
    return 3.5 + m1* (width-180) + m2*(width-160+(height-250))**2
    
def test_func(x, y):    
    #return np.cos(3*x/np.pi)*np.sin(3*y/np.pi)
    #return np.exp(-(x-5.)**2/2**2)*np.exp(-(y-7.)**2/2**2)
    return test_function(x, y, 250.)
    


Z = test_func(X, Y)

adapt = AdaptiveSampler2D(test_func, x_min, x_max, y_min, y_max,
                             n_parallel=13, max_func=1000)

adapt.init_intervals()
adapt.refine_intervals()
#x = np.array([0.1])
#y = np.array([2.1])
x_interp = X.flatten()
y_interp = Y.flatten()

f_interp = adapt.interpolate(x_interp, y_interp).reshape(X.shape)

fig, ax_array = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw={'hspace':0.3})
axes = ax_array.flatten()

plt.sca(axes[0])
plt.pcolormesh(X, Y, Z, shading='auto', vmin=0., vmax=1.0)
plt.colorbar()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("Original Function\nEvaluated at {} Points".format(x_interp.size))
plt.xlabel("x (arb.)")
plt.ylabel("y (arb.)")


plt.sca(axes[1])

grid, f_vals, triangles = adapt.get_grid(grid_type='six_point')
plt.tripcolor(grid[:,0], grid[:,1], f_vals, triangles=triangles,
              shading = 'gouraud', edgecolor='w',
              vmin=0., vmax=1.0)

plt.colorbar()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.title("Linear Approximation\nBased on {} Evaluations".format(grid.shape[0]))
plt.xlabel("x (arb.)")
plt.ylabel("y (arb.)")

plt.sca(axes[2])
plt.pcolormesh(X, Y, f_interp, shading='auto', vmin=0., vmax=1.0)
plt.colorbar()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.title("Quadratic Approximation\nBased on {} Evaluations".format(adapt.n_evaluations))
plt.xlabel("x (arb.)")
plt.ylabel("y (arb.)")


plt.sca(axes[3])

grid, error, triangles = adapt.get_error_distribution()

plt.tripcolor(grid[:,0], grid[:,1], triangles=triangles,
              shading = 'flat', facecolors=error, edgecolor='w')

plt.colorbar()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.title("Spatial Distribution of Error\nOver {} Intervals".format(len(adapt.intervals)))
plt.xlabel("x (arb.)")
plt.ylabel("y (arb.)")


print("Total evaluations of Interpolator: {}".format(x_interp.size))
print("# points for Interpolator: {}".format(adapt.n_evaluations))
plt.savefig(os.path.join("..","figures","adaptive_sampling_example.png"), dpi=300, bbox_inches='tight')

