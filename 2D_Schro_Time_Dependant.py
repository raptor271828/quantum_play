import numpy as np
import inspect
import scipy as scp
from scipy.integrate import solve_ivp
h_bar = 1
m = 1

#x as dim 1, y as dim 2, periodic (so plane wave basis can be used)
class rectangular_1_particle_domain():
    def __init__(self, size, resolution, initial_psi):

        self.width = size[0]

        self.height = size[1]

        self.num_X = np.floor(self.width*resolution)
        self.X = np.linspace(-self.width/2, self.width/2, self.num_X)

        self.num_Y = np.floor(self.height*resolution)
        self.Y = np.linspace(-self.width/2, self.width/2, resolution, self.num_Y)[np.newaxis,:]

        self.current_psi = initial_psi

    def laplacian(self, psi):
        gx = np.linspace(0,self.width**-1, self.num_X)
        gx[self.num_X//2 + self.num_X % 2:] -= self.width**-1

        gy = np.linspace(0,self.height**-1, self.num_Y)
        gy[self.num_Y//2 + self.num_Y % 2:] -= self.height**-1

        g_squared = gx**2 + gy[np.newaxis,:]**2

        return np.fft.ifft2(g_squared*np.fft.fft2(psi))

    def d_psi_dt(self, psi, V): # -i * hbar**2/2m * laplacian (psi) * V * psi, assiming real_space_psi
        return -1j * h_bar * self.laplacian(psi) / (2*m) + V * psi

    def time_evolution(self, time_array, V, static_V = True):
        if static_V:
            def derivative_for_ode_solver(t,y):
                return self.d_psi_dt(y.reshape(self.num_X,self.num_Y), V).reshape(-1, order='C')
        else:
            def derivative_for_ode_solver(t,y):
                t_index = np.argmin(np.abs(time_array-t))
                return self.d_psi_dt(y.reshape(self.num_X,self.num_Y), V[t_index,...]).reshape(-1, order='C')
        