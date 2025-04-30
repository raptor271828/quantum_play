import numpy as np
import inspect
import scipy as scp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate._ivp.radau import P
h_bar = 1
m = 1

#x as dim 1, y as dim 2, periodic (so plane wave basis can be used)
class rectangular_1_particle_domain():
    def __init__(self, size, resolution, initial_psi_func):

        self.width = size[0]

        self.height = size[1]

        self.num_X = np.floor(self.width*resolution)
        self.X = np.linspace(-self.width/2, self.width/2, self.num_X)[:,np.newaxis]

        self.num_Y = np.floor(self.height*resolution)
        self.Y = np.linspace(-self.width/2, self.width/2, self.num_Y)[np.newaxis,:]

        self.initial_psi = initial_psi(self.X, self.Y)

        self.gx = np.linspace(0,resolution, self.num_X)
        self.gx[self.num_X//2 + self.num_X % 2:] -= resolution

        self.gy = np.linspace(0,resolution, self.num_Y)
        self.gy[self.num_Y//2 + self.num_Y % 2:] -= resolution

        self.g_squared = self.gx[:,np.newaxis]**2 + self.gy[np.newaxis,:]**2

    def laplacian(self, psi):

        return np.fft.ifft2(self.g_squared*np.fft.fft2(psi))

    def d_psi_dt(self, psi, V): # -i * hbar**2/2m * laplacian (psi) * V * psi, assiming real_space_psi
        #print(np.abs(-1j * h_bar**2 * self.laplacian(psi) / (2*m) + V * psi).max())
        return -1j * (-h_bar**2 * self.laplacian(psi) / (2*m) + V * psi) / h_bar

    def time_evolution(self, time_array, V, static_V = True):
        if static_V:
            def func_for_ode_solver(t,y):
                return self.d_psi_dt(y.reshape(self.num_X,self.num_Y, order='C'), V).reshape(-1, order='C')
        else:
            def func_for_ode_solver(t,y):
                t_index = np.argmin(np.abs(time_array-t))
                return self.d_psi_dt(y.reshape((self.num_X,self.num_Y), order='C'), V[:,:,t_index]).reshape(-1, order='C')

        ivp_out = solve_ivp(func_for_ode_solver,(time_array[0],time_array[-1]),self.initial_psi.reshape(-1, order='C'), t_eval=time_array)

        return ivp_out['t'], ivp_out['y'].reshape((self.num_X,self.num_Y,-1))

def normalized_2d_gaussian(X, Y, sigma):
    return (2. * np.pi * sigma)**-1 * np.exp(-(X**2+Y**2)/(2*sigma))

def initial_psi(X,Y):
    return normalized_2d_gaussian(X,Y,0.05) * np.exp(1j*X*7)

def create_cmap_from_csv(directory, cmap_name, n_bin=0):
    colors = np.genfromtxt(directory + cmap_name + '.csv', delimiter=',')
    colorsRGBa = colors/255
    if n_bin < colors.shape[0]:
        n_bin = colors.shape[0]
    new_cmap = LinearSegmentedColormap.from_list(cmap_name, colorsRGBa, N=n_bin)
    return new_cmap


if __name__=='__main__':

    test_particle = rectangular_1_particle_domain((10,10),20,initial_psi)

    t, psi = test_particle.time_evolution(np.linspace(0,10,100), 0)

    # for i in range(psi.shape[-1]):
    #     psi[:,:,i] = test_particle.laplacian(psi[:,:,i])

    peak_amplitude = np.abs(psi).max().max().max()
    #print(peak_amplitude)


    cyclic_cmap = create_cmap_from_csv("../CET_colormaps/", "CET-C2")

    fig, ax = plt.subplots()
    ax.imshow(np.zeros(psi.shape[:2]+(3,)))
    # im = ax.imshow(np.angle(psi[:,:,0]), vmin=-np.pi, vmax=np.pi, cmap=cyclic_cmap, alpha=np.abs(psi[:,:,0])/peak_amplitude)
    # for i in range(t.shape[0]):
    #     title = ax.set_title(f"{i}")
    #     ax.imshow(np.zeros(psi.shape[:2]+(3,)))
    #     ax.imshow(np.angle(psi[:,:,i]), vmin=-np.pi, vmax=np.pi,
    #     cmap=cyclic_cmap, alpha=np.abs(psi[:,:,i])/peak_amplitude)
    #     plt.pause(0.001)

    im = ax.imshow(np.angle(psi[:,:,0]), vmin=-np.pi, vmax=np.pi, cmap=cyclic_cmap, alpha=np.abs(psi[:,:,0])/peak_amplitude)
    im.set_animated(True)

    def update(frame):
        title = ax.set_title(f"{frame}")
        im.set_array(np.angle(psi[:,:,0]))
        im.set_alpha(np.abs(psi[:,:,frame])/np.abs(psi[:,:,frame]).max().max())
        return [title,im]

    ani = animation.FuncAnimation(fig=fig, func=update, frames=t.shape[0], interval=10)
    plt.show()
