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

        ###initialize Real space ###
        self.width = size[0]

        self.height = size[1]

        self.num_X = np.floor(self.width*resolution)
        self.X = np.linspace(-self.width/2, self.width/2, self.num_X).reshape((-1,1))

        self.num_Y = np.floor(self.height*resolution)
        self.Y = np.linspace(-self.width/2, self.width/2, self.num_Y).reshape((1,-1))

        self.initial_psi = initial_psi(self.X, self.Y)

        ###inintialize reciprocal space####
        self.GX = np.linspace(0,resolution, self.num_X).reshape((-1,1))
        self.GX[self.num_X//2 + self.num_X % 2:,0] -= resolution
        self.GX_sort_indices = np.argsort(self.GX)

        self.GY = np.linspace(0,resolution, self.num_Y).reshape((1,-1))
        self.GY[0,self.num_Y//2 + self.num_Y % 2:] -= resolution
        self.GY_sort_indices = np.argsort(self.GY[0,:])

        self.G_sqr = self.GX**2 + self.GY**2
        print(self.G_sqr.shape)

    def laplacian(self, psi):
        if (len(psi.shape) == 3) & (psi.shape[:2] == (self.num_X, self.num_Y)): #3rd dim is time

            return np.fft.ifft2(self.G_sqr[:,:,np.newaxis]*np.fft.fft2(psi, axes=(0,1)),axes=(0,1))

        elif (len(psi.shape) == 2) & (psi.shape[:2] == (self.num_X, self.num_Y)): # no time dim

            return np.fft.ifft2(self.G_sqr*np.fft.fft2(psi, axes=(0,1)),axes=(0,1))
        else:
            raise ValueError("psi shape is not 2d or 2d+time")
#

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
    return normalized_2d_gaussian(X,Y,0.5) * np.exp(1j*X*7)

def create_cmap_from_csv(directory, cmap_name, n_bin=0):
    colors = np.genfromtxt(directory + cmap_name + '.csv', delimiter=',')
    colorsRGBa = colors/255
    if n_bin < colors.shape[0]:
        n_bin = colors.shape[0]
    new_cmap = LinearSegmentedColormap.from_list(cmap_name, colorsRGBa, N=n_bin)
    return new_cmap


if __name__=='__main__':

    test_particle = rectangular_1_particle_domain((10,10),20,initial_psi)

    t, psi = test_particle.time_evolution(np.linspace(0,50,100), 0)

    # for i in range(psi.shape[-1]):
    #     psi[:,:,i] = test_particle.laplacian(psi[:,:,i])

    peak_amplitude = np.abs(psi).max().max().max()
    #print(peak_amplitude)


    cyclic_cmap = create_cmap_from_csv("../CET_colormaps/", "CET-C7")

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
    ani.save(filename="./particle.mp4", writer="ffmpeg")
    #plt.show()
