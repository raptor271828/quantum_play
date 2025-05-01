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
        self.GX_sort_indices = np.argsort(self.GX[:,0]).reshape((-1,1))

        self.GY = np.linspace(0,resolution, self.num_Y).reshape((1,-1))
        self.GY[0,self.num_Y//2 + self.num_Y % 2:] -= resolution
        self.GY_sort_indices = np.argsort(self.GY[0,:]).reshape((1,-1))

        self.G_sqr = self.GX**2 + self.GY**2
        print(self.G_sqr.shape)

    ###position operators

    def distance_from_indices(self, r1, r2):
        X1 = self.X[r1[0, ...],0]
        Y1 = self.X[r1[1, ...],0]
        X2 = self.X[r2[0, ...],0]
        Y2 = self.Y[r2[1, ...],0]

        dX = (X2-X1) % self.width/2
        dY = (Y2-Y1) % self.width/2

        return (dX**2 + dY**2)**0.5



    ###momentum (and momentum related) operators!

    def laplacian(self, psi):
        if (len(psi.shape) == 3) & (psi.shape[:2] == (self.num_X, self.num_Y)): #3rd dim is time

            return -np.fft.ifft2(self.G_sqr[:,:,np.newaxis]*np.fft.fft2(psi, axes=(0,1)),axes=(0,1))

        elif (len(psi.shape) == 2) & (psi.shape[:2] == (self.num_X, self.num_Y)): # no time dim

            return -np.fft.ifft2(self.G_sqr*np.fft.fft2(psi, axes=(0,1)),axes=(0,1))
        else:
            raise ValueError("psi shape is not 2d or 2d+time")

    def momentum_space(self, psi):

        if (len(psi.shape) == 3) & (psi.shape[:2] == (self.num_X, self.num_Y)): #3rd dim is time
            fft_out = np.fft.fft2(psi[self.GX_sort_indices,self.GY_sort_indices,:], axes=(0,1), norm='forward') #shifting psi needed to define origen at center of unit cell
            return fft_out[self.GX_sort_indices,self.GY_sort_indices,:]

        elif (len(psi.shape) == 2) & (psi.shape[:2] == (self.num_X, self.num_Y)): # no time dim
            fft_out = np.fft.fft2(psi[self.GX_sort_indices,self.GY_sort_indices], axes=(0,1), norm='forward')
            return fft_out[self.GX_sort_indices,self.GY_sort_indices]#[self.GX_sort_indices,self.GY_sort_indices]

        else:
            raise ValueError("psi shape is not 2d or 2d+time")

    # def momentum(self, psi):
    #     if (len(psi.shape) == 3) & (psi.shape[:2] == (self.num_X, self.num_Y)): #3rd dim is time

    #         return np.fft.ifft2(self.G_sqr[:,:,np.newaxis]*np.fft.fft2(psi, axes=(0,1)),axes=(0,1))



#
    ###hey! its the schrodinger equation!
    def d_psi_dt(self, psi, V): # -i * hbar**2/2m * laplacian (psi) * V * psi, assiming real_space_psi
        #print(np.abs(-1j * h_bar**2 * self.laplacian(psi) / (2*m) + V * psi).max())
        return -1j * (-h_bar**2 * self.laplacian(psi) / (2*m) + V * psi) / h_bar

    ###wrapper for np.integrate.solveIVP
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

    ###XY mesh utility (rather than have it be cached)
    def mesh(self, real=True):
        if real:
            return np.stack(np.meshgrid(self.X[:,0], self.Y[0,:]))
        else:
            return np.stack(np.meshgrid(self.GX[:,0], self.GY[0,:]))



###random utils

def normalized_2d_gaussian(X, Y, sigma, dX=0, dY=0):
    return (2. * np.pi * sigma)**-1 * np.exp(-((X-dX)**2+(Y-dY)**2)/(2*sigma), dtype=complex)

def create_cmap_from_csv(directory, cmap_name, n_bin=0):
    colors = np.genfromtxt(directory + cmap_name + '.csv', delimiter=',')
    colorsRGBa = colors/255
    if n_bin < colors.shape[0]:
        n_bin = colors.shape[0]
    new_cmap = LinearSegmentedColormap.from_list(cmap_name, colorsRGBa, N=n_bin)
    return new_cmap






if __name__=='__main__':
    file_name= input("name your video: ")

    #potential


    def initial_psi(X,Y):
        return normalized_2d_gaussian(X,Y,0.7, dX=5) #* np.exp(1j*X*7)
    # #create simulation domain
    test_particle = rectangular_1_particle_domain((20,20),5,initial_psi)

    # #potential

    V = normalized_2d_gaussian(test_particle.X,test_particle.Y, 5)*-20
    #V[np.isnan(V)] =np.nanmin(V)*1.5


    # t, psi = test_particle.time_evolution(np.linspace(0,1000,1000), V)

    # np.save(file_name+"_psi.npy", psi)
    # np.save(file_name+"_t.npy", t)

    psi = np.load(file_name+"_psi.npy")
    t = np.load(file_name+"_t.npy")
    momentum = test_particle.momentum_space(psi)
    # for i in range(psi.shape[-1]):
    #     psi[:,:,i] = test_particle.laplacian(psi[:,:,i])

    peak_psi_amplitude = np.abs(psi).max().max().max()
    peak_mom_amplitude =  np.abs(momentum).max().max().max()
    #print(peak_amplitude)


    ###prep for animating
    cyclic_cmap = create_cmap_from_csv("../CET_colormaps/", "CET-C6")

    fig, (ax_pos, ax_mom) = plt.subplots(1,2)
    fig.patch.set_color("Black")



    ####create images for animation###
    ax_pos.imshow(np.zeros(psi.shape[:2]+(3,)))
    im_pos = ax_pos.imshow(np.angle(psi[:,:,0]), vmin=-np.pi, vmax=np.pi, cmap=cyclic_cmap, alpha=np.abs(psi[:,:,0])/peak_psi_amplitude, interpolation='none')
    im_pos.set_animated(True)


    ax_mom.imshow(np.zeros(psi.shape[:2]+(3,)))
    im_mom = ax_mom.imshow(np.angle(momentum[:,:,0]), vmin=-np.pi, vmax=np.pi,cmap=cyclic_cmap, alpha=np.abs(momentum[:,:,0])/peak_mom_amplitude, interpolation='none')
    im_mom.set_animated(True)

    axis_to_data = ax_mom.transAxes + ax_mom.transData.inverted()
    half_x, half_y = axis_to_data.transform((0.5,0.5))

    ax_mom.axhline((half_x), linestyle=":", color='#131313', zorder=0)
    ax_mom.axvline((half_y), linestyle=":", color='#131313', zorder=0)


    ##### animate!!!!
    def update(frame):
        title = ax_pos.set_title(f"{frame}")
        im_pos.set_array(np.angle(psi[:,:,frame]))
        im_pos.set_alpha(np.abs(psi[:,:,frame])/np.abs(psi[:,:,frame]).max().max())

        im_mom.set_array(np.angle(momentum[:,:,frame]))
        im_mom.set_alpha(np.abs(momentum[:,:,frame])/np.abs(momentum[:,:,frame]).max().max())
        fig.savefig("./output/"+file_name+f"{frame:05d}"+".png")
        return [title,im_pos, im_mom]

    ani = animation.FuncAnimation(fig=fig, func=update, frames=t.shape[0], interval=10)

    ##output
    ani.save(filename="./"+file_name+"high_res"+".mp4", writer="ffmpeg", fps=30, extra_args=['-vcodec', 'libx264', '-crf', '9'])
    plt.show()
