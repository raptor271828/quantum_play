import numpy as np
import scipy as scp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

h_bar = 1
m = 1

#x as dim 1, y as dim 2, periodic (so plane wave basis can be used)
class rectangular_1_particle_domain():
    def __init__(self, size, resolution):

        ###initialize Real space ###
        self.width = size[0]
        self.height = size[1]
        self.resolution = resolution

        self.num_X = np.array(np.floor(self.width*resolution),dtype=int)
        self.X = np.linspace(-self.width/2, self.width/2, self.num_X).reshape((-1,1))

        self.num_Y = np.array(np.floor(self.height*resolution),dtype=int)
        self.Y = np.linspace(-self.width/2, self.width/2, self.num_Y).reshape((1,-1))



        #self.initial_psi = initial_psi(self.X, self.Y)

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
    def time_evolution(self, initial_psi, time_array, V, static_V = True):
        if static_V:
            def func_for_ode_solver(t,y):
                print(f"fraction_complete: {t/time_array.max()}", end='\r')
                return self.d_psi_dt(y.reshape(self.num_X,self.num_Y, order='C'), V).reshape(-1, order='C')
        else:
            def func_for_ode_solver(t,y):
                print(f"fraction_complete: {t/time_array.max()}", end='\r')
                t_index = np.argmin(np.abs(time_array-t))
                return self.d_psi_dt(y.reshape((self.num_X,self.num_Y), order='C'), V[:,:,t_index]).reshape(-1, order='C')

        ivp_out = solve_ivp(func_for_ode_solver,(time_array[0],time_array[-1]),initial_psi.reshape(-1, order='C'), t_eval=time_array, atol=1e-7, rtol=1e-4)

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

def plot_and_save_psi_vs_t(psi_list, t, file_path, cmap='CET-C6', max_normalization=False):

    cyclic_cmap = create_cmap_from_csv("../CET_colormaps/", cmap)

    num_figs = len(psi_list) #psi is a tuple of 3-arrays

    fig, ax_list = plt.subplots(1,num_figs, figsize=(num_figs,1), frameon=False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    psi_max_values = []
    im_list = []
    for ax, psi in zip(ax_list, psi_list):
            #calculate global max (with repect to time) of max_normalization is on
        if max_normalization:
            psi_max_values.append(np.abs(psi).max().max().max())

        ax.set_axis_off()
        ax.imshow(np.zeros(psi.shape[:2]+(3,)))
        im_list.append(ax.imshow(np.zeros(psi.shape[:2]), vmin=-np.pi, vmax=np.pi, cmap=cyclic_cmap, alpha=np.zeros(psi.shape[:2]), interpolation='none'))
        im_list[-1].set_animated(True)

    for frame_num in range(t.shape[0]):

        for n, (ax, psi, im) in enumerate(zip(ax_list, psi_list, im_list)):
            im.set_array(np.angle(psi[:,:,frame_num]))
            if max_normalization:
                im.set_alpha(np.abs(psi[:,:,frame_num])/psi_max_values[n])
            else:
                im.set_alpha(np.abs(psi[:,:,frame_num])/np.abs(psi[:,:,frame_num]).max().max())


        fig.savefig(file_path+f"{frame_num:05d}"+".png", bbox_inches='tight', pad_inches=0, dpi=psi_list[0].shape[0])


class eigenvector_finder:
    def __init__(self, O, H) -> None:
        self.O = O
        self.H = H

    def U(self, W):
        return W.T.conj(self.O(W))

    def getgrad(self, W):
        U_inv = np.linalg.inv(self.U(W))
        H_cached = self.H(W)
        return (H_cached - self.O(W@U_inv@W.T.conj()@H_cached))@ U_inv




# if __name__=='__main__':
#     file_name= input("name your video: ")
#     #potential


#     def initial_psi(X,Y):
#         return normalized_2d_gaussian(X,Y,0.5, dX=8) #* np.exp(1j*Y*4)
#     # #create simulation domain
#     test_particle = rectangular_1_particle_domain((25,25),30,initial_psi)

#     # #potential

#     V = normalized_2d_gaussian(test_particle.X,test_particle.Y, 6.25)*-70
#     #V[np.isnan(V)] =np.nanmin(V)*1.5


#     t, psi = test_particle.time_evolution(np.linspace(0,2000,1500), V)

    # np.save(file_name+"_psi.npy", psi)
    # np.save(file_name+"_t.npy", t)

    # print("Visualizing")
    # psi = np.load(file_name+"_psi.npy")
    # t = np.load(file_name+"_t.npy")
    # momentum = test_particle.momentum_space(psi)
    # # for i in range(psi.shape[-1]):
    # #     psi[:,:,i] = test_particle.laplacian(psi[:,:,i])

    # peak_psi_amplitude = np.abs(psi).max().max().max()
    # peak_mom_amplitude =  np.abs(momentum).max().max().max()
    # #print(peak_amplitude)


    # ###prep for animating
    # cyclic_cmap = create_cmap_from_csv("../CET_colormaps/", "CET-C6")


    # fig, (ax_pos,ax_mom) = plt.subplots(1,2, figsize=(test_particle.width*2,test_particle.height), frameon=False)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)



    # ####create images for animation###
    # ax_pos.set_axis_off()
    # ax_pos.imshow(np.zeros(psi.shape[:2]+(3,)))
    # im_pos = ax_pos.imshow(np.angle(psi[:,:,0]), vmin=-np.pi, vmax=np.pi, cmap=cyclic_cmap, alpha=np.abs(psi[:,:,0])/peak_psi_amplitude, interpolation='none')
    # im_pos.set_animated(True)

    # ax_mom.set_axis_off()
    # ax_mom.imshow(np.zeros(psi.shape[:2]+(3,)))
    # im_mom = ax_mom.imshow(np.angle(momentum[:,:,0]), vmin=-np.pi, vmax=np.pi,cmap=cyclic_cmap, alpha=np.abs(momentum[:,:,0])/peak_mom_amplitude, interpolation='none')
    # im_mom.set_animated(True)

    # ##### animate!!!!
    # def update(frame):
    #     #title = ax_pos.set_title(f"{frame}")
    #     im_pos.set_array(np.angle(psi[:,:,frame]))
    #     im_pos.set_alpha(np.abs(psi[:,:,frame])/np.abs(psi[:,:,frame]).max().max())

    #     im_mom.set_array(np.angle(momentum[:,:,frame]))
    #     im_mom.set_alpha(np.abs(momentum[:,:,frame])/np.abs(momentum[:,:,frame]).max().max())

    #     fig.savefig("./output/"+file_name+f"{frame:05d}"+".png", bbox_inches='tight', pad_inches=0, dpi=test_particle.resolution)
    #     return [im_pos, im_mom]

    # for i in range(t.shape[0]):
    #     update(i)

    # #ani = animation.FuncAnimation(fig=fig, func=update, frames=t.shape[0], interval=10)

    # ##output
    # #ani.save(filename="./"+file_name+".avi", writer="ffmpeg", fps=30, extra_args=['-vcodec', 'ffv1'])
    # plt.show()
