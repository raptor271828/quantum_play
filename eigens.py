import numpy as np
import scipy.linalg as sp
import Schro_Time_Dependant_2D as schro
import matplotlib.pyplot as plt
#find eignenvectors via variational method (eignevalues must be bounded from below)

class eigenvector_finder:
    def __init__(self, O, H,num_basis, num_eigens, K=None) -> None:
        self.O = O
        self.H = H
        self.K = K
        #initial guess
        self.W_initial=np.random.rand(num_basis, num_eigens)+1j*np.random.rand(num_basis, num_eigens)
        self.W_initial=self.W_initial@np.linalg.inv(sp.sqrtm(self.U(self.W_initial))) #orthonormalize

    def find_eignens(self, Nit_lm, Nit_cg, conv_crit_cg, plot_out=None, pc=False):
        W, _ = self.lm(self.W_initial,Nit_lm, pc=pc, plot_out=plot_out)
        W, _ = self.cg(W,Nit_cg, conv_criterion=conv_crit_cg, plot_out=plot_out)

        eigenvectors, eigenvalues = self.getpsi(W)



        return eigenvectors, eigenvalues

    def U(self, W):
        return W.T.conj()@self.O(W)

    def dot(self, W1,W2):
        return np.real(W1.conj()*W2).sum()

    def getexpect(self, W):
            U_sqrt_inv = np.linalg.inv(sp.sqrtm(self.U(W)))
            return self.dot(W@U_sqrt_inv,self.H(W@U_sqrt_inv))

    def getgrad(self, W):
        U_inv = np.linalg.inv(self.U(W))
        H_cached = self.H(W)
        return (H_cached - self.O(W@U_inv@W.T.conj()@H_cached))@ U_inv


    def getpsi(self, W):
        U_cached = self.U(W)
        Y = W@np.linalg.inv(sp.sqrtm(U_cached))
        mu= Y.T.conj()@self.H(Y)
        epsilon,D=np.linalg.eig(mu) # This line does not need changing
        epsilon=np.real(epsilon) # Does not need change, removing numerical round off in imaginary component
        Psi= Y@D

        sort_indices = np.argsort(epsilon)

        return Psi[:,sort_indices],epsilon[sort_indices] # Does not need to be changed

    def cg(self, W, Nit, conv_criterion=0, plot_out=None):
        Elist=np.zeros(Nit+1,dtype=complex)
        Elist[0]=self.getexpect(W)
        alpha_trial=0.00005
        d = 0
        g = 0
        if plot_out != None:
            self.plot_and_save_minimization(W, plot_out+f"_cg_iter{0:04d}"+".png")

        for i in range(Nit):
            W_old = W
            d_old = d
            g_old = g

            g = self.getgrad(W_old)
            if i == 0 :
                d=-g
            else:
                beta = self.dot((g - g_old),g)/g_old_mag
                if beta < 0:
                    print("eep!")
                    beta = 0
                d=-g + beta * d_old

            # if i != 0 :
            #     print(f"linmin test{self.dot(g,d_old)*(self.dot(g,g) * self.dot(d_old,d_old))**-0.5}")
            #     print(f" cg test{self.dot(g,g_old)*(self.dot(g,g) * self.dot(g_old,g_old))**-0.5}")

            g_t = self.getgrad(W_old+alpha_trial*d)
            best_alpha = alpha_trial * self.dot(g,d)/(self.dot((g-g_t),d))

            W = W_old + best_alpha * d

            #printProgressBar(i,Nit+1,suffix=f"Complete E={getE(W)}")
            Elist[i+1]=self.getexpect(W)

            g_old_mag = self.dot(g,g)
            print(f"\r cg: iter {i/Nit:02f}, conv: {(Elist[i+1]-Elist[i])}, expectation:{Elist[i+1]}, grad:{g_old_mag}", end="")

            if plot_out != None:
                self.plot_and_save_minimization(W, plot_out+f"_cg_iter{i+1:04d}", form="psi")
                self.plot_and_save_minimization(W, plot_out+f"_cg_iter{i+1:04d}", form="Y")


            if g_old_mag < conv_criterion:
                Elist=Elist[:i+2]
                break
        return W, Elist


    def lm(self, W,Nit, conv_criterion=0, pc=False, plot_out=None):
        Elist=np.zeros(Nit+1,dtype=complex)
        Elist[0]=self.getexpect(W)
        alpha_trial=0.0001
        d = 0
        for i in range(Nit):
            W_old = W
            d_old = d

            g = self.getgrad(W_old)
            if pc:
                d=-self.K(g) #type: ignore
            else:
                d=-g

            # if i != 0 :
            #     print(f"linmin test{self.dot(g,d_old)/(self.dot(g,g) * self.dot(d_old,d_old))}")

            g_t = self.getgrad(W_old+alpha_trial*d)
            best_alpha = alpha_trial * self.dot(g,d)/(self.dot((g-g_t),d))

            W = W_old + best_alpha * d

            #printProgressBar(i,Nit+1,suffix=f"Complete E={getE(W)}")
            Elist[i+1]=self.getexpect(W)

            print(f"\r lm: iter {i/Nit:02f}, conv: {(Elist[i+1]-Elist[i])}, expectation:{Elist[i+1]}", end="")

            if plot_out != None:
                self.plot_and_save_minimization(W, plot_out+f"_lm_iter{i+1:04d}", form="psi")
                self.plot_and_save_minimization(W, plot_out+f"_lm_iter{i+1:04d}", form="Y")

            if (-(Elist[i+1]-Elist[i]) < conv_criterion) & ((Elist[i+1]-Elist[i])< 0):
                Elist=Elist[:i+2]
                break
            #printProgressBar(i+1,Nit,suffix=f"Complete")
        return W, Elist

    #assumes square domain
    def plot_and_save_minimization(self, W, file_path, cmap='CET-C6',upscale=3, form="psi"):
        cyclic_cmap = schro.create_cmap_from_csv("../CET_colormaps/", cmap)

        width = int(W.shape[0]**0.5)

        if form=="psi":
            wavefunctions, eigenvalues = self.getpsi(W)
        else:
            wavefunctions = W@np.linalg.inv(sp.sqrtm(self.U(W)))

        num_eigens = W.shape[-1]

        if num_eigens%8 == 0:
            fig, ax_list= plt.subplots(8, num_eigens//8, figsize=(num_eigens//8,8), frameon=False)
            ax_list = ax_list.flatten()
        elif num_eigens%4 == 0:
            fig, ax_list= plt.subplots(4, num_eigens//4, figsize=(num_eigens//4,4), frameon=False)
            ax_list = ax_list.flatten()
        elif num_eigens%5 == 0:
            fig, ax_list = plt.subplots(5,num_eigens//5, figsize=(num_eigens//5,5), frameon=False)
            ax_list = ax_list.flatten()
        elif num_eigens%2 == 0:
            fig, ax_list = plt.subplots(2,num_eigens//2, figsize=(num_eigens//2,2), frameon=False)
            ax_list = ax_list.flatten()
        elif num_eigens%3 == 0:
            fig, ax_list= plt.subplots(3, num_eigens//3, figsize=(num_eigens//3,3), frameon=False)
            ax_list = ax_list.flatten()
        else:
            fig, ax_list = plt.subplots(1,num_eigens, figsize=(num_eigens,1), frameon=False)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        if num_eigens ==1:
            ax_list = (ax_list,)

        psi_max_values = []
        for i, ax in enumerate(ax_list):
                #calculate global max (with repect to time) of max_normalization is on

            psi = wavefunctions[:,i].reshape((width, width))

            ax.set_axis_off()
            ax.imshow(np.zeros(psi.shape+(3,)))
            im = ax.imshow(np.angle(psi), vmin=-np.pi, vmax=np.pi, cmap=cyclic_cmap, alpha=np.abs(psi[:,:])/np.abs(psi[:,:]).max().max(), interpolation='none')

        fig.savefig(file_path+"_"+form+"_"+cmap+".png", bbox_inches='tight', pad_inches=0, dpi=width*upscale)
        plt.close()

    def eigen_decomposition(self, psi, eigenvectors, prenormalize=True): #psi should be nx1, eigennvectors nxm
        if prenormalize:
            psi = self.normalize(psi)
            eigenvectors = self.normalize(eigenvectors)
        return eigenvectors.T.conj() @ self.O(psi)

    def normalize(self, psis):
        inner_product = (psis.conj()* self.O(psis)).sum(axis=0)
        #print(inner_product)
        return psis / np.abs(inner_product)[np.newaxis,:]**0.5
