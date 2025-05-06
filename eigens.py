import numpy as np
import scipy.linalg as sp
#find eignenvectors via variational method (eignevalues must be bounded from below)

class eigenvector_finder:
    def __init__(self, O, H, num_basis, num_eigens) -> None:
        self.O = O
        self.H = H

        #initial guess
        self.W_initial=np.random.rand(num_basis, num_eigens)+1j*np.random.rand(num_basis, num_eigens)

    def find_eignens(self, Nit_lm, Nit_cg, conv_crit_cg):
        W, _ = self.lm(self.W_initial,Nit_lm)
        W, _ = self.lm(W,Nit_cg, conv_criterion=conv_crit_cg)

        eigenvectors, eigenvalues = self.getpsi(W)

        return eigenvectors, eigenvalues


    def U(self, W):
        return W.T.conj(self.O(W))

    def dot(self, W1,W2):
        return (W1.conj()*W2).sum()

    def getexpect(self, W):
        return self.dot(W,self.H(W))

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
        return Psi,epsilon # Does not need to be changed

    def cg(self, W, Nit, conv_criterion=0):
        Elist=np.zeros(Nit+1,dtype=complex)
        Elist[0]=self.getexpect(W)
        alpha_trial=0.00003
        d = 0
        g = 0
        for i in range(Nit):
            W_old = W
            d_old = d
            g_old = g

            g = self.getgrad(W_old)
            if i == 0 :
                d=-g
            else:
                beta = self.dot((g - g_old),g)/(self.dot(g_old, g_old))
                d=-g + beta * d_old

            # if i != 0 :
            #     print(f"linmin test{self.dot(g,d_old)*(self.dot(g,g) * self.dot(d_old,d_old))**-0.5}")
            #     print(f" cg test{self.dot(g,g_old)*(self.dot(g,g) * self.dot(g_old,g_old))**-0.5}")

            g_t = self.getgrad(W_old+alpha_trial*d)
            best_alpha = alpha_trial * self.dot(g,d)/(self.dot((g-g_t),d))

            W = W_old + best_alpha * d

            #printProgressBar(i,Nit+1,suffix=f"Complete E={getE(W)}")
            Elist[i+1]=self.getexpect(W)

            if (Elist[i+1]-Elist[i])/Elist[i+1] < conv_criterion:
                Elist=Elist[:i+2]
                break
                print(f"\r cg: iter {i/Nit:02f}, conv: {(Elist[i+1]-Elist[i])/Elist[i+1]}", end="")
        return W, Elist


    def lm(self, W,Nit, conv_criterion=0):
        Elist=np.zeros(Nit+1,dtype=complex)
        Elist[0]=self.getexpect(W)
        alpha_trial=0.00003
        d = 0
        for i in range(Nit):
            W_old = W
            d_old = d

            g = self.getgrad(W_old)
            d=-g

            # if i != 0 :
            #     print(f"linmin test{self.dot(g,d_old)/(self.dot(g,g) * self.dot(d_old,d_old))}")

            g_t = self.getgrad(W_old+alpha_trial*d)
            best_alpha = alpha_trial * self.dot(g,d)/(self.dot((g-g_t),d))

            W = W_old + best_alpha * d

            #printProgressBar(i,Nit+1,suffix=f"Complete E={getE(W)}")
            Elist[i+1]=self.getexpect(W)

            if (Elist[i+1]-Elist[i])/Elist[i+1] < conv_criterion:
                Elist=Elist[:i+2]
                break
            print(f"\r lm: iter {i/Nit:02f}, conv: {(Elist[i+1]-Elist[i])/Elist[i+1]}", end="")
            #printProgressBar(i+1,Nit,suffix=f"Complete")
        return W, Elist
