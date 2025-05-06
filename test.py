from numpy._core.numeric import inf
import Schro_Time_Dependant_2D as schro
import numpy as np
out_path = "./test/"

domain = schro.rectangular_1_particle_domain((20,20),5)

V = np.log((domain.X**2 + domain.Y**2)**0.5)*100
V[V==-inf] = V[V != -inf].min()
V = 0
initial_psi = schro.normalized_2d_gaussian(domain.X,domain.Y,0.5, dX=4) * np.exp(1j*domain.Y*7)

# t, psi = domain.time_evolution(initial_psi, np.linspace(0,150,150), V)

# domain_params = np.array([domain.width, domain.height, domain. resolution])

# np.save(out_path+"_V.npy",V)
# np.save(out_path+"_domain_whr.npy",domain_params)
# np.save(out_path+"_psi.npy", psi)
# np.save(out_path+"_t.npy", t)

psi = np.load(out_path+"_psi.npy")
t = np.load(out_path+"_t.npy")

momentum = domain.momentum_space(psi)
print(t.shape)
schro.plot_and_save_psi_vs_t((psi,momentum), t, out_path+"frame")
