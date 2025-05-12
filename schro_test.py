from numpy._core.numeric import inf
import Schro_Time_Dependant_2D as schro
import numpy as np
out_path = "./test/"
file_prefix=input("file name: ")
cmap=input("color map: ")
simulating= (input("are_you_simulating?: (y/n): ")=="y")




if simulating:
    domain = schro.rectangular_1_particle_domain((20,20),7)

    # V = np.log(((domain.X**2 + domain.Y**2)**1 ))*0.35
    # V[V==-inf] = V[V != -inf].min()
    # V -= 0.5
    V = (domain.X**2+domain.Y**2)*0.015
    print(V.max())
    initial_psi = schro.normalized_2d_gaussian(domain.X,domain.Y, 2.5, dX=2, dY=2) #* np.exp(1j*domain.Y*)

    t, psi = domain.time_evolution(initial_psi, np.linspace(0,1500,2000), V)

    domain_params = np.array([domain.width, domain.height, domain.resolution, domain.m, domain.h_bar])

    np.save(out_path+file_prefix+"_V.npy",V)
    np.save(out_path+file_prefix+"_domain_whrmh.npy",domain_params)
    np.save(out_path+file_prefix+"_psi.npy", psi)
    np.save(out_path+file_prefix+"_t.npy", t)
else:
    energy_shift = 0

    domain_params = np.load(out_path+file_prefix+"_domain_whrmh.npy")
    V=np.load(out_path+file_prefix+"_V.npy")
    domain = schro.rectangular_1_particle_domain((domain_params[0],domain_params[1]),domain_params[2])
    psi = np.load(out_path+file_prefix+"_psi.npy")
    t = np.load(out_path+file_prefix+"_t.npy")

    shift = np.exp(-1j * t * energy_shift)
    psi *= shift[np.newaxis, np.newaxis,:]


    print(
    np.real((psi[:,:,0].conj() * 1j* domain.d_psi_dt(psi[:,:,0],V)).sum().sum().sum())
    /(np.real((psi[:,:,0].conj()* psi[:,:,0]))).sum().sum().sum()
    ) #energy expectation

momentum = domain.momentum_space(psi)
print(t.shape)
schro.plot_and_save_psi_vs_t((psi,momentum), t, out_path+file_prefix+"_frame", upscale=4, cmap=cmap)
