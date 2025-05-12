from os import W_OK
import Schro_Time_Dependant_2D as schro
import eigens
import hear_eigens
import numpy as np
out_path = "./eigens/"
in_path = "./test/"
file_prefix=input("file name: ")
minimizing= (input("are_you_minizing? (y/n): ")=="y")

domain_params = np.load(in_path+file_prefix+"_domain_whrmh.npy")
V=np.load(in_path+file_prefix+"_V.npy")
print(V)
psi = np.load(in_path+file_prefix+"_psi.npy")
t = np.load(in_path+file_prefix+"_t.npy")
domain = schro.rectangular_1_particle_domain((domain_params[0],domain_params[1]),domain_params[2], m=domain_params[3], h_bar=domain_params[4])
domain.m = 0.11

    #plane wave basis9
def O_for_minimizer(W_unraveled):
    return W_unraveled #* domain.width * domain.height

    # def K_for_minimizer(W_unraveled):
    #     return W_unraveled * (domain.G_sqr.reshape((domain.num_X*domain.num_Y,1)) + 1)**-1

def H_for_minimizer(W_unraveled):

    H = (domain.H(W_unraveled.reshape(domain.num_X,domain.num_Y,-1), V)).reshape(domain.num_X*domain.num_Y,-1)

    return O_for_minimizer(H)

finder = eigens.eigenvector_finder(O_for_minimizer,H_for_minimizer,domain.num_X*domain.num_Y ,64)# K=K_for_minimizer)

if minimizing:

    eigenvectors, eigenvalues = finder.find_eignens(0,450,1e-14, pc=False, plot_out=out_path+file_prefix+"8x8")

    np.save(out_path+file_prefix+"_eigenvectors.npy", eigenvectors)
    np.save(out_path+file_prefix+"_eigenvalues.npy", eigenvalues)
    print(eigenvalues)


else:
    eigenvectors = np.load(out_path+file_prefix+"_eigenvectors.npy")
    eigenvalues = np.load(out_path+file_prefix+"_eigenvalues.npy")

    coeficients = finder.eigen_decomposition(psi[:,:,0].reshape(domain.num_X*domain.num_Y,1), eigenvectors)

    print(np.abs(coeficients)**2)
    print((np.abs(coeficients)**2).sum())

    fps=45
    t_step_per_frame = t[1]-t[0]
    t_sim_per_second = t_step_per_frame*fps

    frequencies = eigenvalues/(np.pi*2*domain.h_bar) * t_sim_per_second

    length = t.shape[0]/fps

    base_frequency = 50
    #frequencies += frequencies - frequencies.min()
    frequencies *= 50/frequencies.min()
    #frequencies +=base_frequency

    hear_eigens.synthesizer(out_path+file_prefix+".wav", frequencies, coeficients,length)



eigenvectors_raveled = eigenvectors.reshape((domain.num_X, domain.num_Y, -1))

    # eignenvectors_raveled_list = []
    # for i in range(eigenvectors_raveled.shape[-1]):
    #     eignenvectors_raveled_list.append(eigenvectors_raveled[:,:,i, np.newaxis]) #extra unused dimension for
    #
print(eigenvectors_raveled.shape)
schro.plot_and_save_psi_vs_t((eigenvectors_raveled,), np.ones(eigenvectors.shape[-1]), out_path+file_prefix)
