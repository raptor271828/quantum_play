from os import W_OK
import Schro_Time_Dependant_2D as schro
import eigens
import numpy as np
out_path = "./eigens/"
in_path = "./test/"
file_prefix=input("file name: ")
minimizing= (input("are_you_minizing? (y/n): ")=="y")

domain_params = np.load(in_path+file_prefix+"_domain_whr.npy")
V=np.load(in_path+file_prefix+"_V.npy")
print(V)
psi = np.load(in_path+file_prefix+"_psi.npy")
#t = np.load(out_path+file_prefix+"_t.npy")
domain = schro.rectangular_1_particle_domain((domain_params[0],domain_params[1]),domain_params[2])




if minimizing:

    #plane wave basis
    def O_for_minimizer(W_unraveled):
        return W_unraveled #* domain.width * domain.height

    # def K_for_minimizer(W_unraveled):
    #     return W_unraveled * (domain.G_sqr.reshape((domain.num_X*domain.num_Y,1)) + 1)**-1

    def H_for_minimizer(W_unraveled, m=1, hbar=1):

        H = (domain.H(W_unraveled.reshape(domain.num_X,domain.num_Y,-1), V)).reshape(domain.num_X*domain.num_Y,-1)

        return O_for_minimizer(H)

    finder = eigens.eigenvector_finder(O_for_minimizer,H_for_minimizer,domain.num_X*domain.num_Y ,5)# K=K_for_minimizer)
    eigenvectors, eigenvalues = finder.find_eignens(40,0,0, pc=False)

    np.save(out_path+file_prefix+"_eigenvectors.npy", eigenvectors)
    np.save(out_path+file_prefix+"_eigenvalues.npy", eigenvalues)
    print(eigenvalues)


else:
    eigenvectors = np.load(out_path+file_prefix+"_eigenvectors.npy")
    eigenvalues = np.load(out_path+file_prefix+"_eigenvalues.npy")

eigenvectors_raveled = eigenvectors.reshape((domain.num_X, domain.num_Y, -1))

    # eignenvectors_raveled_list = []
    # for i in range(eigenvectors_raveled.shape[-1]):
    #     eignenvectors_raveled_list.append(eigenvectors_raveled[:,:,i, np.newaxis]) #extra unused dimension for
    #
print(eigenvectors_raveled.shape)
schro.plot_and_save_psi_vs_t((eigenvectors_raveled,), np.ones(eigenvectors.shape[-1]), "./eigens/vector")
