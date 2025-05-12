import numpy as np
from scipy.io.wavfile import write
import matplotlib
# out_path = "./eigens/"
# in_path = "./eigens/"
# file_prefix=input("file name: ")

def synthesizer(file_path, frequencies, coefficients, length, fadein=0.1, fadeout=0.1, rate = 44100, volume=0.3):
    time = np.linspace(0,length,np.int64(np.floor(rate*length)))

    fade_array = 1 - np.exp(-time/fadein) - np.exp(-(length-time)/fadeout)

    signal=np.zeros_like(time)
    for i,(frequency,coef) in enumerate(zip(frequencies,coefficients)):
        signal += np.real(coef * np.exp(1j*np.pi*2*frequency*time))

    signal *= fade_array * volume
    signal_scaled = np.int16(signal*volume / np.max(np.abs(signal)) * 32767)
    write(file_path, rate, signal)




# eigenvectors = np.load(in_path+file_prefix+"_eigenvectors.npy")
# eigenvalues = np.load(in_path+file_prefix+"_eigenvalues.npy")

# print(eigenvalues)




# fundemental_frequency = 200
# range =10

# eigenvalues_shifted = eigenvalues-eigenvalues.min()

# eigenvalues_shifted *= range/eigenvalues_shifted[2]

# eigenvalues_shifted += fundemental_frequency

# scale=np.ones_like(eigenvalues_shifted)
