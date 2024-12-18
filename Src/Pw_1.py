import os
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np
import librosa


from ModuleTPClassif import *

tout_mfcc=[]
chemin_fichiers = "/home/python/PW_Machine-Learning/Signaux"
prefixes = ['aa', 'ee', 'eh', 'eu', 'ii', 'oe', 'oh', 'oo', 'uu', 'yy']
for nom_fichier in os.listdir(chemin_fichiers):
    if nom_fichier.endswith(".wav"):
        echantillons,fe = librosa.load(os.path.join(chemin_fichiers, nom_fichier) , sr=None)
        
        prefixe = nom_fichier[:2]
        numero_classe = prefixes.index(prefixe)
        liste_coef=librosa.feature.mfcc(y=echantillons, sr=fe, hop_length=len(echantillons)+1 , win_length= len(echantillons),n_mfcc=10)
        tout_mfcc.append(liste_coef.T)


coef_mat = np.zeros((len(tout_mfcc), 10)) 

# Fill the matrix
for i, mfcc in enumerate(tout_mfcc):
    coef_mat[i, :] = mfcc

print (coef_mat)




