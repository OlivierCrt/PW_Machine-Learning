import os
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np

chemin_fichiers = "/home/python/PW_Machine-Learning/Signaux"
prefixes = ['aa', 'ee', 'eh', 'eu', 'ii', 'oe', 'oh', 'oo', 'uu', 'yy']

for nom_fichier in os.listdir(chemin_fichiers):
    if nom_fichier.endswith(".wav"):
        fe, echantillons = wavfile.read(os.path.join(chemin_fichiers, nom_fichier))
        prefixe = nom_fichier[:2]
        numero_classe = prefixes.index(prefixe)


plt.figure()
plt.plot(echantillons)
plt.title(f"Représentation temporelle du signal - Classe : {prefixe} ({numero_classe})")
plt.xlabel("Échantillons")
plt.ylabel("Amplitude")
plt.grid()

# Calcul de la Transformée de Fourier et de la DSP
tf_signal = fft(echantillons)
dsp = np.abs(tf_signal) ** 2  # Puissance spectrale
frequence = np.fft.fftfreq(len(echantillons), d=1/fe)

# Affichage de la DSP (seulement les fréquences positives)
plt.figure()
plt.plot(frequence[:len(frequence)//2], dsp[:len(dsp)//2])
plt.title(f"Densité Spectrale de Puissance - Classe : {prefixe} ({numero_classe})")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid()

# Afficher les figures avant de passer au fichier suivant
plt.show()