from TP1 import *

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_apprentissage, X_test, y_apprentissage, y_test = train_test_split(
    mfcc_centres_red, labels, test_size=0.1, random_state=42
)

def maximum_a_posteriori(X_train, y_train, X_test):
    classes = np.unique(y_train)
    probs = []

    for c in classes:
        X_c = X_train[y_train == c]
        mean = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False)
        prob = multivariate_normal(mean=mean, cov=cov).pdf(X_test)
        probs.append(prob)

    probs = np.array(probs)
    y_pred = np.argmax(probs, axis=0) + 1
    return y_pred

y_predict_map = maximum_a_posteriori(X_apprentissage, y_apprentissage, X_test)
precision_map = accuracy_score(y_test, y_predict_map)
print(f"\nTaux de reussite sur le Maximum à Posteriori : {precision_map:.2f}")

cm_map = confusion_matrix(y_test, y_predict_map)
ConfusionMatrixDisplay(confusion_matrix=cm_map).plot()
plt.title("Matrice de Confusion - Maximum à Posteriori")
plt.show()

# KMeans
Classif = KMeans(n_clusters=len(np.unique(y_apprentissage)), random_state=42)
Classif.fit(X_apprentissage)
y_predict = Classif.predict(X_test)
precision_kmeans1 = accuracy_score(y_test, y_predict)              
precision_kmeans2 = adjusted_rand_score(y_test, y_predict)           
print(f"\nTaux de reussite sur le KMeans avec accuracy_score: {precision_kmeans1:.2f}")
print(f"Taux de reussite sur le KMeans avec adjusted_rand_score: {precision_kmeans2:.2f}")


# Matrice de confusion pour KMeans
cm_kmeans = confusion_matrix(y_test, y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm_kmeans).plot()
plt.title("Matrice de Confusion - KMeans")
plt.show()

# Perceptron multicouche
Classif = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
Classif.fit(X_apprentissage, y_apprentissage)
y_predict = Classif.predict(X_test)
precision_mlp = accuracy_score(y_test, y_predict)
print(f"\nTaux de reussite sur la Perceptron Multicouche : {precision_mlp:.2f}")

# Matrice de confusion pour Perceptron Multicouche
cm_mlp = confusion_matrix(y_test, y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm_mlp).plot()
plt.title("Matrice de Confusion - Perceptron Multicouche")
plt.show()

# SVM
Classif = SVC(kernel='linear', random_state=42)
Classif.fit(X_apprentissage, y_apprentissage)
y_predict = Classif.predict(X_test)
precision_svm = accuracy_score(y_test, y_predict)
print(f"Taux de reussite SVM : {precision_svm:.2f}")

# Matrice de confusion pour SVM
cm_svm = confusion_matrix(y_test, y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm_svm).plot()
plt.title("Matrice de Confusion - SVM")
plt.show()

# Random Forest
Classif = RandomForestClassifier(n_estimators=100, random_state=42)
Classif.fit(X_apprentissage, y_apprentissage)
y_predict = Classif.predict(X_test)
precision_rf = accuracy_score(y_test, y_predict)
print(f"Taux de reussite sur la la Forêt Aléatoire : {precision_rf:.2f}")

# Matrice de confusion pour Random Forest
cm_rf = confusion_matrix(y_test, y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot()
plt.title("Matrice de Confusion - Forêt Aléatoire")
plt.show()



