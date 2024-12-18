import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
date=20231130


def TesterClasses(NoClasses) :
	numMini=min(NoClasses)
	numMaxi=max(NoClasses)

	if numMini<1 :
		print(colored('Attention, les numeros de classe doivent commencer à 1 (ici {0}). La valeur 0 est réservée pour désigner l''ensemble des individus (voir la fonction CalculerCentresGravite).\n'.format(numMini), 'red'))
	for i in range(1, numMaxi+1) :   
		if len(np.argwhere(NoClasses==i))==0 :
			print(colored('Attention, la classe numero {0} est vide.\n'.format(i), 'red'))
         

def CalculerIndividusCentresReduits(Individus, CentresGravite) :
#---------------------------------------------------------------------------------
#function [IndividusCentresReduits]=CalculerIndividusCentresReduits(Individus, CentresGravite);
#
# Calcul des individus centrés réduits
#
#---------------------------------------------------------------------------------
# Entree : Individus                 [NbrIndividus x NbrParametres]
#          CentresGravite            [NbrClasses   x NbrParametres]
#
# Sortie : IndividusCentresReduits   [NbrIndividus x NbrParametres]
#---------------------------------------------------------------------------------
	NbrIndividus, NbrVariables = np.shape(Individus)

	IndividusCentres=Individus-np.ones((NbrIndividus,1))*CentresGravite[0,:].reshape((1,NbrVariables))
	CT=np.dot(IndividusCentres.T, IndividusCentres)/NbrIndividus
	IndividusCentresReduits=np.multiply(IndividusCentres, np.dot(np.ones((NbrIndividus,1)), np.power(np.diag(CT),-0.5).reshape((1,NbrVariables))))

	return IndividusCentresReduits

def CalculerCentresGravite(Individus, NoClasses) :
#---------------------------------------------------------------------------------
#function [CentresGravite]=CalculerCentresGravite(Individus, NoClasses);
#
# Calcul des centres de gravité de chaque classe.
#
#---------------------------------------------------------------------------------
# Entree : Individus                [NbrIndividus x NbrParametres]
#          NoClasses                [NbrIndividus]
#
# Sortie : CentresGravite           [NbrClasses   x NbrParametres]
#          Nb : CentreGravite[0] =  Centre gravite global
#---------------------------------------------------------------------------------

	TesterClasses(NoClasses)

	NbrIndividus, NbrVariables = np.shape(Individus)
	NbrClasses=np.max(NoClasses)+1

	CentresGravite=np.zeros((NbrClasses,NbrVariables))
	CentresGravite[0]=np.mean(Individus, axis=0)
	for q in range(1,NbrClasses) :
		IndClasse=np.argwhere(NoClasses==q)[:,0]
		CentresGravite[q]=np.mean(Individus[IndClasse], axis=0)
	return CentresGravite

def CalculerVariances(Individus, NoClasses, CentresGravite) :
#---------------------------------------------------------------------------------
# function [VT, VA, VE]=CalculerVariances(Individus, NoClasses, CentresGravite);
#
# Calcul des variances Totale, Intraclasses et Interclasses
#---------------------------------------------------------------------------------
# Entree : Individus          [NbrIndividus x NbrParametres]
#          NoClasses          [NbrIndividus x 1 ]
#          CentresGravite     [NbrClasses   x NbrParametres]
#
# Sortie : VT = Variance totale       [1 x 1]
#          VA = Variance intraclasses [1 x 1]
#          VE = Variance interclasses [1 x 1]
#----------------------------------------------------------------------------------

	TesterClasses(NoClasses)

	NbrIndividus, NbrVariables = np.shape(Individus)
	NbrClasses=np.max(NoClasses)+1

	VA=0
	VE=0
	if NbrClasses!=1 :
	### Variance intraclasses
		VA=0
		for q in range(1,NbrClasses) :
			IndClasse=np.argwhere(NoClasses==q)[:,0]
			LngIndClasse=np.size(IndClasse)
			vect= Individus[IndClasse] - np.ones((LngIndClasse,1))*CentresGravite[q]
			VA+= np.trace(np.dot(vect.T,vect))
		VA /= NbrIndividus

	### Variance interclasses
		VE=0
		for q in range(1,NbrClasses) :
			LngIndClasse=np.size(np.argwhere(NoClasses==q)[:,0])
			vect=CentresGravite[q]-CentresGravite[0]
			VE +=LngIndClasse*np.dot(vect.T, vect)
		VE/=NbrIndividus

### Variance totale

	vect=Individus-np.ones((NbrIndividus,1))*CentresGravite[0,:]
	VT=np.trace(np.dot(vect.T, vect)) / NbrIndividus

### Vérification

	if (abs(VT-VA-VE)>1e-9) and (NbrClasses!=1) :
		print(colored('Attention pb de calcul pour les variances (VT<>VA+VE)!\n VT={0}, VA={1}, VE={2}, difference={3}'.format(VT,VA,VE, VT-VA-VE), 'red'))

	return VT, VA, VE


def CalculerMatricesCovariance(Individus, NoClasses, CentresGravite):
#---------------------------------------------------------------------------------
#  function [CT, CA, CE]=CalculerMatricesCovariance(Individus, NoClasses, CentresGravite);
#
# Calcul des matrices de covariance Totale, Intraclasses et Interclasses
#---------------------------------------------------------------------------------
# Entree : Individus          [NbrIndividus x NbrParametres]
#          NoClasses          [NbrIndividus x 1 ]
#          CentresGravite     [NbrClasses   x NbrParametres]
#
# Sortie : CT = Matrice de covariance totale       [NbrParametres x NbrParametres]
#          CA = Matrice de covariance intraclasses [NbrParametres x NbrParametres]
#          CE = Matrice de covariance interclasses [NbrParametres x NbrParametres]
#----------------------------------------------------------------------------------

	TesterClasses(NoClasses)

	NbrIndividus, NbrVariables = np.shape(Individus)
	NbrClasses=np.max(NoClasses)+1


# Variables centrées

	IndividusCentres=Individus - np.dot(np.ones((NbrIndividus,1)), CentresGravite[0,:].reshape((1, NbrVariables)))

	if NbrClasses==1 :

		CA=np.zeros((NbrVariables, NbrVariables))
		CE=np.zeros((NbrVariables, NbrVariables))
	else :
	# Variables centrées par classe

		TabMoyenneClasse=np.zeros((NbrIndividus, NbrVariables))
		for q in range(1,NbrClasses):
			IndClasse=np.argwhere(NoClasses==q)[:,0]
			LngIndClasse=np.size(IndClasse)
			TabMoyenneClasse[IndClasse]=np.dot(np.ones((LngIndClasse,1)), CentresGravite[q,:].reshape((1, NbrVariables)))

		IndividusCentresClasse=Individus-TabMoyenneClasse;

	# Matrice de covariance intraclasses

		CA=np.dot( IndividusCentresClasse.T, IndividusCentresClasse ) / NbrIndividus

	# Matrice de covariance interclasses

		CE=np.zeros((NbrVariables, NbrVariables))
		for q in range(1, NbrClasses):
			IndClasse=np.argwhere(NoClasses==q)[:,0]
			LngIndClasse=np.size(IndClasse)
			vect=(CentresGravite[q] - CentresGravite[0]).reshape((NbrVariables,1))
			CE += LngIndClasse*np.dot(vect, vect.T)
		CE /= NbrIndividus

# Matrice de covariance totale

	CT=np.dot(IndividusCentres.T, IndividusCentres)/NbrIndividus

# Vérifications

	if (abs(np.sum(np.sum(CA+CE-CT)))>1e-9) and (NbrClasses!=1) :
		t, a, e = CalculerVariances(Individus, NoClasses, CalculerCentresGravite(Individus, NoClasses));

		if abs(np.trace(CT)-t)>1e-9:
			print(colored('Attention pb de calcul sur la matrice de covariance totale CT!\n','red'))
			CT=np.zeros((NbrVariables, NbrVariables))
		if abs(np.trace(CA)-a)>1e-9:
			print(colored('Attention pb de calcul sur la matrice de covariance intraclasses CA!\n','red'))
			CA=np.zeros((NbrVariables, NbrVariables))
		if abs(np.trace(CE)-e)>1e-9:
			print(colored('Attention pb de calcul sur la matrice de covariance interclasses CE!\n','red'))
			CE=np.zeros((NbrVariables, NbrVariables))


	return CT, CA, CE

def CalculerMatricesCovarianceInverseClasses(Individus, NoClasses, CentresGravite):
#---------------------------------------------------------------------------------
#  function [Cq_inv]=CalculerMatricesCovarianceInverseClasses(Individus, NoClasses, CentresGravite);
#
# Calcul des matrices de covariance inverse de chacune des classes
#---------------------------------------------------------------------------------
# Entree : Individus          [NbrIndividus x NbrParametres]
#          NoClasses          [NbrIndividus x 1 ]
#          CentresGravite     [NbrClasses   x NbrParametres]
#
# Sortie : Cq_inv = Liste des matrices de covariance inverse des classes    Q x [NbrParametres x NbrParametres]
#----------------------------------------------------------------------------------

	TesterClasses(NoClasses)

	NbrIndividus, NbrVariables = np.shape(Individus)
	NbrClasses=np.max(NoClasses)+1

# Variables centrées

	IndividusCentres=Individus - np.dot(np.ones((NbrIndividus,1)), CentresGravite[0,:].reshape((1, NbrVariables)))

	ListeMatricesCovariance=[]
	ListeMatricesCovarianceInverse=[]
	MatriceCovarianceClasse=np.zeros(( NbrVariables,  NbrVariables))
	ListeMatricesCovariance.append(MatriceCovarianceClasse)             # Indice 0 (vide)
	ListeMatricesCovarianceInverse.append(MatriceCovarianceClasse)      # Indice 0 (vide)
	
    # Calcul CA
	if NbrClasses==1 :
		CA=np.dot(IndividusCentres.T, IndividusCentres)/NbrIndividus
		ListeMatricesCovariance.append(CA)
		CAcontrole=CA
		
		if  not np.allclose(np.dot(MatriceCovarianceClasse, np.linalg.inv(MatriceCovarianceClasse)), np.eye(NbrVariables)) :
			print(colored('Attention : probleme lors de l''inversion de la matrice de covariance de la classe {0:} !\n'.format(q),'red'))
		else :
			ListeMatricesCovarianceInverse.append(np.linalg.inv(MatriceCovarianceClasse))
	else :
		# Variables centrées par classe

		TabMoyenneClasse=np.zeros((NbrIndividus, NbrVariables))
		for q in range(1,NbrClasses):
			IndClasse=np.argwhere(NoClasses==q)[:,0]
			LngIndClasse=np.size(IndClasse)
			TabMoyenneClasse[IndClasse]=np.dot(np.ones((LngIndClasse,1)), CentresGravite[q,:].reshape((1, NbrVariables)))

		IndividusCentresClasse=Individus-TabMoyenneClasse;
 
        # Matrice de covariance intraclasses

		CA=np.dot( IndividusCentresClasse.T, IndividusCentresClasse ) / NbrIndividus

	    # Matrices de covariance par classe

		CAcontrole=np.zeros(( NbrVariables,  NbrVariables))
		

		for q in range(1,NbrClasses):
			IndClasse=np.argwhere(NoClasses==q)[:,0]
			MatriceCovarianceClasse=np.dot( IndividusCentresClasse[IndClasse].T, IndividusCentresClasse[IndClasse] ) / LngIndClasse
			ListeMatricesCovariance.append(MatriceCovarianceClasse)
			CAcontrole= np.add(CAcontrole, LngIndClasse * MatriceCovarianceClasse / NbrIndividus)
			
			if  not np.allclose(np.dot(MatriceCovarianceClasse, np.linalg.inv(MatriceCovarianceClasse)), np.eye(NbrVariables)) :
				print(colored('Attention : probleme lors de l''inversion de la matrice de covariance de la classe {0:} !\n'.format(q),'red'))
				ListeMatricesCovarianceInverse.append(np.zeros(( NbrVariables,  NbrVariables)))
			else :
				ListeMatricesCovarianceInverse.append(np.linalg.inv(MatriceCovarianceClasse))

        # Vérification

		if ( abs(np.sum(np.sum(CA-CAcontrole))) > 1e-9 ) :
			print(colored('Attention pb de calcul sur les matrices de covariance des classes!\n','red'))
			print(CA,'\n\n', CAcontrole)
			ListeMatricesCovariance=[]

	return ListeMatricesCovarianceInverse
	
def PresenterClasses(Individus, NoClasses, Titre, CentresGravite=[], NomParametres=[], MaxGraphes=10, ParamX=0, ParamY=0):
#---------------------------------------------------------------------------
#
#     Présentation des individus et le centre de gravité des classes
#             à partir des paramètres pris deux par deux
#
#---------------------------------------------------------------------------
# Entree : Individus            [NbrIndividus x NbrParametres]
#          NoClasses            [NbrIndividus x 1 ]
#          Titre                [Chaine de caractères ]
#          CentresGravite       [NbrClasses   x NbrParametres] - Optionnel
#	   MaxGraphes           Scalaire                       - Optionnel
#	   ParamX               Scalaire                       - Optionnel
#	   ParamY               Scalaire                       - Optionnel
#
# Sortie : Visualisation graphique
#
#---------------------------------------------------------------------------

	TesterClasses(NoClasses)

	NbrIndividus, NbrParametres= np.shape(Individus)
	NbrClasses=np.max(NoClasses)
	
	motif=('1','v','2','s','3','p','4','*','^','+','x','.','o')
	if ((ParamX != 0) or (ParamY != 0)) and (ParamX != ParamY) :
		plt.figure(1)
		for q in range(1,NbrClasses+1) :
			IndClasse=np.argwhere(NoClasses==q)[:,0]
			plt.plot(Individus[IndClasse,ParamX],Individus[IndClasse,ParamY],  motif[q % len(motif)])
			if np.shape(CentresGravite)==(NbrClasses+1, NbrParametres) :
				plt.plot(CentresGravite[q,ParamX],CentresGravite[q,ParamY],'or')
				plt.title('{0} - paramètres {1} et {2}'.format(Titre, ParamX, ParamY))
			if len(NomParametres)==NbrParametres :
				plt.xlabel('- {0} -'.format(NomParametres[ParamX]))
				plt.ylabel('- {0} -'.format(NomParametres[ParamY]))
			else :    
				plt.xlabel('Paramètre n°{0}'.format(ParamX))
				plt.ylabel('Paramètre n°{0}'.format(ParamY))
		plt.grid()
		plt.show()
	else :
		nbrGraphes=0
		for i in range(1,NbrParametres) :
			nbrGraphes+=i
    
		if nbrGraphes>MaxGraphes :
			print(colored('Attention, le nombre de graphiques est limité à {0} : {1} paramètres pris deux à deux, donnent {2} graphiques.'.format(MaxGraphes, NbrParametres, nbrGraphes), 'red'))
			print(colored('En conséquence, les graphes ne seront pas affichés.\n', 'red'))
		else :	
			no=1;
			for i in range(NbrParametres-1) :
				for j in range(i+1,NbrParametres) :
					plt.figure(no)
					for q in range(1,NbrClasses+1) :
						IndClasse=np.argwhere(NoClasses==q)[:,0]
						plt.plot(Individus[IndClasse,i],Individus[IndClasse,j],  motif[q % len(motif)])
						if np.shape(CentresGravite)==(NbrClasses+1, NbrParametres) :
						    plt.plot(CentresGravite[q,i],CentresGravite[q,j],'or')
	
						plt.title('{0} - paramètres {1} et {2}'.format(Titre, i,j))
						if len(NomParametres)==NbrParametres :
							plt.xlabel('- {0} -'.format(NomParametres[i]))
							plt.ylabel('- {0} -'.format(NomParametres[j]))
						else :    
							plt.xlabel('Paramètre n°{0}'.format(i))
							plt.ylabel('Paramètre n°{0}'.format(j))
						no=no + 1;
					plt.grid()

			plt.show()