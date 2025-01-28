from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Définir la structure du réseau Bayésien
model = BayesianNetwork([('CC', 'Risque'), 
                         ('SC', 'Risque'), 
                         ('CV', 'Risque'), 
                         ('TV', 'Risque')])

# Définir les tables de probabilités conditionnelles (CPD)
cpd_cc = TabularCPD(variable='CC', variable_card=2, values=[[0.7], [0.3]])
cpd_sc = TabularCPD(variable='SC', variable_card=2, values=[[0.6], [0.4]])
cpd_cv = TabularCPD(variable='CV', variable_card=2, values=[[0.5], [0.5]])
cpd_tv = TabularCPD(variable='TV', variable_card=2, values=[[0.8], [0.2]])

# Corriger la table de probabilités conditionnelles pour 'Risque'
cpd_risque = TabularCPD(
    variable='Risque', 
    variable_card=2,
    values=[
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ],
    evidence=['CC', 'SC', 'CV', 'TV'],
    evidence_card=[2, 2, 2, 2]
)

# Ajouter les CPDs au modèle
model.add_cpds(cpd_cc, cpd_sc, cpd_cv, cpd_tv, cpd_risque)

# Vérifier si le modèle est valide
assert model.check_model()

# Faire des inférences
inference = VariableElimination(model)

# Exemple 1 : Probabilité que le conducteur soit "passionné de vitesse"
q1 = inference.query(variables=['CC'], evidence={'TV': 1})
print("Probabilité pour un conducteur passionné de vitesse :")
print(q1)

# Exemple 2 : Risque pour un conducteur sérieux de plus de 50 ans avec un véhicule de luxe
q2 = inference.query(variables=['Risque'], evidence={'CC': 0, 'SC': 0, 'CV': 1, 'TV': 1})
print("Risque pour un conducteur sérieux avec un véhicule de luxe :")
print(q2)
