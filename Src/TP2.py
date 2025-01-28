from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Définir la structure du réseau bayésien
model = BayesianNetwork([
    ('CC', 'Accident'),
    ('SC', 'MedCost'),
    ('CV', 'ThisCarDam'),
    ('TV', 'Accident'),
    ('Accident', 'ThisCarDam'),
    ('Accident', 'OtherCarCost'),
    ('Accident', 'MedCost')
])

# Ajouter des données d'exemple (remplacer par des données réelles)
import pandas as pd

data = pd.DataFrame({
    'CC': ['Normal', 'Poor', 'Excellent'],
    'SC': ['Fair', 'Poor', 'Good'],
    'CV': ['Thousand', 'TenThou', 'HundredThou'],
    'TV': ['FamilySedan', 'SportsCar', 'Tank'],
    'Accident': ['None', 'Mild', 'Severe'],
    'ThisCarDam': ['None', 'Mild', 'Severe'],
    'OtherCarCost': ['Thousand', 'TenThou', 'HundredThou'],
    'MedCost': ['Thousand', 'TenThou', 'HundredThou']
})

# Estimation des probabilités à partir des données
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inférence
inference = VariableElimination(model)
result = inference.query(variables=['Accident'], evidence={'CC': 'Poor', 'TV': 'FamilySedan'})
print(result)
