import pandas as pd
import mlflow
from commun import FEATURES, charger

mlflow.set_tracking_uri('http://127.0.0.1:5000')
m = mlflow.pyfunc.load_model('models:/rendement-maroc@production')
X = charger('prochaine')
res = X[['produit', 'filiere', 'annee_cible', 'rend_t']].copy()
res['rendement_prevu'] = m.predict(X[FEATURES]).round(3)
res['variation_%'] = ((res['rendement_prevu'] / res['rend_t'] - 1) * 100).round(1)
annee = int(res['annee_cible'].iloc[0])
nom = f'previsions/previsions_{annee}_{annee + 1}.csv'
res.sort_values('produit').to_csv(nom, index=False)
print(res.sort_values('produit').to_string(index=False))
print('fichier ecrit :', nom)