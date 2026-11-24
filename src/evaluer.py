import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from commun import FEATURES, CIBLE, charger, metriques

p = yaml.safe_load(open('params.yaml', encoding='utf-8'))['evaluation']
modele = joblib.load('models/model.pkl')
val, test = charger('val'), charger('test')

m_val = metriques(val[CIBLE], modele.predict(val[FEATURES]), val['rend_t'])
pred = modele.predict(test[FEATURES])
mets = metriques(test[CIBLE], pred, test['rend_t'])       # metriques de reference = TEST
mets['val_mae'] = m_val['mae']
mets['val_gain_vs_persistance'] = m_val['gain_vs_persistance']
mets['n_test'] = int(len(test))

os.makedirs('reports', exist_ok=True)
json.dump(mets, open('reports/metrics.json', 'w'), indent=2)

# --- explicabilite : importance par permutation (validation + test, plus stable) ---
ref = pd.concat([val, test])
r = permutation_importance(modele, ref[FEATURES], ref[CIBLE], n_repeats=p['n_repeats'],
                           random_state=0, scoring='neg_mean_absolute_error')
imp = pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)
json.dump({k: round(float(v), 4) for k, v in imp.items()},
          open('reports/importances.json', 'w'), indent=2)
plt.figure(figsize=(7, 4))
imp[::-1].plot.barh()
plt.xlabel('hausse de la MAE (t/ha) quand la variable est melangee')
plt.title('Importance par permutation')
plt.savefig('reports/importances.png', dpi=110, bbox_inches='tight')
plt.close()

# --- erreur par culture et prediction vs realite ---
t = test.assign(pred=pred, erreur_abs=np.abs(test[CIBLE] - pred))
par_culture = (t.groupby('produit')
                .agg(mae=('erreur_abs', 'mean'), rendement_reel_moyen=(CIBLE, 'mean'))
                .round(3).sort_values('mae'))
par_culture.to_csv('reports/erreurs_par_culture.csv')
plt.figure(figsize=(5, 5))
plt.scatter(test[CIBLE], pred, s=15)
lim = [test[CIBLE].min() * 0.9, test[CIBLE].max() * 1.1]
plt.plot(lim, lim, 'k--', linewidth=1)
plt.xscale('log'); plt.yscale('log')
plt.xlabel('rendement reel (t/ha)'); plt.ylabel('rendement prevu (t/ha)')
plt.title('Prevision vs realite (jeu de test)')
plt.savefig('reports/prediction_vs_reel.png', dpi=110, bbox_inches='tight')
print('evaluation :', mets)
print('importances :', dict(imp.round(3).head(5)))