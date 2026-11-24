import sys
import time
import mlflow
from flaml import AutoML
from commun import FEATURES, CIBLE, charger, metriques

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60
tr, va, te = charger('train'), charger('val'), charger('test')
journal = f'automl/flaml_{budget}s.log'

a = AutoML()
t0 = time.time()
a.fit(tr[FEATURES], tr[CIBLE], X_val=va[FEATURES], y_val=va[CIBLE],
      task='regression', metric='mae', time_budget=budget, seed=42,
      verbose=0, log_file_name=journal, log_type='all')       # trace de TOUS les essais
duree = round(time.time() - t0, 1)
mets = metriques(te[CIBLE], a.predict(te[FEATURES]), te['rend_t'])

print(f"budget {budget} s | famille : {a.best_estimator} | MAE test = {mets['mae']} "
      f"| gain vs persistance = {mets['gain_vs_persistance']} | {duree} s")
print('meilleure config :', a.best_config)

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('rendement-automl')
with mlflow.start_run(run_name=f'flaml-{budget}s'):
    mlflow.log_params({'outil': 'flaml', 'budget_s': budget, 'famille': a.best_estimator})
    mlflow.log_metrics({**mets, 'mae_validation': round(a.best_loss, 4), 'duree_s': duree})
    mlflow.log_artifact(journal)