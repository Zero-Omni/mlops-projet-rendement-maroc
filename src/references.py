import time
import yaml
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from commun import FEATURES, FEATURES_CAT, FEATURES_NUM, CIBLE, charger, construire_modele, metriques

P = yaml.safe_load(open('params.yaml', encoding='utf-8'))
tr, te = charger('train'), charger('test')

ridge = Pipeline([('prep', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES_CAT),
                                              ('num', StandardScaler(), FEATURES_NUM)])),
                  ('m', Ridge(alpha=1.0))])
modeles = {'reference-persistance': None,
           'reference-ridge': ridge,
           'manuel-rf': construire_modele(P['entrainement'])}

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('rendement-automl')
for nom, m in modeles.items():
    t0 = time.time()
    if m is None:
        pred = te['rend_t']                                  # persistance : rendement de l'an dernier
    else:
        pred = m.fit(tr[FEATURES], tr[CIBLE]).predict(te[FEATURES])
    duree = round(time.time() - t0, 2)
    mets = metriques(te[CIBLE], pred, te['rend_t'])
    with mlflow.start_run(run_name=nom):
        mlflow.log_params({'outil': 'scikit-learn', 'approche': nom})
        mlflow.log_metrics({**mets, 'duree_s': duree})
    print(f"{nom:24s} MAE = {mets['mae']:<8} gain vs persistance = {mets['gain_vs_persistance']:<8} ({duree} s)")