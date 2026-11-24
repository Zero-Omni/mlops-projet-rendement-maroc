import json
import subprocess
import joblib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from commun import FEATURES, charger

P = yaml.safe_load(open('params.yaml', encoding='utf-8'))
ticket = yaml.safe_load(open('data/production.csv.dvc', encoding='utf-8'))['outs'][0]
mets = json.load(open('reports/metrics.json'))
commit = subprocess.getoutput('git rev-parse --short HEAD')
n_lignes = len(pd.read_csv('data/production.csv'))
gain_min = P['validation']['gain_min']
accepte = mets['gain_vs_persistance'] >= gain_min          # REGLE DE VALIDATION

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('rendement-pipeline')

with mlflow.start_run(run_name=f"donnees-{ticket['md5'][:8]}"):
    mlflow.log_params({f'entrainement.{k}': v for k, v in P['entrainement'].items()})
    mlflow.log_params({f'donnees.{k}': v for k, v in P['donnees'].items()})
    mlflow.log_param('gain_min', gain_min)
    mlflow.log_metrics(mets)
    mlflow.set_tags({'commit': commit,                     # quel code
                     'donnees_md5': ticket['md5'],         # quelles donnees, exactement
                     'donnees_lignes': n_lignes,
                     'dernieres_campagnes_test': 'voir reports/metrics.json',
                     'auteur': '5IASD'})               # <-- a remplacer
    mlflow.log_artifact('dvc.lock')                        # photo complete du pipeline
    mlflow.log_artifacts('reports')                        # importances, erreurs, graphiques
    if accepte:
        exemple = charger('test')[FEATURES].head(3)
        mlflow.sklearn.log_model(joblib.load('models/model.pkl'), name='model',
                                 input_example=exemple,    # la signature en est deduite
                                 registered_model_name='rendement-maroc',
                                 serialization_format='cloudpickle')
        mlflow.set_tag('statut', 'accepte')
        print(f"modele accepte (gain {mets['gain_vs_persistance']} >= {gain_min}) et enregistre")
    else:
        mlflow.set_tag('statut', 'refuse')

if not accepte:
    raise SystemExit(f"modele refuse : gain {mets['gain_vs_persistance']} < seuil {gain_min}")