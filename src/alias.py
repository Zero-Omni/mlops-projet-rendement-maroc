import sys
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')
c = MlflowClient()
prod = int(sys.argv[1])
c.set_registered_model_alias('rendement-maroc', 'production', prod)
c.update_model_version('rendement-maroc', prod,
                       description='Valide : bat la persistance sur le jeu de test')
for a in ['production']:
    v = c.get_model_version_by_alias('rendement-maroc', a)
    print(a, '-> version', v.version)