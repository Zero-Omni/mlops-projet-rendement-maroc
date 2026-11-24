import os
import yaml
import joblib
from commun import FEATURES, CIBLE, charger, construire_modele

p = yaml.safe_load(open('params.yaml', encoding='utf-8'))['entrainement']
train = charger('train')                                  # entree : sortie de l'etape 1
modele = construire_modele(p).fit(train[FEATURES], train[CIBLE])
os.makedirs('models', exist_ok=True)
joblib.dump(modele, 'models/model.pkl')                   # sortie : le modele
print('entrainement : modele ecrit dans models/model.pkl')