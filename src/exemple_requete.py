import json
from commun import FEATURES, charger

ligne = charger('test')[FEATURES].head(1).to_dict(orient='records')[0]
json.dump({'dataframe_records': [ligne]}, open('requete_valide.json', 'w'), indent=2)
json.dump({'dataframe_records': [{'produit': ligne['produit']}]},
          open('requete_invalide.json', 'w'), indent=2)    # colonnes manquantes
print('culture :', ligne['produit'], '| rendement actuel :', round(ligne['rend_t'], 2), 't/ha')