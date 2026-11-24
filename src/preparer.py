import os
import yaml
import pandas as pd

P = yaml.safe_load(open('params.yaml', encoding='utf-8'))
pd_, pp = P['donnees'], P['preparation']

df = pd.read_csv('data/production.csv')
df = df.dropna(subset=['production_t', 'superficie_ha'])
df = df[(df['production_t'] > 0) & (df['superficie_ha'] > 0)]
df = df[~df['filiere'].isin(pd_['filieres_exclues'])].copy()
df['annee'] = df['campagne'].str[:4].astype(int)         # '2012/2013' -> 2012
df['rend_t'] = df['production_t'] / df['superficie_ha']  # rendement t/ha

# 1) principales cultures : assez de campagnes, puis les plus etendues
stats = df.groupby('produit').agg(n=('annee', 'nunique'), sup=('superficie_ha', 'mean'))
stats = stats[stats['n'] >= pd_['min_campagnes']].sort_values('sup', ascending=False)
cultures = stats.head(pd_['nb_cultures']).index.tolist()
df = df[df['produit'].isin(cultures)].sort_values(['produit', 'annee']).reset_index(drop=True)
print('cultures retenues (', len(cultures), ') :', cultures)

# 2) decalages par culture, valides seulement si les annees sont consecutives
g = df.groupby('produit')


def decale(col, k):
    v = g[col].shift(k)
    ok = g['annee'].shift(k) == df['annee'] - k
    return v.where(ok)


df['rend_lag1'] = decale('rend_t', 1)
df['rend_lag2'] = decale('rend_t', 2)
df['sup_lag1'] = decale('superficie_ha', 1)
df['var_superficie'] = df['superficie_ha'] / df['sup_lag1'] - 1
df['rend_moy3'] = df[['rend_t', 'rend_lag1', 'rend_lag2']].mean(axis=1)
# ecart du rendement courant a la moyenne des 2 campagnes precedentes ...
df['ecart_rend'] = df['rend_t'] / df[['rend_lag1', 'rend_lag2']].mean(axis=1) - 1
# ... moyenne sur toutes les cultures : "la campagne t etait-elle bonne ou mauvaise ?"
df['anomalie_nationale'] = df.groupby('annee')['ecart_rend'].transform('mean')

df['cible'] = decale('rend_t', -1)                      # rendement de la campagne t+1
df['production_suivante'] = decale('production_t', -1)  # NON utilisee par le modele (voir fuite, Bloc E)
df['annee_cible'] = df['annee'] + 1

df = df.dropna(subset=['rend_lag1', 'rend_lag2'])
colonnes = ['produit', 'filiere', 'campagne', 'annee_cible', 'rend_t', 'rend_lag1',
            'rend_lag2', 'rend_moy3', 'superficie_ha', 'var_superficie', 'production_t',
            'annee', 'anomalie_nationale', 'cible', 'production_suivante']

# 3) lignes a prevoir : la derniere campagne connue (cible inconnue)
prochaine = df[df['annee'] == df['annee'].max()][colonnes]
# 4) lignes d'apprentissage : cible connue
appr = df.dropna(subset=['cible'])[colonnes]

# 5) decoupage CHRONOLOGIQUE sur la campagne cible
cibles = sorted(appr['annee_cible'].unique())
nt, nv = pp['n_campagnes_test'], pp['n_campagnes_val']
test_c, val_c = cibles[-nt:], cibles[-(nt + nv):-nt]
test = appr[appr['annee_cible'].isin(test_c)]
val = appr[appr['annee_cible'].isin(val_c)]
train = appr[appr['annee_cible'] < min(val_c)]

os.makedirs('data/prepare', exist_ok=True)
for nom, d in [('train', train), ('val', val), ('test', test), ('prochaine', prochaine)]:
    d.to_csv(f'data/prepare/{nom}.csv', index=False)
print('preparation :', len(train), 'train /', len(val), 'val /', len(test), 'test /',
      len(prochaine), 'a prevoir')
print('campagnes cibles -> test :', test_c, '| val :', val_c, '| train jusqu\'a :', min(val_c) - 1)