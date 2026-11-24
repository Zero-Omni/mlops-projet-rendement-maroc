import pandas as pd
import matplotlib
matplotlib.use('Agg')                                  # pas de fenetre graphique
import matplotlib.pyplot as plt

df = pd.read_csv('data/production.csv')
print('lignes :', len(df), '| campagnes :', df['campagne'].nunique())
print('\nproduits par filiere :')
print(df.groupby('filiere')['produit'].nunique().to_string())

print('\nvaleurs manquantes ou nulles :')
print('production_t manquante :', df['production_t'].isna().sum(),
      '| nulle :', (df['production_t'] <= 0).sum())
print('superficie_ha manquante :', df['superficie_ha'].isna().sum(),
      '| nulle :', (df['superficie_ha'] <= 0).sum())

df = df.dropna(subset=['production_t', 'superficie_ha'])
df = df[(df['production_t'] > 0) & (df['superficie_ha'] > 0)].copy()
df['rendement'] = df['production_t'] / df['superficie_ha']

g = df.groupby('produit').agg(campagnes=('campagne', 'nunique'),
                              superficie_moy=('superficie_ha', 'mean'),
                              rendement_moy=('rendement', 'mean'),
                              rendement_cv=('rendement', lambda s: s.std() / s.mean()))
print('\n10 cultures les plus etendues (superficie moyenne) :')
print(g.sort_values('superficie_moy', ascending=False).head(10).round(2).to_string())

top = g.sort_values('superficie_moy', ascending=False).head(6).index
plt.figure(figsize=(8, 4))
for p in top:
    d = df[df['produit'] == p].sort_values('campagne')
    plt.plot(d['campagne'], d['rendement'], 'o-', label=p)
plt.xticks(rotation=45)
plt.ylabel('rendement (t/ha)')
plt.yscale('log')
plt.legend(fontsize=7)
plt.title('Rendement des 6 cultures les plus etendues')
plt.savefig('automl/eda_rendements.png', dpi=110, bbox_inches='tight')
print('\ngraphique ecrit : automl/eda_rendements.png')