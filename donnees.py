import sys
import pandas as pd

derniere = sys.argv[1] if len(sys.argv) > 1 else '2019/2020'

df = pd.read_excel('data/source/production-vegetale.xlsx')
df.columns = [c.strip() for c in df.columns]          # ' Valeur' a un espace
df = df.rename(columns={'Occurrence': 'campagne', 'Filière': 'filiere',
                        'Produit': 'produit', 'Indicateur': 'indicateur',
                        'Valeur': 'valeur'})
df['indicateur'] = df['indicateur'].map({'Production-T (Tonnes)': 'production_t',
                                         'Superficie (Ha)': 'superficie_ha'})

# format long -> une ligne par (campagne, filiere, produit)
large = df.pivot_table(index=['campagne', 'filiere', 'produit'],
                       columns='indicateur', values='valeur',
                       aggfunc='sum').reset_index()
large.columns.name = None
large = large[large['campagne'] <= derniere]          # '2019/2020' <= ... : ordre alphabetique = ordre chronologique

large.to_csv('data/production.csv', index=False)
print('data/production.csv ecrit :', len(large), 'lignes,',
      large['campagne'].nunique(), 'campagnes, derniere =', large['campagne'].max())