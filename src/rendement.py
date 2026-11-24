import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mlflow.set_tracking_uri('http://127.0.0.1:5000')
runs = mlflow.search_runs(experiment_names=['rendement-automl'])
auto = runs[runs['params.outil'] == 'flaml'].copy()
auto['budget'] = auto['params.budget_s'].astype(int)
auto = auto.drop_duplicates('budget').sort_values('budget')    # dernier run par budget
refs = runs[runs['params.outil'] == 'scikit-learn'].drop_duplicates('tags.mlflow.runName')

plt.figure(figsize=(7, 4))
plt.plot(auto['budget'], auto['metrics.mae'], 'o-', label='AutoML (FLAML)')
for _, r in refs.iterrows():
    nom, v = r['tags.mlflow.runName'], r['metrics.mae']
    plt.axhline(v, linestyle='--', color='gray', linewidth=1)
    plt.text(auto['budget'].min(), v, f'{nom} ({v:.3f})', fontsize=7, color='gray')
plt.xscale('log')
plt.xticks(sorted(auto['budget'].unique()), [str(b) for b in sorted(auto['budget'].unique())])
plt.minorticks_off()
plt.xlabel('budget de calcul (secondes, echelle log)')
plt.ylabel('MAE sur le jeu de test (t/ha, plus bas = mieux)')
plt.title('Rendement du budget de calcul')
plt.savefig('automl/rendement_budget.png', dpi=120, bbox_inches='tight')
print('graphique ecrit : automl/rendement_budget.png')