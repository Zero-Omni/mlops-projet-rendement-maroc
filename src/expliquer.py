import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from commun import FEATURES, charger

modele = joblib.load('models/model.pkl')
X = charger('test')[FEATURES]
Xt = modele.named_steps['prep'].transform(X)
if hasattr(Xt, 'toarray'):
    Xt = Xt.toarray()
noms = modele.named_steps['prep'].get_feature_names_out()
sv = shap.TreeExplainer(modele.named_steps['rf']).shap_values(Xt)
shap.summary_plot(sv, Xt, feature_names=noms, show=False)
plt.savefig('automl/shap_summary.png', dpi=110, bbox_inches='tight')
print('graphique ecrit : automl/shap_summary.png')