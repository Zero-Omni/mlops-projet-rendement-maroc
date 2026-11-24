import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES_CAT = ['produit', 'filiere']
FEATURES_NUM = ['rend_t', 'rend_lag1', 'rend_lag2', 'rend_moy3', 'superficie_ha',
                'var_superficie', 'production_t', 'annee', 'anomalie_nationale']
FEATURES = FEATURES_CAT + FEATURES_NUM
CIBLE = 'cible'          # rendement (t/ha) de la campagne SUIVANTE


def charger(nom):
    df = pd.read_csv(f'data/prepare/{nom}.csv')
    df[FEATURES_NUM] = df[FEATURES_NUM].astype('float64')   # signature MLflow stable
    return df


def construire_modele(p):
    prep = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'),
                               FEATURES_CAT)], remainder='passthrough')
    rf = RandomForestRegressor(n_estimators=p['n_estimators'],
                               max_depth=p['max_depth'],
                               min_samples_leaf=p['min_samples_leaf'],
                               random_state=p['random_state'])
    return Pipeline([('prep', prep), ('rf', rf)])


def metriques(y, pred, y_naif):
    """MAE/RMSE en t/ha, R2, erreur relative ponderee, et gain face a la persistance."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = mean_absolute_error(y, pred)
    mae_naif = mean_absolute_error(y, y_naif)
    return {'mae': round(float(mae), 4),
            'rmse': round(float(np.sqrt(mean_squared_error(y, pred))), 4),
            'r2': round(float(r2_score(y, pred)), 4),
            'wmape': round(float(np.abs(y - pred).sum() / np.abs(y).sum()), 4),
            'mae_persistance': round(float(mae_naif), 4),
            'gain_vs_persistance': round(float(1 - mae / mae_naif), 4)}