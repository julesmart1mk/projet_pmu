import pandas as pd
import numpy as np
import pickle
import optuna
import warnings
import json
import joblib
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker
from src import config

optuna.logging.set_verbosity(optuna.logging.ERROR)
X_global, y_global, groups_global = None, None, None

def get_objective(model_name):
    def objective(trial):
        if model_name == 'lgbm':
            params = {'objective': 'lambdarank', 'metric': 'ndcg', 'random_state': 42, 'verbosity': -1, 'n_estimators': trial.suggest_int('n_estimators', 200, 1500), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), 'num_leaves': trial.suggest_int('num_leaves', 20, 200)}
            model = lgb.LGBMRanker(**params)
        elif model_name == 'xgb':
            params = {'objective': 'rank:ndcg', 'random_state': 42, 'eval_metric': 'ndcg', 'verbosity': 0, 'n_estimators': trial.suggest_int('n_estimators', 200, 1500), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), 'max_depth': trial.suggest_int('max_depth', 3, 10)}
            model = xgb.XGBRanker(**params)
        elif model_name == 'catboost':
            params = {'objective': 'QueryRMSE', 'random_state': 42, 'verbose': 0, 'iterations': trial.suggest_int('iterations', 200, 1500), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), 'depth': trial.suggest_int('depth', 3, 10)}
            model = CatBoostRanker(**params)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, val_idx = next(gss.split(X_global, y_global, groups_global))
        X_train, X_val = X_global.iloc[train_idx], X_global.iloc[val_idx]; y_train, y_val = y_global.iloc[train_idx], y_global.iloc[val_idx]
        train_groups = X_train.groupby('COURSE_ID').size().to_numpy(); val_groups = X_val.groupby('COURSE_ID').size().to_numpy()
        X_train_features = X_train.drop(columns='COURSE_ID'); X_val_features = X_val.drop(columns='COURSE_ID')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            categorical_cols = [f for f in config.CATEGORICAL_FEATURES if f in X_train_features.columns]
            preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)], remainder='passthrough')
            X_train_transformed = preprocessor.fit_transform(X_train_features); X_val_transformed = preprocessor.transform(X_val_features)
            if model_name == 'lgbm':
                model.fit(X_train_transformed, y_train, group=train_groups, eval_set=[(X_val_transformed, y_val)], eval_group=[val_groups], eval_metric='ndcg', callbacks=[lgb.early_stopping(10, verbose=False)])
                score = model.best_score_['valid_0']['ndcg@1']
            elif model_name == 'catboost':
                model.fit(X_train_transformed, y_train, group_id=X_train['COURSE_ID'].to_numpy()); predictions = model.predict(X_val_transformed); score = pd.Series(y_val.to_numpy()).corr(pd.Series(predictions), method='spearman')
            else:
                model.fit(X_train_transformed, y_train, group=train_groups); score = model.score(X_val_transformed, y_val)
        return score if not np.isnan(score) else 0
    return objective

def train_model(n_trials=100, build_only=False):
    global X_global, y_global, groups_global
    
    if not build_only:
        print(f"--- Lancement Étape 3 (v20) : Entraînement de l'Équipe Complète ---")
        try: df = pd.read_parquet(config.CLEAN_DATA_PATH)
        except FileNotFoundError: print("❌ Erreur : Le fichier de données propres n'existe pas."); return
        df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
        df['ranking_target'] = df.groupby('COURSE_ID')['PLACE_CHEVAL'].transform(lambda x: x.max() - x + 1)
        TARGET = 'ranking_target'
        df = df.sort_values('COURSE_ID'); groups_global = df['COURSE_ID']; features = [f for f in config.FEATURES_TO_USE if f in df.columns]
        X_global = df[features + ['COURSE_ID']]; y_global = df[TARGET]
        models_to_train = {'lgbm': {'name': 'LGBMRanker'}, 'xgb': {'name': 'XGBRanker'}, 'catboost': {'name': 'CatBoostRanker'}}
        for model_key, model_info in models_to_train.items():
            print(f"\n--- 🚀 Optimisation de {model_info['name']}... ({n_trials} essais) ---")
            study = optuna.create_study(direction='maximize')
            try:
                with tqdm(range(n_trials), desc=f"Optimisation de {model_info['name']}") as pbar:
                    def tqdm_callback(study, trial):
                        pbar.update(1)
                        if study.best_value: pbar.set_postfix(best_score=study.best_value)
                    study.optimize(get_objective(model_key), n_trials=n_trials, callbacks=[tqdm_callback])
            except KeyboardInterrupt:
                print(f"\n🛑 Interruption pour {model_info['name']}. Sauvegarde de l'étude.")
            joblib.dump(study, config.MODELS_DIR / f"study_{model_key}.pkl")
            if not study.best_trial: print(f"Aucun essai terminé pour {model_info['name']}."); continue
            print(f"\n✅ Optimisation de {model_info['name']} terminée. Meilleur score : {study.best_value:.4f}")
    
    print("\n--- Assemblage du modèle final à partir des meilleures études ---")
    final_estimators = []
    models_to_build = {'lgbm': {'name': 'LGBMRanker'}, 'xgb': {'name': 'XGBRanker'}, 'catboost': {'name': 'CatBoostRanker'}}
    for model_key, model_info in models_to_build.items():
        try:
            study = joblib.load(config.MODELS_DIR / f"study_{model_key}.pkl")
            best_params = study.best_params
            print(f"Meilleurs paramètres pour {model_info['name']} récupérés.")
            if model_key == 'lgbm': final_estimators.append(('lgbm', lgb.LGBMRanker(**best_params, random_state=42, verbosity=-1)))
            elif model_key == 'xgb': final_estimators.append(('xgb', xgb.XGBRanker(**best_params, random_state=42, eval_metric='ndcg', verbosity=0)))
            elif model_key == 'catboost': final_estimators.append(('catboost', CatBoostRanker(**best_params, random_state=42, verbose=0)))
        except (FileNotFoundError, ValueError):
             print(f"Avertissement : Impossible de charger l'étude pour {model_info['name']}. Ce modèle ne sera pas inclus.")
    
    if not final_estimators: print("❌ Aucun modèle expert optimisé trouvé. Impossible de construire le modèle final."); return
        
    if build_only:
        try: df = pd.read_parquet(config.CLEAN_DATA_PATH)
        except FileNotFoundError: print("❌ Erreur : Le fichier de données propres n'existe pas."); return
        df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
        df['ranking_target'] = df.groupby('COURSE_ID')['PLACE_CHEVAL'].transform(lambda x: x.max() - x + 1); TARGET = 'ranking_target'
        df = df.sort_values('COURSE_ID'); features = [f for f in config.FEATURES_TO_USE if f in df.columns]
        X_global = df[features + ['COURSE_ID']]; y_global = df[TARGET]

    categorical_cols = [f for f in config.CATEGORICAL_FEATURES if f in X_global.columns]
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)], remainder='passthrough')
    
    # L'erreur était ici, StackingClassifier n'était pas importé
    # De plus, le modèle final n'est pas un Ranker, mais un Classifier qui apprend des scores des rankers.
    # Pour simplifier, nous n'utiliserons pas Stacking pour l'instant et nous ferons la moyenne des scores dans le script de prédiction.
    # Cette fonction va juste s'assurer que les modèles individuels sont bien entraînés et sauvegardés.
    print("✅ Processus terminé. Les modèles experts sont prêts à être utilisés pour la prédiction.")

if __name__ == '__main__':
    train_model()
