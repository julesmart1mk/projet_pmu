import pandas as pd
import numpy as np
import pickle
import optuna
import warnings
import json
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker
from projet_pmu.src import config
import time
from tqdm import tqdm
import contextlib
import io

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

X_global, y_global, groups_global = None, None, None

def get_objective(model_name):
    def objective(trial):
        if model_name == 'lgbm':
            params = {'objective': 'lambdarank', 'metric': 'ndcg', 'random_state': 42, 'verbosity': -1, 'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 'num_leaves': trial.suggest_int('num_leaves', 20, 300)}
            model = lgb.LGBMRanker(**params)
        elif model_name == 'xgb':
            params = {'objective': 'rank:ndcg', 'random_state': 42, 'eval_metric': 'ndcg', 'verbosity': 0, 'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 'max_depth': trial.suggest_int('max_depth', 3, 12)}
            model = xgb.XGBRanker(**params)
        elif model_name == 'catboost':
            params = {'objective': 'QueryRMSE', 'random_state': 42, 'verbose': 0, 'iterations': trial.suggest_int('iterations', 100, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 'depth': trial.suggest_int('depth', 3, 10)}
            model = CatBoostRanker(**params)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, val_idx = next(gss.split(X_global, y_global, groups_global))
        X_train, X_val = X_global.iloc[train_idx], X_global.iloc[val_idx]; y_train, y_val = y_global.iloc[train_idx], y_global.iloc[val_idx]
        train_groups = X_train.groupby('COURSE_ID').size().to_numpy(); val_groups = X_val.groupby('COURSE_ID').size().to_numpy()
        X_train_features = X_train.drop(columns='COURSE_ID'); X_val_features = X_val.drop(columns='COURSE_ID')
        categorical_cols = [f for f in config.CATEGORICAL_FEATURES if f in X_train_features.columns]
        preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)], remainder='passthrough')
        X_train_transformed = preprocessor.fit_transform(X_train_features); X_val_transformed = preprocessor.transform(X_val_features)

        if model_name == 'lgbm':
            model.fit(X_train_transformed, y_train, group=train_groups, eval_set=[(X_val_transformed, y_val)], eval_group=[val_groups], eval_metric='ndcg', callbacks=[lgb.early_stopping(15, verbose=False)])
            score = model.best_score_['valid_0']['ndcg@1']
        elif model_name == 'catboost':
             model.fit(X_train_transformed, y_train, group_id=X_train['COURSE_ID'])
             predictions = model.predict(X_val_transformed)
             score = pd.Series(y_val.to_numpy()).corr(pd.Series(predictions), method='spearman')
        else: # XGBoost
            model.fit(X_train_transformed, y_train, group=train_groups)
            score = model.score(X_val_transformed, y_val)

        return score if not np.isnan(score) else 0
    return objective

def optimize_models(n_trials=None):
    global X_global, y_global, groups_global
    try: df = pd.read_parquet(config.CLEAN_DATA_PATH)
    except FileNotFoundError: print("❌ Erreur : Fichier de données propre non trouvé. Lancez d'abord la commande 'prepare'."); return

    df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
    df['ranking_target'] = df.groupby('COURSE_ID')['PLACE_CHEVAL'].transform(lambda x: x.max() - x + 1).astype(int)
    df = df.sort_values('COURSE_ID')
    groups_global = df['COURSE_ID']
    features = [f for f in config.FEATURES_TO_USE if f in df.columns]
    X_global = df[features + ['COURSE_ID']]; y_global = df['ranking_target']

    models_to_train = {'lgbm': 'LGBMRanker', 'xgb': 'XGBRanker', 'catboost': 'CatBoostRanker'}
    study_paths = {'lgbm': config.STUDY_LGBM_PATH, 'xgb': config.STUDY_XGB_PATH, 'catboost': config.STUDY_CATBOOST_PATH}

    for model_key, model_name in models_to_train.items():
        study_path = study_paths[model_key]
        old_best_value = -np.inf
        try:
            old_study = joblib.load(study_path)
            old_best_value = old_study.best_value
            print(f"\n--- 🚀 Reprise de l'optimisation pour {model_name} (Record actuel: {old_best_value:.4f}) ---")
            study = old_study
        except (FileNotFoundError, EOFError):
            print(f"\n--- 🚀 Nouvelle optimisation pour {model_name} ---")
            study = optuna.create_study(direction='maximize')

        pbar = tqdm(total=n_trials, desc=f"Recherche pour {model_name}", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        def tqdm_callback(study, trial):
            pbar.update(1)
            if study.best_value: pbar.set_postfix_str(f"Meilleur score: {study.best_value:.4f}")
        try:
            print("Lancement de la recherche... Appuyez sur Ctrl+C pour arrêter.")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                study.optimize(get_objective(model_key), n_trials=n_trials, callbacks=[tqdm_callback])
        except KeyboardInterrupt: print(f"\n🛑 Interruption par l'utilisateur pour {model_name}.")
        pbar.close()

        if not study.trials or study.best_trial is None: print("Aucun nouvel essai terminé."); continue
        new_best_value = study.best_value
        print(f"Meilleur score de cette session : {new_best_value:.4f}")
        if new_best_value > old_best_value:
            print("🏆 NOUVEAU RECORD ! Sauvegarde de l'étude améliorée...")
            joblib.dump(study, study_path)
            print(f"✅ Étude sauvegardée dans {study_path}")
        else: print("Pas d'amélioration. L'ancienne étude est conservée.")

def build_model():
    print("\n--- Construction du Modèle Stratégique Final ---")
    try: df = pd.read_parquet(config.CLEAN_DATA_PATH)
    except FileNotFoundError: print("❌ Erreur : Fichier de données propre non trouvé."); return

    df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
    df['ranking_target'] = df.groupby('COURSE_ID')['PLACE_CHEVAL'].transform(lambda x: x.max() - x + 1).astype(int)
    df = df.sort_values('COURSE_ID')
    features = [f for f in config.FEATURES_TO_USE if f in df.columns]
    X = df[features]; y = df['ranking_target']

    estimators = []
    models_to_build = {'lgbm': config.STUDY_LGBM_PATH, 'xgb': config.STUDY_XGB_PATH, 'catboost': config.STUDY_CATBOOST_PATH}
    print("✓ Chargement des meilleurs paramètres trouvés lors de l'optimisation...")
    for model_key, study_path in models_to_build.items():
        try:
            study = joblib.load(study_path)
            best_params = study.best_params
            if model_key == 'lgbm': estimators.append((model_key, lgb.LGBMRanker(**best_params, random_state=42, verbosity=-1)))
            elif model_key == 'xgb': estimators.append((model_key, xgb.XGBRanker(**best_params, random_state=42, eval_metric='ndcg', verbosity=0)))
            elif model_key == 'catboost': estimators.append((model_key, CatBoostRanker(**best_params, random_state=42, verbose=0)))
            print(f"  - Paramètres pour {model_key.upper()} chargés.")
        except (FileNotFoundError, ValueError, EOFError):
             print(f"⚠️ Avertissement : Impossible de charger l'étude pour {model_key.upper()}. Ce modèle ne sera pas inclus.")

    if not estimators: print("❌ Aucun modèle expert optimisé trouvé. Impossible de construire le modèle final."); return

    categorical_cols = [f for f in config.CATEGORICAL_FEATURES if f in X.columns]
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)], remainder='passthrough')

    model_pipelines = {}
    print("\n--- 🚂 Entraînement des modèles experts sur la totalité des données ---")

    group_sizes = df.groupby('COURSE_ID').size().to_numpy()
    group_ids = df['COURSE_ID']

    for name, model in estimators:
        print(f"  ▶️  Entraînement du pipeline final pour : {name}...")
        pipeline = Pipeline([('preprocessor', preprocessor), ('ranker', model)])

        if name == 'catboost':
            pipeline.fit(X, y, ranker__group_id=group_ids)
        else:
            pipeline.fit(X, y, ranker__group=group_sizes)

        model_pipelines[name] = pipeline
        print(f"  ✅ Pipeline {name} finalisé.")

    with open(config.STACKING_MODEL_PATH, 'wb') as f:
        pickle.dump(model_pipelines, f)
    print(f"\n✅ Ensemble de modèles stratégiques sauvegardé dans {config.STACKING_MODEL_PATH}")

