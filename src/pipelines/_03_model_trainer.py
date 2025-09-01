import pandas as pd
import numpy as np
import pickle
import optuna
import warnings
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker
from projet_pmu.src import config
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
            score = model.best_score_['valid_0']['ndcg@1'] if model.best_score_ else 0
        elif model_name == 'catboost':
             model.fit(X_train_transformed, y_train, group_id=X_train['COURSE_ID'])
             predictions = model.predict(X_val_transformed)
             score = pd.Series(y_val.to_numpy()).corr(pd.Series(predictions), method='spearman')
        else: # XGBoost
            model.fit(X_train_transformed, y_train, group=train_groups)
            predictions = model.predict(X_val_transformed)
            score = pd.Series(y_val.to_numpy()).corr(pd.Series(predictions), method='spearman')

        return score if not np.isnan(score) else 0
    return objective

def optimize_models(n_trials=None):
    global X_global, y_global, groups_global
    try: df = pd.read_parquet(config.CLEAN_DATA_PATH)
    except FileNotFoundError: print("❌ Erreur : Fichier de données propre non trouvé. Lancez 'prepare'."); return

    df[config.TARGET_COLUMN] = pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce')
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
            # --- CORRECTION : Le pruner est défini ici, à la création de l'étude ---
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
            study = optuna.create_study(direction='maximize', pruner=pruner)
            
        pbar = tqdm(total=n_trials, desc=f"Recherche pour {model_name}")

        def tqdm_callback(study, trial):
            pbar.update(1)
            if study.best_value: pbar.set_postfix_str(f"Meilleur score: {study.best_value:.4f}")

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                # On enlève l'argument 'pruner' de l'appel à optimize
                study.optimize(
                    get_objective(model_key), 
                    n_trials=n_trials, 
                    callbacks=[tqdm_callback]
                )
        except KeyboardInterrupt: print(f"\n🛑 Interruption par l'utilisateur.")
        except optuna.exceptions.OptunaError as e: print(f"\n⏹️ Arrêt anticipé ou erreur Optuna: {e}")
        pbar.close()

        if not study.trials or study.best_trial is None: print("Aucun nouvel essai terminé."); continue
        
        new_best_value = study.best_value
        print(f"Meilleur score de cette session : {new_best_value:.4f}")
        if new_best_value > old_best_value:
            print("🏆 NOUVEAU RECORD ! Sauvegarde de l'étude améliorée...")
            joblib.dump(study, study_path)
        else: print("Pas d'amélioration.")

def build_model():
    print("\n--- Construction du Modèle Stratégique Final (avec Stacking) ---")
    try: df = pd.read_parquet(config.CLEAN_DATA_PATH)
    except FileNotFoundError: print("❌ Erreur : Fichier de données propre non trouvé."); return

    df[config.TARGET_COLUMN] = pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[config.TARGET_COLUMN], inplace=True)

    df['ranking_target'] = df.groupby('COURSE_ID')['PLACE_CHEVAL'].transform(lambda x: x.max() - x + 1).astype(int)
    df = df.sort_values('COURSE_ID')
    features = [f for f in config.FEATURES_TO_USE if f in df.columns]
    X = df[features + ['COURSE_ID']]; y = df['ranking_target']

    base_models = {}
    models_to_build = {'lgbm': config.STUDY_LGBM_PATH, 'xgb': config.STUDY_XGB_PATH, 'catboost': config.STUDY_CATBOOST_PATH}
    for model_key, study_path in models_to_build.items():
        try:
            study = joblib.load(study_path)
            best_params = study.best_params
            if model_key == 'lgbm': base_models[model_key] = lgb.LGBMRanker(**best_params, random_state=42, verbosity=-1)
            elif model_key == 'xgb': base_models[model_key] = xgb.XGBRanker(**best_params, random_state=42, eval_metric='ndcg', verbosity=0)
            elif model_key == 'catboost': base_models[model_key] = CatBoostRanker(**best_params, random_state=42, verbose=0)
        except Exception as e:
             print(f"⚠️ Avertissement : échec du chargement de {model_key.upper()}: {e}")

    if not base_models: print("❌ Aucun modèle de base trouvé."); return

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros((len(X), len(base_models)))
    
    X_features_only = X.drop(columns=['COURSE_ID'])
    categorical_features = [f for f in config.CATEGORICAL_FEATURES if f in X_features_only.columns]
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)], remainder='passthrough')
    X_processed = preprocessor.fit_transform(X_features_only)

    print("✓ Préparation du méta-modèle (cela peut prendre un moment)...")
    for i, (model_name, model) in enumerate(base_models.items()):
        print(f"  - Traitement des prédictions pour {model_name}...")
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
            y_train = y.iloc[train_idx]
            
            fit_params = {}
            if model_name in ['lgbm', 'xgb']:
                fit_params['group'] = X.iloc[train_idx].groupby('COURSE_ID').size().to_numpy()
            elif model_name == 'catboost':
                fit_params['group_id'] = X.iloc[train_idx]['COURSE_ID']
            
            model.fit(X_train, y_train, **fit_params)
            meta_features[val_idx, i] = model.predict(X_val)

    print("✓ Entraînement du superviseur (méta-modèle)...")
    meta_model = LogisticRegression(random_state=42)
    y_binary = (df['PLACE_CHEVAL'] == 1).astype(int)
    meta_model.fit(meta_features, y_binary)

    print("✓ Ré-entraînement des modèles de base sur 100% des données...")
    final_base_models = {}
    for name, model in base_models.items():
        fit_params = {}
        if name in ['lgbm', 'xgb']:
            fit_params['group'] = X.groupby('COURSE_ID').size().to_numpy()
        elif name == 'catboost':
            fit_params['group_id'] = X['COURSE_ID']
        model.fit(X_processed, y, **fit_params)
        final_base_models[name] = model

    full_system = {
        'preprocessor': preprocessor,
        'base_models': final_base_models,
        'meta_model': meta_model
    }
    with open(config.STACKING_MODEL_PATH, 'wb') as f:
        pickle.dump(full_system, f)
    print(f"\n✅ Système de Stacking complet sauvegardé dans {config.STACKING_MODEL_PATH}")
