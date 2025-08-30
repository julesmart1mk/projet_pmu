import pandas as pd
import numpy as np
import pickle
import warnings
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker
from projet_pmu.src import config

def train_temporary_model(train_df):
    """Entraîne un comité d'experts sur un jeu de données spécifique."""
    
    # Prépare les données pour l'entraînement
    train_df = train_df.sort_values('COURSE_ID')
    train_df['ranking_target'] = train_df.groupby('COURSE_ID')['PLACE_CHEVAL'].transform(lambda x: x.max() - x + 1)
    TARGET = 'ranking_target'
    features = [f for f in config.FEATURES_TO_USE if f in train_df.columns]
    X_train = train_df[features]
    y_train = train_df[TARGET]
    train_groups = train_df.groupby('COURSE_ID').size().to_numpy()

    # Création des 3 modèles experts avec des paramètres par défaut robustes
    lgbm = lgb.LGBMRanker(random_state=42, verbosity=-1)
    xgbr = xgb.XGBRanker(random_state=42, eval_metric='ndcg', verbosity=0)
    catr = CatBoostRanker(random_state=42, verbose=0)
    
    # Création du pré-processeur
    categorical_cols = [f for f in config.CATEGORICAL_FEATURES if f in X_train.columns]
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)], remainder='passthrough')

    # Création des pipelines pour chaque modèle
    lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', lgbm)])
    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', xgbr)])
    cat_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', catr)])
    
    # Entraînement des 3 modèles
    lgbm_pipeline.fit(X_train, y_train, ranker__group=train_groups)
    xgb_pipeline.fit(X_train, y_train, ranker__group=train_groups)
    cat_pipeline.fit(X_train, y_train, ranker__group_id=train_df['COURSE_ID'].to_numpy())
    
    return lgbm_pipeline, xgb_pipeline, cat_pipeline

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def run_backtest():
    print("--- 🏁 Lancement du Backtest Chronologique (Épreuve du Feu) ---")
    
    try:
        df_full = pd.read_parquet(config.CLEAN_DATA_PATH)
        df_full.dropna(subset=[config.TARGET_COLUMN], inplace=True)
        df_full['DATE'] = pd.to_numeric(df_full['DATE'])
    except FileNotFoundError:
        print("❌ Erreur : Le fichier de données propres n'existe pas. Lancez d'abord un entraînement.")
        return

    # On trie les courses par date
    sorted_dates = sorted(df_full['DATE'].unique())
    print(f"🔎 {len(sorted_dates)} jours de courses uniques vont être simulés...")

    all_results = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # On itère sur chaque jour, en s'arrêtant à l'avant-dernier
        for i in tqdm(range(1, len(sorted_dates)), desc="Simulation chronologique"):
            current_date = sorted_dates[i]
            
            # Données d'entraînement = tout ce qui s'est passé AVANT la date actuelle
            train_df = df_full[df_full['DATE'] < current_date].copy()
            # Données de test = uniquement les courses de la date actuelle
            test_df = df_full[df_full['DATE'] == current_date].copy()

            if train_df.empty or test_df.empty:
                continue

            # On entraîne un nouveau comité d'experts sur les données du passé
            lgbm_pipe, xgb_pipe, cat_pipe = train_temporary_model(train_df)
            
            # On prédit sur les courses du jour
            features = [f for f in config.FEATURES_TO_USE if f in test_df.columns]
            X_test = test_df[features]
            
            lgbm_scores = lgbm_pipe.predict(X_test); xgb_scores = xgb_pipe.predict(X_test); cat_scores = cat_pipe.predict(X_test)
            final_scores = (lgbm_scores + xgb_scores + cat_scores) / 3.0
            
            df_pred = test_df[['COURSE_ID', 'NUMERO_CHEVAL', 'COTE_CHEVAL', 'PLACE_CHEVAL']].copy()
            df_pred['SCORE'] = final_scores
            
            # On analyse le résultat pour chaque course de la journée
            for course_id in df_pred['COURSE_ID'].unique():
                df_course_pred = df_pred[df_pred['COURSE_ID'] == course_id].sort_values(by='SCORE', ascending=False)
                
                pred_gagnant = df_course_pred.iloc[0]['NUMERO_CHEVAL']
                vrai_gagnant = df_course_pred.loc[df_course_pred['PLACE_CHEVAL'].idxmin()]['NUMERO_CHEVAL']
                
                cote_gagnant_trouve = pd.to_numeric(df_course_pred.iloc[0]['COTE_CHEVAL'], errors='coerce') if pred_gagnant == vrai_gagnant else 0

                all_results.append({
                    'Gagnant_Trouve': 1 if pred_gagnant == vrai_gagnant else 0,
                    'Rapport_Gagnant': cote_gagnant_trouve
                })

    if not all_results:
        print("Pas assez de données pour un backtest chronologique (il faut au moins 2 jours de courses)."); return

    # --- RAPPORT FINAL ---
    df_report = pd.DataFrame(all_results)
    total_courses = len(df_report)
    total_gagnants_trouves = df_report['Gagnant_Trouve'].sum()
    reussite_gagnant = total_gagnants_trouves / total_courses
    cout_total = total_courses * 1
    gain_total = df_report['Rapport_Gagnant'].sum()
    roi = ((gain_total - cout_total) / cout_total) * 100
    
    print("\n\n--- 📊 RAPPORT DE BACKTESTING CHRONOLOGIQUE FINAL 📊 ---")
    print(f"Courses simulées           : {total_courses}")
    print(f"Gagnants correctement prédits : {total_gagnants_trouves} ({reussite_gagnant:.2%})")
    print("---")
    print("Simulation de Rentabilité (stratégie: 1€ sur le 'Simple Gagnant' prédit)")
    print(f"Coût total des mises       : {cout_total:.2f} €")
    print(f"Gains totaux               : {gain_total:.2f} €")
    print(f"Bénéfice / Perte           : {gain_total - cout_total:.2f} €")
    print(f"Retour sur Investissement (ROI): {roi:.2f} %")
    print("-------------------------------------------------")

if __name__ == '__main__':
    run_backtest()
