import pandas as pd
import numpy as np
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
    train_df[config.TARGET_COLUMN] = pd.to_numeric(train_df[config.TARGET_COLUMN], errors='coerce')
    train_df.dropna(subset=[config.TARGET_COLUMN], inplace=True)

    train_df = train_df.sort_values('COURSE_ID')
    train_df['ranking_target'] = train_df.groupby('COURSE_ID')['PLACE_CHEVAL'].transform(lambda x: x.max() - x + 1)
    TARGET = 'ranking_target'
    features = [f for f in config.FEATURES_TO_USE if f in train_df.columns]
    X_train = train_df[features]
    y_train = train_df[TARGET]
    train_groups = train_df.groupby('COURSE_ID').size().to_numpy()

    lgbm = lgb.LGBMRanker(random_state=42, verbosity=-1)
    xgbr = xgb.XGBRanker(random_state=42, eval_metric='ndcg', verbosity=0)
    catr = CatBoostRanker(random_state=42, verbose=0)
    
    categorical_cols = [f for f in config.CATEGORICAL_FEATURES if f in X_train.columns]
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)], remainder='passthrough')

    lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', lgbm)])
    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', xgbr)])
    cat_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', catr)])
    
    lgbm_pipeline.fit(X_train, y_train, ranker__group=train_groups)
    xgb_pipeline.fit(X_train, y_train, ranker__group=train_groups)
    cat_pipeline.fit(X_train, y_train, ranker__group_id=train_df['COURSE_ID'].to_numpy())
    
    return lgbm_pipeline, xgb_pipeline, cat_pipeline

def run_backtest():
    COMMISSION = 0.20
    print(f"--- 🏁 Lancement du Backtest Chronologique (avec commission de {COMMISSION:.0%}) ---")
    
    try:
        df_full = pd.read_parquet(config.CLEAN_DATA_PATH)
        # --- CORRECTION : Le fichier Parquet contient déjà des dates au bon format ---
        # On s'assure juste que la colonne est bien de type datetime
        if 'DATE' not in df_full.columns:
            raise KeyError("La colonne 'DATE' est manquante.")
            
        df_full['DATE'] = pd.to_datetime(df_full['DATE'], errors='coerce')
        df_full.dropna(subset=['DATE', config.TARGET_COLUMN], inplace=True)

    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Erreur : {e}. Lancez 'prepare' pour générer le fichier de données.")
        return

    sorted_dates = sorted(df_full['DATE'].unique())
    
    if len(sorted_dates) < 2:
        print(f"Pas assez de données pour un backtest. Jours uniques valides trouvés : {len(sorted_dates)}.")
        return
        
    print(f"🔎 {len(sorted_dates) - 1} jours de courses vont être simulés...")
    all_results = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for i in tqdm(range(1, len(sorted_dates)), desc="Simulation chronologique"):
            current_date = sorted_dates[i]
            train_df = df_full[df_full['DATE'] < current_date].copy()
            test_df = df_full[df_full['DATE'] == current_date].copy()

            if train_df.empty or test_df.empty: continue

            lgbm_pipe, xgb_pipe, cat_pipe = train_temporary_model(train_df)
            
            features = [f for f in config.FEATURES_TO_USE if f in test_df.columns]
            X_test = test_df[features]
            
            lgbm_scores = lgbm_pipe.predict(X_test); xgb_scores = xgb_pipe.predict(X_test); cat_scores = cat_pipe.predict(X_test)
            final_scores = (lgbm_scores + xgb_scores + cat_scores) / 3.0
            
            df_pred = test_df[['COURSE_ID', 'NUMERO_CHEVAL', 'COTE_CHEVAL', 'PLACE_CHEVAL']].copy()
            df_pred['SCORE'] = final_scores
            
            for course_id in df_pred['COURSE_ID'].unique():
                df_course_pred = df_pred[df_pred['COURSE_ID'] == course_id].sort_values(by='SCORE', ascending=False)
                
                if df_course_pred.empty: continue

                pred_gagnant = df_course_pred.iloc[0]
                is_winner = (pd.to_numeric(pred_gagnant['PLACE_CHEVAL'], errors='coerce') == 1)
                
                mise = 1.0
                cote_gagnant = pd.to_numeric(pred_gagnant['COTE_CHEVAL'], errors='coerce')
                gain_brut = cote_gagnant if is_winner and not pd.isna(cote_gagnant) else 0.0
                gain_net = gain_brut * (1 - COMMISSION)
                benefice = gain_net - mise

                all_results.append({
                    'Mise': mise,
                    'Gain_Net_Apres_Commission': gain_net,
                    'Benefice': benefice,
                    'Gagnant_Trouve': 1 if is_winner else 0
                })

    if not all_results:
        print("Le backtest n'a pu générer aucun résultat."); return

    df_report = pd.DataFrame(all_results)
    total_mises = df_report['Mise'].sum()
    total_benefice = df_report['Benefice'].sum()
    roi = (total_benefice / total_mises) * 100 if total_mises > 0 else 0
    reussite = df_report['Gagnant_Trouve'].mean() * 100
    
    print("\n\n--- 📊 RAPPORT DE BACKTESTING CHRONOLOGIQUE FINAL 📊 ---")
    print(f"Stratégie: 1€ sur le 'Simple Gagnant' prédit pour chaque course.")
    print("---")
    print(f"Courses simulées           : {len(df_report)}")
    print(f"Taux de réussite           : {reussite:.2f} %")
    print("---")
    print("Simulation de Rentabilité (Commission incluse)")
    print(f"Coût total des mises       : {total_mises:.2f} €")
    print(f"Gain Net Total             : {df_report['Gain_Net_Apres_Commission'].sum():.2f} €")
    print(f"Bénéfice / Perte           : {total_benefice:.2f} €")
    print(f"Retour sur Investissement (ROI): {roi:.2f} %")
    print("-------------------------------------------------")

if __name__ == '__main__':
    run_backtest()
