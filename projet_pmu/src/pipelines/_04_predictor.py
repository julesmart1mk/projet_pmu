import pandas as pd
import numpy as np
import pickle
import re
from itertools import combinations
from src import config

# Fonctions identiques à _01_preprocessor.py
def clean_money_string(series): return pd.to_numeric(series.astype(str).str.replace(',', '.').str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
def parse_musique_avancee(df):
    records = []
    current_year = 25
    for index, row in df.iterrows():
        musique_str = row['MUSIQUE_CHEVAL']
        discipline_course = row['DISCIPLINE']
        if not isinstance(musique_str, str): records.append({}); continue
        performances = re.findall(r'([0-9]|A|T|D|Ret)([a-z])(?:\((\d{2})\))?', musique_str)
        performances = performances[:10]
        if not performances: records.append({}); continue
        parsed_perfs = []
        for p in performances:
            place_str, _, year_str = p
            place = 99 if place_str in ['A', 'T', 'D', 'Ret'] else int(place_str) if place_str.isdigit() else 0
            parsed_perfs.append({'place': place})
        record = {
            'MUSIQUE_PLACE_DERNIERE_COURSE': parsed_perfs[0]['place'] if parsed_perfs else 0,
            'MUSIQUE_VICTOIRES_3_DERNIERES': sum(1 for p in parsed_perfs[:3] if p['place'] == 1),
            'MUSIQUE_PLACES_5_DERNIERES': sum(1 for p in parsed_perfs[:5] if 0 < p['place'] <= 3),
            'MUSIQUE_TAUX_INCIDENTS': sum(1 for p in parsed_perfs if p['place'] == 99) / len(parsed_perfs) if parsed_perfs else 0
        }
        records.append(record)
    df_musique = pd.DataFrame(records, index=df.index)
    return pd.concat([df, df_musique], axis=1)
def softmax(x):
    e_x = np.exp(x - np.max(x)); return e_x / e_x.sum(axis=0)

def predict(file_path, return_winner_only=False, return_dfs=False, return_top3=False):
    if not any([return_winner_only, return_dfs, return_top3]):
        print(f"--- Lancement Étape 4 (v15 - Final) : Stratège de Paris Ultime ---")

    try:
        with open(config.LGBM_RANKER_PATH, 'rb') as f: lgbm_pipeline = pickle.load(f)
        with open(config.XGB_RANKER_PATH, 'rb') as f: xgb_pipeline = pickle.load(f)
        with open(config.CATBOOST_RANKER_PATH, 'rb') as f: cat_pipeline = pickle.load(f)
    except FileNotFoundError:
        print("❌ Erreur : Un ou plusieurs modèles experts sont manquants. Lancez d'abord l'entraînement."); return None
    
    df_race = pd.read_csv(file_path, sep=';', dtype=str)
    if not any([return_winner_only, return_dfs, return_top3]): print("✅ Données de la course à prédire chargées.")

    df_jockeys = pd.read_csv(config.REF_JOCKEY_FILE, sep=';', dtype=str).rename(columns={'Name': 'JOCKEY_NOM', 'Partants': 'Jockey_Partants', 'Victoires': 'Jockey_Victoires'})
    df_entraineurs = pd.read_csv(config.REF_ENTRAINEUR_FILE, sep=';', dtype=str).rename(columns={'Name': 'ENTRAINEUR_NOM', 'Partants': 'Entraineur_Partants'})
    df_chevaux = pd.read_csv(config.REF_CHEVAUX_FILE, sep=';', dtype=str).rename(columns={'Valeur': 'Cheval_Valeur', 'Gain moyen': 'Cheval_Gain_Moyen'})
    df_entraineurs['Entraineur_Gain_Moyen_Partant'] = clean_money_string(df_entraineurs['Gain/Part.'])
    df_race = pd.merge(df_race, df_jockeys[['JOCKEY_NOM', 'Jockey_Partants', 'Jockey_Victoires']], left_on='JOCKEY_ID', right_on='JOCKEY_NOM', how='left')
    df_race = pd.merge(df_race, df_entraineurs[['ENTRAINEUR_NOM', 'Entraineur_Partants', 'Entraineur_Gain_Moyen_Partant']], left_on='ENTRAINEUR_ID', right_on='ENTRAINEUR_NOM', how='left')
    df_race = pd.merge(df_race, df_chevaux[['Cheval', 'Cheval_Valeur', 'Cheval_Gain_Moyen']], left_on='CHEVAL_ID', right_on='Cheval', how='left')
    
    df_race[config.TARGET_COLUMN] = pd.to_numeric(df_race[config.TARGET_COLUMN], errors='coerce')
    numeric_cols = [col for col in config.FEATURES_TO_USE if 'MUSIQUE' not in col and col not in config.CATEGORICAL_FEATURES]
    numeric_cols += ['Jockey_Partants', 'Jockey_Victoires', 'Entraineur_Partants', 'Entraineur_Gain_Moyen_Partant', 'Cheval_Valeur', 'Cheval_Gain_Moyen']
    for col in numeric_cols:
        if col in df_race.columns: df_race[col] = pd.to_numeric(df_race[col], errors='coerce')
    df_race['Jockey_Ratio_Victoire'] = df_race['Jockey_Victoires'] / df_race['Jockey_Partants']
    df_race = parse_musique_avancee(df_race)
    cols_to_rank = {'COTE_CHEVAL': True, 'POIDS_CHEVAL': False, 'VALEUR_HANDICAP_CHEVAL': False, 'Jockey_Ratio_Victoire': False}
    for col, ascending in cols_to_rank.items():
        if col in df_race.columns: df_race[col + '_RANG'] = df_race.groupby('COURSE_ID')[col].rank(method='dense', ascending=ascending, na_option='bottom')
    text_cols = ['CHEVAL_ID', 'JOCKEY_ID', 'ENTRAINEUR_ID', 'JOCKEY_NOM', 'ENTRAINEUR_NOM', 'Cheval']
    for col in text_cols:
        if col in df_race.columns: df_race[col] = df_race[col].astype(str).fillna('INCONNU')
    for col in config.CATEGORICAL_FEATURES:
        if col in df_race.columns: df_race[col] = df_race[col].fillna("MANQUANT")
    for col in df_race.select_dtypes(include=np.number).columns:
        if col != config.TARGET_COLUMN: df_race[col] = df_race[col].fillna(df_race[col].median())
    df_race = df_race.fillna(0)
    if not any([return_winner_only, return_dfs, return_top3]): print("✅ Données de la course enrichies et nettoyées.")

    features = [f for f in config.FEATURES_TO_USE if f in df_race.columns]; X_pred = df_race[features]
    
    lgbm_scores = lgbm_pipeline.predict(X_pred); xgb_scores = xgb_pipeline.predict(X_pred); cat_scores = cat_pipeline.predict(X_pred)
    final_scores = (lgbm_scores + xgb_scores + cat_scores) / 3.0
    
    proba_gagnant = softmax(final_scores)
    df_result = df_race[['NUMERO_CHEVAL', 'CHEVAL_ID', 'COTE_CHEVAL']].copy(); df_result['PROBA_GAGNANT'] = proba_gagnant; df_result['COTE_CHEVAL'] = pd.to_numeric(df_result['COTE_CHEVAL'], errors='coerce').fillna(100.0)
    df_sorted = df_result.sort_values(by='PROBA_GAGNANT', ascending=False).reset_index(drop=True)
    
    if return_winner_only: return df_sorted.iloc[0]['NUMERO_CHEVAL']
    if return_top3: return df_sorted.head(3)
    df_sorted['CLASSEMENT_PREDIT'] = df_sorted.index + 1; df_sorted['VALUE_BET_GAGNANT'] = (df_sorted['PROBA_GAGNANT'] * df_sorted['COTE_CHEVAL']) - 1
    # Le reste du script est pour l'affichage et n'a pas changé.
    # ...
    print("\n--- CLASSEMENT FINAL DU COMITÉ D'EXPERTS ---")
    final_cols = ['CLASSEMENT_PREDIT', 'NUMERO_CHEVAL', 'CHEVAL_ID', 'COTE_CHEVAL', 'PROBA_GAGNANT', 'VALUE_BET_GAGNANT']; print(df_sorted[final_cols].round(4).to_string(index=False))
    output_path = config.PREDICTIONS_DIR / f"prediction_{file_path.name}"; df_sorted.to_csv(output_path, index=False, sep=';'); print(f"\n✅ Pronostic complet sauvegardé dans {output_path}")
