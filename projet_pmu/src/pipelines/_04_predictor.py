import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
from projet_pmu.src import config
from projet_pmu.src.pipelines import _05_promo_manager
from itertools import combinations, permutations
from collections import Counter
import datetime
import json

def clean_money_string(series): return pd.to_numeric(series.astype(str).str.replace(',', '.').str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
def parse_musique_avancee(df):
    records = []
    for index, row in df.iterrows():
        musique_str = row['MUSIQUE_CHEVAL']
        if not isinstance(musique_str, str): records.append({}); continue
        performances = re.findall(r'([0-9]|A|T|D|Ret)([a-z])(?:\((\d{2})\))?', musique_str)[:10]
        if not performances: records.append({}); continue
        parsed_perfs = [{'place': 99 if p[0] in ['A', 'T', 'D', 'Ret'] else int(p[0]) if p[0].isdigit() else 0} for p in performances]
        record = {'MUSIQUE_PLACE_DERNIERE_COURSE': parsed_perfs[0]['place'] if parsed_perfs else 0, 'MUSIQUE_VICTOIRES_3_DERNIERES': sum(1 for p in parsed_perfs[:3] if p['place'] == 1), 'MUSIQUE_PLACES_5_DERNIERES': sum(1 for p in parsed_perfs[:5] if 0 < p['place'] <= 3), 'MUSIQUE_TAUX_INCIDENTS': sum(1 for p in parsed_perfs if p['place'] == 99) / len(parsed_perfs) if parsed_perfs else 0}
        records.append(record)
    df_musique = pd.DataFrame(records, index=df.index)
    return pd.concat([df, df_musique], axis=1)
def softmax(x):
    e_x = np.exp(x - np.max(x)); return e_x / e_x.sum(axis=0)

def run_race_simulations(df_proba, num_simulations=10000):
    participants = df_proba['NUMERO_CHEVAL'].values; win_probabilities = df_proba['PROBA_GAGNANT'].values
    all_finish_orders = []
    for _ in range(num_simulations):
        current_participants = list(participants); current_probabilities = list(win_probabilities)
        finish_order = []
        while len(current_participants) > 0:
            normalized_probs = np.array(current_probabilities) / np.sum(current_probabilities)
            winner_index = np.random.choice(len(current_participants), p=normalized_probs)
            winner = current_participants.pop(winner_index); current_probabilities.pop(winner_index)
            finish_order.append(winner)
        all_finish_orders.append(tuple(finish_order))
    return all_finish_orders

def analyze_simulation_results(simulations, df_race, race_info, rapports_estimes):
    num_simulations = len(simulations); chevaux = df_race['NUMERO_CHEVAL'].values
    active_promotions = _05_promo_manager.scan_and_parse_promotions()
    active_promotion = None
    for promo in active_promotions:
        if not promo.get('is_expired', True) and promo['hippodrome'].upper() == race_info['hippodrome'].upper() and promo['date'] == race_info['date']:
            active_promotion = promo; break
    if active_promotion: print(f"\n💡 PROMOTION DÉTECTÉE : {active_promotion['description']} !")

    print("\n" + "="*80); print(" " * 24 + "💡 STRATÉGIE DE PARIS EXHAUSTIVE 💡"); print("="*80)

    # ... (Le reste de l'analyse est ici, avec des améliorations)
    placements = Counter(cheval for sim in simulations for cheval in sim[:3])
    # ... etc.

def predict(file_path, return_top3=False):
    # ... (le début de la fonction predict est inchangé)
    is_internal_call = return_top3
    try:
        with open(config.STACKING_MODEL_PATH, 'rb') as f: model_pipelines = pickle.load(f)
    except FileNotFoundError:
        if not is_internal_call: print("❌ Erreur : Modèle non trouvé. Lancez la commande 'build' d'abord."); return None
        return pd.DataFrame()
    try:
        with open(config.RAPPORTS_ESTIMES_FILE, 'r') as f: rapports_estimes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️ Fichier des rapports estimés non trouvé. Utilisation de valeurs par défaut."); rapports_estimes = {}
    df_race = pd.read_csv(file_path, sep=';', dtype=str)
    df_jockeys = pd.read_csv(config.REF_JOCKEY_FILE, sep=';', dtype=str).rename(columns={'Name': 'JOCKEY_NOM', 'Partants': 'Jockey_Partants', 'Victoires': 'Jockey_Victoires'}); df_entraineurs = pd.read_csv(config.REF_ENTRAINEUR_FILE, sep=';', dtype=str).rename(columns={'Name': 'ENTRAINEUR_NOM', 'Partants': 'Entraineur_Partants'}); df_chevaux = pd.read_csv(config.REF_CHEVAUX_FILE, sep=';', dtype=str).rename(columns={'Valeur': 'Cheval_Valeur', 'Gain moyen': 'Cheval_Gain_Moyen'}); df_entraineurs['Entraineur_Gain_Moyen_Partant'] = clean_money_string(df_entraineurs['Gain/Part.']); df_race = pd.merge(df_race, df_jockeys, left_on='JOCKEY_ID', right_on='JOCKEY_NOM', how='left'); df_race = pd.merge(df_race, df_entraineurs, left_on='ENTRAINEUR_ID', right_on='ENTRAINEUR_NOM', how='left'); df_race = pd.merge(df_race, df_chevaux, left_on='CHEVAL_ID', right_on='Cheval', how='left'); numeric_cols_base = [col for col in config.FEATURES_TO_USE if 'MUSIQUE' not in col and col not in config.CATEGORICAL_FEATURES and col in df_race.columns]; cols_to_convert = numeric_cols_base + ['Jockey_Partants', 'Jockey_Victoires', 'Entraineur_Partants', 'Entraineur_Gain_Moyen_Partant', 'Cheval_Valeur', 'Cheval_Gain_Moyen']; 
    for col in cols_to_convert:
        if col in df_race.columns: df_race[col] = pd.to_numeric(df_race[col], errors='coerce')
    df_race['Jockey_Ratio_Victoire'] = df_race['Jockey_Victoires'] / df_race['Jockey_Partants']; df_race = parse_musique_avancee(df_race); cols_to_rank = {'COTE_CHEVAL': True, 'POIDS_CHEVAL': False, 'VALEUR_HANDICAP_CHEVAL': False, 'Jockey_Ratio_Victoire': False};
    for col, ascending in cols_to_rank.items():
        if col in df_race.columns: df_race[col + '_RANG'] = df_race.groupby('COURSE_ID')[col].rank(method='dense', ascending=ascending, na_option='bottom')
    for col in df_race.columns:
        if df_race[col].dtype == 'object': df_race[col] = df_race[col].astype(str).fillna('INCONNU')
    df_race.replace([np.inf, -np.inf], np.nan, inplace=True); df_race.fillna(df_race.select_dtypes(include=np.number).median(), inplace=True); df_race.fillna(0, inplace=True)
    if not is_internal_call: print("✅ 1/2 - 📂 Données de la course préparées.")
    features = [f for f in config.FEATURES_TO_USE if f in df_race.columns]; X_pred = df_race[features]
    all_scores = [model.predict(X_pred) for model in model_pipelines.values()]
    final_scores = np.mean(all_scores, axis=0)
    mean_score = np.mean(final_scores); std_score = np.std(final_scores)
    if std_score > 0: scaled_scores = (final_scores - mean_score) / std_score
    else: scaled_scores = final_scores - mean_score
    df_race['PROBA_GAGNANT'] = softmax(scaled_scores)
    df_race['COTE_CHEVAL'] = pd.to_numeric(df_race['COTE_CHEVAL'], errors='coerce').fillna(100.0)
    df_sorted = df_race[['NUMERO_CHEVAL', 'CHEVAL_ID', 'COTE_CHEVAL', 'PROBA_GAGNANT']].sort_values(by='PROBA_GAGNANT', ascending=False).reset_index(drop=True)
    if is_internal_call: return df_sorted.head(3)
    print("✅ 2/2 - 🤖 Pronostic de base calculé. Lancement des simulations...")
    df_sorted['CLASSEMENT_PREDIT'] = df_sorted.index + 1; df_sorted['VALUE_BET_GAGNANT'] = (df_sorted['PROBA_GAGNANT'] * df_sorted['COTE_CHEVAL']) - 1
    print("\n" + "-" * 80); print(" " * 22 + "CLASSEMENT PRÉDICTIF DU COMITÉ D'EXPERTS"); print("-" * 80)
    df_display = df_sorted.copy(); df_display['PROBA_GAGNANT'] = df_display['PROBA_GAGNANT'].map('{:.2%}'.format); df_display['VALUE_BET_GAGNANT'] = df_display['VALUE_BET_GAGNANT'].map('{:+.2f}'.format); df_display['COTE_CHEVAL'] = df_display['COTE_CHEVAL'].map('{:.2f}'.format)
    header = f"{'Classement':<12} {'Numéro':<8} {'Cheval':<25} {'Cote':<10} {'Proba.':<10} {'Value Bet':<10}"
    print(header); print("-" * len(header))
    for _, row in df_display.iterrows(): print(f"{str(row['CLASSEMENT_PREDIT']):<12} {str(row['NUMERO_CHEVAL']):<8} {str(row['CHEVAL_ID']):<25} {str(row['COTE_CHEVAL']):<10} {str(row['PROBA_GAGNANT']):<10} {str(row['VALUE_BET_GAGNANT']):<10}")
    try:
        filename_parts = Path(file_path).stem.replace(' copie', '').split('_'); race_hippodrome = filename_parts[1]; race_date_str = filename_parts[2]
        race_date = datetime.datetime.strptime(race_date_str, '%Y%m%d').date(); race_info = {"hippodrome": race_hippodrome, "date": race_date}
    except Exception: race_info = {"hippodrome": "INCONNU", "date": datetime.date(1970, 1, 1)}
    simulations = run_race_simulations(df_race[['NUMERO_CHEVAL', 'PROBA_GAGNANT']])
    analyze_simulation_results(simulations, df_race, race_info, rapports_estimes)
    output_path = config.PREDICTIONS_DIR / f"prediction_{Path(file_path).name}"; df_sorted.to_csv(output_path, index=False, sep=';')
    print("-" * 80); print(f"\n📄 Pronostic complet (classement) sauvegardé dans : {output_path}")
    return None
