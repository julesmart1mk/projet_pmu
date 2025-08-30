import pandas as pd
import numpy as np
import glob
import re
from src import config

def clean_money_string(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.').str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

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

def process_data():
    print("--- Lancement Étape 1 (v12 - Final Synchro) : Prétraitement Définitif ---")
    raw_files = glob.glob(str(config.RAW_COURSES_DIR / "*.csv"))
    if not raw_files:
        print("❌ Erreur : Aucun fichier de course trouvé dans data/courses_brutes/"); return
    df_courses = pd.concat((pd.read_csv(f, sep=';', dtype=str) for f in raw_files), ignore_index=True)
    print(f"✅ {len(df_courses)} lignes chargées depuis {len(raw_files)} fichiers de courses.")

    df_jockeys = pd.read_csv(config.REF_JOCKEY_FILE, sep=';', dtype=str).rename(columns={'Name': 'JOCKEY_NOM', 'Partants': 'Jockey_Partants', 'Victoires': 'Jockey_Victoires'})
    df_entraineurs = pd.read_csv(config.REF_ENTRAINEUR_FILE, sep=';', dtype=str).rename(columns={'Name': 'ENTRAINEUR_NOM', 'Partants': 'Entraineur_Partants'})
    df_chevaux = pd.read_csv(config.REF_CHEVAUX_FILE, sep=';', dtype=str).rename(columns={'Valeur': 'Cheval_Valeur', 'Gain moyen': 'Cheval_Gain_Moyen'})
    df_entraineurs['Entraineur_Gain_Moyen_Partant'] = clean_money_string(df_entraineurs['Gain/Part.'])
    df_courses = pd.merge(df_courses, df_jockeys[['JOCKEY_NOM', 'Jockey_Partants', 'Jockey_Victoires']], left_on='JOCKEY_ID', right_on='JOCKEY_NOM', how='left')
    df_courses = pd.merge(df_courses, df_entraineurs[['ENTRAINEUR_NOM', 'Entraineur_Partants', 'Entraineur_Gain_Moyen_Partant']], left_on='ENTRAINEUR_ID', right_on='ENTRAINEUR_NOM', how='left')
    df_courses = pd.merge(df_courses, df_chevaux[['Cheval', 'Cheval_Valeur', 'Cheval_Gain_Moyen']], left_on='CHEVAL_ID', right_on='Cheval', how='left')
    print("✅ Fusion des données terminée.")
    
    df_courses[config.TARGET_COLUMN] = pd.to_numeric(df_courses[config.TARGET_COLUMN], errors='coerce')
    numeric_cols = [col for col in config.FEATURES_TO_USE if 'MUSIQUE' not in col and col not in config.CATEGORICAL_FEATURES]
    numeric_cols += ['Jockey_Partants', 'Jockey_Victoires', 'Entraineur_Partants', 'Entraineur_Gain_Moyen_Partant', 'Cheval_Valeur', 'Cheval_Gain_Moyen']
    for col in numeric_cols:
        if col in df_courses.columns:
            df_courses[col] = pd.to_numeric(df_courses[col], errors='coerce')

    df_courses['Jockey_Ratio_Victoire'] = df_courses['Jockey_Victoires'] / df_courses['Jockey_Partants']
    df_courses = parse_musique_avancee(df_courses)
    
    cols_to_rank = {'COTE_CHEVAL': True, 'POIDS_CHEVAL': False, 'VALEUR_HANDICAP_CHEVAL': False, 'Jockey_Ratio_Victoire': False}
    for col, ascending in cols_to_rank.items():
        if col in df_courses.columns:
            df_courses[col + '_RANG'] = df_courses.groupby('COURSE_ID')[col].rank(method='dense', ascending=ascending, na_option='bottom')
    print("✅ Création des features terminée.")
    
    text_cols = ['CHEVAL_ID', 'JOCKEY_ID', 'ENTRAINEUR_ID', 'JOCKEY_NOM', 'ENTRAINEUR_NOM', 'Cheval']
    for col in text_cols:
        if col in df_courses.columns:
            df_courses[col] = df_courses[col].astype(str).fillna('INCONNU')
    for col in config.CATEGORICAL_FEATURES:
        if col in df_courses.columns:
            df_courses[col] = df_courses[col].fillna("MANQUANT")
    for col in df_courses.select_dtypes(include=np.number).columns:
        df_courses[col] = df_courses[col].fillna(df_courses[col].median())
    df_courses = df_courses.fillna(0)
    print("✅ Nettoyage des données terminé.")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_courses.to_parquet(config.CLEAN_DATA_PATH, index=False)
    print(f"✅ Base de données propre et ultra-enrichie sauvegardée.")
