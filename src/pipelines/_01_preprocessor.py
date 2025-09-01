import pandas as pd
import numpy as np
import glob
import re
from sklearn.base import BaseEstimator, TransformerMixin
from projet_pmu.src import config

class MusiqueParser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        records = []
        for index, row in df.iterrows():
            musique_str = row['MUSIQUE_CHEVAL']
            if not isinstance(musique_str, str): records.append({}); continue
            performances = re.findall(r'([0-9]|A|T|D|Ret)([a-z])(?:\((\d{2})\))?', musique_str)[:10]
            if not performances: records.append({}); continue
            parsed_perfs = [{'place': 99 if p[0] in ['A', 'T', 'D', 'Ret'] else int(p[0]) if p[0].isdigit() else 0} for p in performances]
            record = {
                'MUSIQUE_PLACE_DERNIERE_COURSE': parsed_perfs[0]['place'] if parsed_perfs else 0,
                'MUSIQUE_VICTOIRES_3_DERNIERES': sum(1 for p in parsed_perfs[:3] if p['place'] == 1),
                'MUSIQUE_PLACES_5_DERNIERES': sum(1 for p in parsed_perfs[:5] if 1 <= p['place'] <= 3),
                'MUSIQUE_TAUX_INCIDENTS': sum(1 for p in parsed_perfs if p['place'] == 99) / len(parsed_perfs) if parsed_perfs else 0
            }
            records.append(record)
        df_musique = pd.DataFrame(records, index=df.index)
        return pd.concat([df, df_musique], axis=1)

class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df['Jockey_Ratio_Victoire'] = df['Jockey_Victoires'] / df['Jockey_Partants']
        cols_to_rank = {
            'COTE_CHEVAL': ('COTE_RANG', True),
            'POIDS_CHEVAL': ('POIDS_RANG', False),
            'VALEUR_HANDICAP_CHEVAL': ('VALEUR_HANDICAP_RANG', False),
            'Jockey_Ratio_Victoire': ('JOCKEY_RATIO_RANG', False)
        }
        for col, (new_name, ascending) in cols_to_rank.items():
            if col in df.columns:
                df[new_name] = df.groupby('COURSE_ID')[col].rank(method='dense', ascending=ascending, na_option='bottom')
        return df

class SynergyCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.synergy_stats = {}

    def fit(self, X, y=None):
        df = X.copy()
        df['gagnant'] = (pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce') == 1).astype(int)
        
        jc_stats = df.groupby(['JOCKEY_ID', 'CHEVAL_ID'])['gagnant'].mean().rename('SYNERGIE_JOCKEY_CHEVAL_RATIO')
        self.synergy_stats['jockey_cheval'] = jc_stats

        ec_stats = df.groupby(['ENTRAINEUR_ID', 'CHEVAL_ID'])['gagnant'].mean().rename('SYNERGIE_ENTRAINEUR_CHEVAL_RATIO')
        self.synergy_stats['entraineur_cheval'] = ec_stats
        
        return self

    def transform(self, X):
        df = X.copy()
        df = pd.merge(df, self.synergy_stats['jockey_cheval'], on=['JOCKEY_ID', 'CHEVAL_ID'], how='left')
        df = pd.merge(df, self.synergy_stats['entraineur_cheval'], on=['ENTRAINEUR_ID', 'CHEVAL_ID'], how='left')
        return df

class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_values_ = X.select_dtypes(include=np.number).median()
        return self
    def transform(self, X):
        df = X.copy()
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str).fillna('INCONNU')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(self.median_values_, inplace=True)
        df.fillna(0, inplace=True)
        return df

def process_data():
    print("--- Étape 1 : Prétraitement des Données ---")
    raw_files = glob.glob(str(config.RAW_COURSES_DIR / "*.csv"))
    if not raw_files: print("❌ Erreur : Aucun fichier de course trouvé."); return

    df_courses = pd.concat((pd.read_csv(f, sep=';', dtype=str, low_memory=False) for f in raw_files), ignore_index=True)
    
    df_jockeys = pd.read_csv(config.REF_JOCKEY_FILE, sep=';', dtype=str).rename(columns={'Name': 'JOCKEY_NOM', 'Partants': 'Jockey_Partants', 'Victoires': 'Jockey_Victoires'})
    df_entraineurs = pd.read_csv(config.REF_ENTRAINEUR_FILE, sep=';', dtype=str).rename(columns={'Name': 'ENTRAINEUR_NOM', 'Partants': 'Entraineur_Partants'})
    
    df_courses = pd.merge(df_courses, df_jockeys, left_on='JOCKEY_ID', right_on='JOCKEY_NOM', how='left')
    df_courses = pd.merge(df_courses, df_entraineurs, left_on='ENTRAINEUR_ID', right_on='ENTRAINEUR_NOM', how='left')
    
    numeric_cols = [col for col in config.FEATURES_TO_USE if col not in config.CATEGORICAL_FEATURES and 'MUSIQUE' not in col and 'SYNERGIE' not in col]
    for col in numeric_cols:
        if col in df_courses.columns:
            df_courses[col] = pd.to_numeric(df_courses[col], errors='coerce')

    synergy_creator = SynergyCreator()
    synergy_creator.fit(df_courses)
    df_courses = synergy_creator.transform(df_courses)

    df_courses = MusiqueParser().transform(df_courses)
    df_courses = FeatureCreator().transform(df_courses)
    df_courses = DataCleaner().fit_transform(df_courses)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_courses.to_parquet(config.CLEAN_DATA_PATH, index=False)
    print(f"✅ Base de données propre sauvegardée.")
