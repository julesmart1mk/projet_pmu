import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from projet_pmu.src import config
from itertools import combinations
from collections import Counter
import datetime
import json

from projet_pmu.src.pipelines._01_preprocessor import MusiqueParser, FeatureCreator, SynergyCreator, DataCleaner

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def calculate_kelly_bet(bankroll, probability, odds):
    if probability <= 0 or odds <= 1: return 0
    net_odds = odds - 1
    if net_odds <= 0: return 0
    kelly_fraction = (probability * net_odds - (1 - probability)) / net_odds
    if kelly_fraction > 0:
        return bankroll * kelly_fraction * 0.25
    return 0

def run_race_simulations(df_proba, num_simulations=5000):
    participants = df_proba['NUMERO_CHEVAL'].values
    win_probabilities = df_proba['PROBA_GAGNANT'].values
    all_finish_orders = []
    for _ in range(num_simulations):
        current_participants = list(participants)
        current_probabilities = list(win_probabilities)
        finish_order = []
        while len(current_participants) > 0:
            normalized_probs = np.array(current_probabilities) / np.sum(current_probabilities)
            winner_index = np.random.choice(len(current_participants), p=normalized_probs)
            winner = current_participants.pop(winner_index)
            current_probabilities.pop(winner_index)
            finish_order.append(winner)
        all_finish_orders.append(tuple(finish_order))
    return all_finish_orders

def analyze_advanced_bets(simulations, rapports_estimes):
    num_simulations = len(simulations)
    if num_simulations == 0: return

    panel_content = ""
    
    top_2 = Counter(tuple(sorted(sim[:2])) for sim in simulations).most_common(1)[0]
    combo, count = top_2
    proba = count / num_simulations
    rapport = rapports_estimes.get("Couple GAGNANT", 15.0)
    value_bet = (proba * rapport) - 1
    style = "[bold green]" if value_bet > 0.1 else "[bold red]"
    panel_content += f"▶️ [bold]Couplé Gagnant Suggéré :[/bold] {combo[0]} - {combo[1]}\n"
    panel_content += f"   - Probabilité : {proba:.2%}\n"
    panel_content += f"   - Value Bet : {style}{value_bet:+.2f}{style}\n\n"

    if len(simulations[0]) >= 3:
        top_3 = Counter(tuple(sorted(sim[:3])) for sim in simulations).most_common(1)[0]
        combo, count = top_3
        proba = count / num_simulations
        rapport = rapports_estimes.get("Tierce DESORDRE", 40.0)
        value_bet = (proba * rapport) - 1
        style = "[bold green]" if value_bet > 0.1 else "[bold red]"
        panel_content += f"▶️ [bold]Tiercé Désordre Suggéré :[/bold] {combo[0]} - {combo[1]} - {combo[2]}\n"
        panel_content += f"   - Probabilité : {proba:.2%}\n"
        panel_content += f"   - Value Bet : {style}{value_bet:+.2f}{style}"

    console.print(Panel(panel_content, title="[bold magenta]📈 Analyse des Paris Avancés 📈[/bold magenta]", expand=False, border_style="magenta"))


def predict(file_path, return_top3=False, bankroll=None):
    try:
        with open(config.STACKING_MODEL_PATH, 'rb') as f: full_system = pickle.load(f)
        preprocessor = full_system['preprocessor']
        base_models = full_system['base_models']
        meta_model = full_system['meta_model']
    except FileNotFoundError:
        console.print("[bold red]❌ Erreur : Modèle non trouvé. Lancez 'build'.[/bold red]"); return
        
    try:
        with open(config.RAPPORTS_ESTIMES_FILE, 'r') as f: rapports_estimes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        rapports_estimes = {}

    df_race = pd.read_csv(file_path, sep=';', dtype=str, low_memory=False)
    
    df_jockeys = pd.read_csv(config.REF_JOCKEY_FILE, sep=';', dtype=str).rename(columns={'Name': 'JOCKEY_NOM', 'Partants': 'Jockey_Partants', 'Victoires': 'Jockey_Victoires'})
    df_entraineurs = pd.read_csv(config.REF_ENTRAINEUR_FILE, sep=';', dtype=str).rename(columns={'Name': 'ENTRAINEUR_NOM', 'Partants': 'Entraineur_Partants'})
    df_entraineurs['Entraineur_Gain_Moyen_Partant'] = pd.to_numeric(df_entraineurs['Gain/Part.'].astype(str).str.replace(',', '.').str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

    X_pred = pd.merge(df_race, df_jockeys, left_on='JOCKEY_ID', right_on='JOCKEY_NOM', how='left')
    X_pred = pd.merge(X_pred, df_entraineurs, left_on='ENTRAINEUR_ID', right_on='ENTRAINEUR_NOM', how='left')

    numeric_cols = ['DISTANCE', 'NOMBRE_PARTANT_COURSE', 'ALLOCATION_TOTAL_COURSE', 'AGE_CHEVAL', 'POIDS_CHEVAL', 'STALLE_CHEVAL', 'VALEUR_HANDICAP_CHEVAL', 'COTE_CHEVAL', 'Jockey_Partants', 'Jockey_Victoires', 'Entraineur_Partants', 'Entraineur_Gain_Moyen_Partant']
    for col in numeric_cols:
        if col in X_pred.columns:
            X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')

    df_historique = pd.read_parquet(config.CLEAN_DATA_PATH)
    synergy_creator = SynergyCreator()
    synergy_creator.fit(df_historique)
    X_pred = synergy_creator.transform(X_pred)
            
    X_pred = MusiqueParser().transform(X_pred)
    X_pred = FeatureCreator().transform(X_pred)
    X_pred = DataCleaner().fit_transform(X_pred)
    
    X_features_only = X_pred.reindex(columns=config.FEATURES_TO_USE, fill_value=0)
    X_processed = preprocessor.transform(X_features_only)
    
    base_predictions = np.zeros((len(X_pred), len(base_models)))
    for i, (name, model) in enumerate(base_models.items()):
        base_predictions[:, i] = model.predict(X_processed)

    final_probas = meta_model.predict_proba(base_predictions)[:, 1]
    
    X_pred['PROBA_GAGNANT'] = final_probas
    X_pred['COTE_CHEVAL'] = pd.to_numeric(X_pred['COTE_CHEVAL'], errors='coerce').fillna(100.0)

    df_sorted = X_pred[['NUMERO_CHEVAL', 'CHEVAL_ID', 'COTE_CHEVAL', 'PROBA_GAGNANT']].sort_values(by='PROBA_GAGNANT', ascending=False).reset_index(drop=True)
    
    if return_top3: return df_sorted.head(3)
    
    df_sorted['VALUE_BET_GAGNANT'] = (df_sorted['PROBA_GAGNANT'] * df_sorted['COTE_CHEVAL']) - 1
    if bankroll:
        df_sorted['MISE_CONSEILLEE'] = [calculate_kelly_bet(bankroll, row['PROBA_GAGNANT'], row['COTE_CHEVAL']) for _, row in df_sorted.iterrows()]

    proba_std = df_sorted['PROBA_GAGNANT'].std()
    confidence_level = "Élevée" if proba_std > 0.08 else "Moyenne" if proba_std > 0.05 else "Faible"
    color = "green" if confidence_level == "Élevée" else "yellow" if confidence_level == "Moyenne" else "red"
    console.print(f"\n[bold {color}]🎯 INDICE DE CONFIANCE : {confidence_level.upper()}[/bold {color}]")
    if confidence_level == "Faible":
        console.print("[italic red]   (Course très ouverte. Paris à haut risque.)[/italic red]")

    table = Table(title="[bold]📊 Classement Prédictif et Mises 📊[/bold]", show_header=True, header_style="bold magenta")
    table.add_column("Cl.", justify="center"); table.add_column("N°", justify="center"); table.add_column("Cheval", style="cyan", no_wrap=True); table.add_column("Cote", justify="right"); table.add_column("Proba.", justify="right"); table.add_column("Value Bet", justify="right")
    if bankroll: table.add_column("Mise Sugg.", justify="right", style="bold yellow")

    for idx, row in df_sorted.iterrows():
        value_style = "green" if row['VALUE_BET_GAGNANT'] > 0.1 else "red"
        row_data = [str(idx + 1), str(row['NUMERO_CHEVAL']), str(row['CHEVAL_ID']), f"{row['COTE_CHEVAL']:.2f}", f"{row['PROBA_GAGNANT']:.2%}", f"[{value_style}]{row['VALUE_BET_GAGNANT']:+.2f}[/{value_style}]"]
        if bankroll: row_data.append(f"€{row['MISE_CONSEILLEE']:.2f}" if row['MISE_CONSEILLEE'] > 0 else "-")
        table.add_row(*row_data)
        
    console.print(table)

    if bankroll and df_sorted['MISE_CONSEILLEE'].sum() > 0:
        console.print(f"💰 [bold]Total à miser (Simple Gagnant) :[/bold] [bold green]€{df_sorted['MISE_CONSEILLEE'].sum():.2f}[/bold green]")
    
    simulations = run_race_simulations(df_sorted[['NUMERO_CHEVAL', 'PROBA_GAGNANT']])
    analyze_advanced_bets(simulations, rapports_estimes)
        
    return None
