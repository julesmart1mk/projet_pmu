import os
import sys
import argparse
import glob
from pathlib import Path
from projet_pmu.src.pipelines import _01_preprocessor, _03_model_trainer, _04_predictor
from projet_pmu.src import backtest
from projet_pmu.src import config
from rich.console import Console
from rich.panel import Panel

console = Console()

def print_header(title):
    """Affiche un titre stylisé avec rich."""
    console.print(Panel(f"[bold cyan]🏇 {title} 🏇[/bold cyan]", expand=False, border_style="blue"))

def run_predictions(files, bankroll=None):
    if len(files) == 5:
        print_header("Générateur de Pronostic BIG 5")
    
    all_winners = []
    for i, filename in enumerate(sorted(files)):
        console.print(f"\n[yellow]--- Analyse Course {i+1}/{len(files)} : {Path(filename).name} ---[/yellow]")
        file_to_predict = Path(filename)
        
        _04_predictor.predict(file_to_predict, bankroll=bankroll)
        
        top3_df = _04_predictor.predict(file_to_predict, return_top3=True)
        if top3_df is not None and not top3_df.empty:
            gagnant = top3_df.iloc[0]
            all_winners.append(str(gagnant['NUMERO_CHEVAL']))

    if len(all_winners) == 5:
        ticket_str = " / ".join(all_winners)
        console.print(Panel(
            f"[bold]Suggestion (1 cheval par course) :[/bold] [white on blue]{ticket_str}[/white on blue]\n[italic]Pour un pari 'Combiné', ajoutez des chevaux autour de ces suggestions.[/italic]",
            title="[bold green]🎟️ SUGGESTION TICKET BIG 5 🎟️[/bold green]",
            expand=False,
            border_style="green"
        ))

def main():
    parser = argparse.ArgumentParser(description="🏇 Pipeline Stratégique PMU - v6.0 (Visuel) 🏇")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles", required=True)

    subparsers.add_parser("prepare", help="Lancer le pré-traitement des données.")
    optimize_parser = subparsers.add_parser("optimize", help="Lancer la recherche des meilleurs hyperparamètres.")
    optimize_parser.add_argument("--trials", type=int, default=None, help="Nombre d'essais (par défaut : infini).")
    subparsers.add_parser("build", help="Construire le modèle final.")
    subparsers.add_parser("backtest", help="Lancer un backtest pour évaluer la rentabilité.")

    bet_parser = subparsers.add_parser("bet", help="Faire un pronostic et calculer les mises.")
    bet_parser.add_argument("--bankroll", type=float, required=True, help="Montant total de votre capital de jeu (ex: 100.0).")

    args = parser.parse_args()

    if args.command == "prepare":
        print_header("Lancement du Pré-Traitement des Données")
        _01_preprocessor.process_data()
    elif args.command == "optimize":
        print_header("Lancement de l'Optimisation des Modèles Experts")
        _03_model_trainer.optimize_models(n_trials=args.trials)
    elif args.command == "build":
        print_header("Construction du Modèle Stratégique Final")
        _03_model_trainer.build_model()
    elif args.command == "backtest":
        backtest.run_backtest()
    elif args.command == "bet":
        files = glob.glob(str(config.PREDICT_DIR / "*.csv"))
        file_count = len(files)
        if file_count == 0:
            console.print(Panel("[bold red]❌ Aucun fichier de course trouvé.[/bold red]", title="[red]Erreur[/red]", expand=False))
            return
        
        print_header(f"Calcul de Mises Stratégiques (Bankroll: €{args.bankroll:.2f})")
        run_predictions(files, bankroll=args.bankroll)

if __name__ == "__main__":
    if sys.platform == 'darwin': os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    main()
