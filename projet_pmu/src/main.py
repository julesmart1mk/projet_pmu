import os
import sys
import argparse
import glob
from pathlib import Path
from projet_pmu.src.pipelines import _01_preprocessor, _03_model_trainer, _04_predictor, _05_promo_manager
from projet_pmu.src import backtest
from projet_pmu.src import config

def print_header(title):
    width = 80; padding = (width - len(title) - 2) // 2
    print("\n" + "=" * width); print(" " * padding, f"🏇 {title} 🏇"); print("=" * width)

def run_big5(files):
    print_header("Générateur de Pronostic BIG 5")
    all_winners = []
    for i, filename in enumerate(sorted(files)):
        print(f"\n--- Analyse Course {i+1}/{len(files)} : {Path(filename).name} ---")
        file_to_predict = Path(filename)
        top3_df = _04_predictor.predict(file_to_predict, return_top3=True)
        if top3_df is not None and not top3_df.empty:
            gagnant = top3_df.iloc[0]; deuxieme = top3_df.iloc[1] if len(top3_df) > 1 else None
            print(f"🥇 Gagnant Prédit : N°{gagnant['NUMERO_CHEVAL']} ({gagnant['CHEVAL_ID']}) - Probabilité: {gagnant['PROBA_GAGNANT']:.2%}")
            if deuxieme is not None: print(f"🥈 Potentiel 2ème : N°{deuxieme['NUMERO_CHEVAL']} ({deuxieme['CHEVAL_ID']}) - Probabilité: {deuxieme['PROBA_GAGNANT']:.2%}")
            all_winners.append(str(gagnant['NUMERO_CHEVAL']))
        else: print("❌ Erreur pendant la prédiction de cette course.")
    if len(all_winners) == 5:
        print("\n" + "-"*50); print(" " * 10, "🎟️ VOTRE TICKET BIG 5 SUGGÉRÉ 🎟️"); print("-"*50)
        ticket_str = " / ".join(all_winners)
        print(f"\nSuggestion (1 cheval par course) : {ticket_str}\n")
        print("Pour un pari 'Combiné', vous pouvez ajouter des chevaux autour de ces suggestions."); print("-"*50)

def main():
    parser = argparse.ArgumentParser(description="🏇 Pipeline Stratégique PMU - v4.0 (Auto-Promo) 🏇")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles", required=True)

    subparsers.add_parser("prepare", help="Lancer uniquement le pré-traitement des données.")
    optimize_parser = subparsers.add_parser("optimize", help="Lancer/continuer la recherche des meilleurs hyperparamètres.")
    optimize_parser.add_argument("--trials", type=int, default=None, help="Nombre d'essais (optionnel, par défaut : infini).")
    subparsers.add_parser("build", help="Construire le modèle final en utilisant les meilleurs hyperparamètres trouvés.")

    predict_parser = subparsers.add_parser("predict", help="Faire un pronostic stratégique sur une course.")
    predict_parser.add_argument("--file", type=str, required=True, help="Nom du fichier CSV de la course.")
    auto_parser = subparsers.add_parser("auto", help="Lancer un pronostic intelligent (Big 5 ou simple).")
    subparsers.add_parser("check-promos", help="Vérifier et lister toutes les promotions détectées dans les PDF.")

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
    elif args.command == "check-promos":
        print_header("Gestionnaire de Promotions")
        _05_promo_manager.scan_and_parse_promotions(verbose=True)
    elif args.command == "auto":
        files = glob.glob(str(config.PREDICT_DIR / "*.csv"))
        file_count = len(files)
        if file_count == 1:
            print_header(f"Pronostic Stratégique pour {Path(files[0]).name}")
            _04_predictor.predict(Path(files[0]))
        elif file_count == 5: run_big5(files)
        else:
            print_header("Erreur de Fichiers")
            print(f"❌ {file_count} fichiers trouvés. Veuillez en placer 1 ou 5 dans '{config.PREDICT_DIR}'.")
    elif args.command == "predict":
        file_to_predict = config.PREDICT_DIR / args.file
        if not file_to_predict.exists():
            print_header("Erreur de Fichier"); print(f"❌ Le fichier {file_to_predict} est introuvable."); return
        print_header(f"Pronostic Stratégique pour {args.file}")
        _04_predictor.predict(file_to_predict)

if __name__ == "__main__":
    if sys.platform == 'darwin': os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    main()
