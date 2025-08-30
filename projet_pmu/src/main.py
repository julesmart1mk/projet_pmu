import argparse
import glob
from pathlib import Path
from src.pipelines import _01_preprocessor, _03_model_trainer, _04_predictor
from src import backtest
from src import config

def run_big5(files):
    print("--- 🏇 Lancement du générateur de pronostic Big 5 ---")
    all_winners = []
    
    for i, filename in enumerate(files):
        print(f"\n--- Analyse Course {i+1}/{len(files)} : {Path(filename).name} ---")
        file_to_predict = Path(filename)
        if not file_to_predict.exists():
            print(f"❌ Erreur : Le fichier {file_to_predict} est introuvable. Annulation.")
            return
        
        # On demande le top 3 pour un affichage plus riche
        top3_df = _04_predictor.predict(file_to_predict, return_top3=True)
        
        if top3_df is not None and not top3_df.empty:
            gagnant = top3_df.iloc[0]
            deuxieme = top3_df.iloc[1] if len(top3_df) > 1 else None
            
            print(f"🥇 Gagnant Prédit : N°{gagnant['NUMERO_CHEVAL']} ({gagnant['CHEVAL_ID']}) - Probabilité: {gagnant['PROBA_GAGNANT']:.2%}")
            if deuxieme is not None:
                print(f"🥈 Potentiel 2ème : N°{deuxieme['NUMERO_CHEVAL']} ({deuxieme['CHEVAL_ID']}) - Probabilité: {deuxieme['PROBA_GAGNANT']:.2%}")
            
            all_winners.append(str(gagnant['NUMERO_CHEVAL']))
        else:
            print("❌ Une erreur est survenue pendant la prédiction. Annulation.")
            return
            
    print("\n\n--- 🎟️ VOTRE TICKET BIG 5 SUGGÉRÉ ---")
    ticket_str = " / ".join(all_winners)
    print(f"Suggestion (1 cheval par course) : {ticket_str}")
    print("\nPour un pari 'Combiné', vous pouvez ajouter manuellement des chevaux autour de ces suggestions.")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de pronostic PMU")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles", required=True)
    
    train_parser = subparsers.add_parser("train", help="Lancer l'entraînement du modèle.")
    train_parser.add_argument("--trials", type=int, default=100, help="Nombre d'essais Optuna par modèle.")
    build_parser = subparsers.add_parser("build", help="Assemble le modèle final à partir des recherches sauvegardées.")
    auto_parser = subparsers.add_parser("auto", help="Lance un pronostic intelligent selon le nombre de fichiers.")
    predict_parser = subparsers.add_parser("predict", help="Faire un pronostic sur une course.")
    predict_parser.add_argument("--file", type=str, required=True, help="Nom du fichier CSV de la course.")
    big5_parser = subparsers.add_parser("big5", help="Générer un pronostic Big 5 sur 5 courses.")
    big5_parser.add_argument("--files", nargs=5, required=True, metavar='FILE', help="Les 5 noms de fichiers CSV des courses du Big 5.")
    backtest_parser = subparsers.add_parser("backtest", help="Lancer une simulation sur les données historiques.")
    
    args = parser.parse_args()

    if args.command == "train":
        _01_preprocessor.process_data()
        _03_model_trainer.train_model(n_trials=args.trials)
    elif args.command == "build":
        _03_model_trainer.train_model(build_only=True)
    elif args.command == "auto":
        files = glob.glob(str(config.PREDICT_DIR / "*.csv"))
        file_count = len(files)
        if file_count == 1:
            print("INFO: 1 fichier détecté, lancement du pronostic détaillé...")
            _04_predictor.predict(Path(files[0]))
        elif file_count == 5:
            print("INFO: 5 fichiers détectés, lancement du pronostic Big 5...")
            run_big5(files)
        else:
            print(f"❌ Erreur : {file_count} fichiers trouvés dans 'data/a_predire'."); print("   Veuillez y placer 1 fichier pour un pronostic simple, ou 5 fichiers pour un Big 5.")
    elif args.command == "predict":
        file_to_predict = config.PREDICT_DIR / args.file
        if not file_to_predict.exists(): print(f"❌ Erreur : Le fichier {file_to_predict} n'a pas été trouvé."); return
        _04_predictor.predict(file_to_predict)
    elif args.command == "big5":
        files_full_path = [str(config.PREDICT_DIR / f) for f in args.files]
        run_big5(files_full_path)
    elif args.command == "backtest":
        backtest.run_backtest()

if __name__ == "__main__":
    main()
