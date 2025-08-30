import PyPDF2
import re
import datetime
from pathlib import Path
from projet_pmu.src import config

def extract_text_from_pdf(pdf_path):
    """Ouvre un PDF et en extrait le texte brut."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                # Remplace les retours à la ligne par des espaces pour fiabiliser les regex
                text += page.extract_text().replace('\n', ' ')
        return " ".join(text.split()) # Nettoie les espaces multiples
    except Exception as e:
        print(f"⚠️  Erreur de lecture du PDF {pdf_path.name}: {e}")
        return ""

def parse_promotion_text(text):
    """Analyse le texte pour trouver les détails de la promotion."""
    # Regex améliorées pour plus de précision
    hippodrome_re = re.search(r"à (\w+)", text, re.IGNORECASE)
    date_re = re.search(r"(\d{1,2} \w+ \d{4})", text)
    # Cible spécifiquement la phrase "augmentés de XX%"
    boost_re = re.search(r"augmentés de (\d+)%", text)
    types_re = re.findall(r"e-([\w-]+)®?", text)

    mois_map = {
        'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
        'juillet': 7, 'août': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
    }

    if not all([hippodrome_re, date_re, boost_re]):
        return None

    try:
        hippodrome = hippodrome_re.group(1).upper()
        date_str = date_re.group(1)
        jour, mois_fr, annee = date_str.split()
        date = datetime.date(int(annee), mois_map[mois_fr.lower()], int(jour))
        boost = 1 + (int(boost_re.group(1)) / 100)
        types = list(set([f"e-{t.capitalize()}" for t in types_re])) if types_re else []
        description = f"Gains boostés de {int((boost-1)*100)}% à {hippodrome.capitalize()}"

        return {"hippodrome": hippodrome, "date": date, "boost": boost, "types": types, "description": description}
    except Exception:
        return None

def scan_and_parse_promotions(verbose=False):
    """Scanne le dossier des promotions, analyse les PDF et retourne une liste de toutes les promotions trouvées."""
    promo_dir = config.PROMOTIONS_PDF_DIR
    all_promotions = []
    
    if verbose: print(f"🔍 Scan du dossier des promotions : {promo_dir}")
    pdf_files = list(promo_dir.glob("*.pdf"))

    if not pdf_files and verbose: print("  - Aucun fichier de promotion (.pdf) trouvé."); return []
    
    for pdf_file in pdf_files:
        if verbose: print(f"\n--- Analyse du fichier : {pdf_file.name} ---")
        text = extract_text_from_pdf(pdf_file)
        if not text: continue

        promo_data = parse_promotion_text(text)
        
        if promo_data:
            promo_data["source_file"] = pdf_file.name
            promo_data["is_expired"] = promo_data['date'] < datetime.date.today()
            all_promotions.append(promo_data)
            if verbose:
                print(f"  ✅ Promotion trouvée : {promo_data['description']}")
                print(f"     - Date : {promo_data['date'].strftime('%d/%m/%Y')}")
                print(f"     - Statut : {'Expirée' if promo_data['is_expired'] else 'Active'}")
        elif verbose:
            print("  ❌ Aucune information de promotion valide n'a pu être extraite de ce fichier.")
    
    return all_promotions
