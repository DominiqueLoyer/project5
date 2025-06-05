# -*- coding: utf-8 -*-
#
# ==================================================
# PLEASE CITE IF YOU USE THE CODE 
# Citation Key: loyerEvaluationModelesRecherche2025
# ===================================================
# ====================================================
# AUTHOR : DOMINIQUE S. LOYER
# TITLE:  Évaluation de Modèles de Recherche d’Information et d’Expansion de Requêtes sur la Collection TREC AP 88-90
# 28 AVRIL 2025
# ====================================================================================================================
#
#
# === Cellule 0.1: Monter Google Drive ===
from google.colab import drive
drive.mount('/content/drive')

# Vérifiez que le dossier du projet est accessible
# Adaptez le chemin si nécessaire en fonction de l'emplacement dans votre Drive
!ls "/content/drive/MyDrive/Projet_RI"

# === Cellule de Vérification du Contenu du Dossier Runs (Corrigé) ===
# Utilise les commandes shell de Colab préfixées par '!'

# Chemin exact où les résultats de recherche sont attendus
# (Défini dans la cellule de configuration complète)
RUN_DIR_PATH="/content/ap_output/runs/"

# Utiliser '!' pour exécuter la commande shell 'echo'
print(f"Vérification du contenu de : {RUN_DIR_PATH}")

# Utiliser '!' pour exécuter la commande shell 'ls -lh'
# Mettre le chemin entre guillemets pour gérer les espaces potentiels (même s'il n'y en a pas ici)
!ls -lh "{RUN_DIR_PATH}"

# === Cellule 4: Exécuter les Recherches (Séquentielles - BM25 & QLD) ===
# Lance les 8 combinaisons de recherche en utilisant BM25 et QLD.
# S'assure que l'environnement Java 21 est actif et que les index/variables sont définis/restaurés.

# Assurer que pyserini est installé avant l'import
# Vous devriez normalement exécuter la Cellule 0 "Configuration Complète" avant celle-ci.
# Cette ligne est ajoutée comme filet de sécurité si la Cellule 0 n'a pas été exécutée
# ou a échoué pour pyserini. Supprimez-la si vous exécutez toujours la Cellule 0.
!pip install pyserini --quiet

from pyserini.search.lucene import LuceneSearcher # Import principal
import time
from tqdm.notebook import tqdm
import traceback
import os
from jnius import JavaException # Importer seulement JavaException, ClassicSimilarity n'est pas utilisé

# Définir K_RESULTS
try: K_RESULTS
except NameError: print("Définition K_RESULTS=1000"); K_RESULTS = 1000

# Vérifier variables nécessaires et existence des index restaurés
try:
    INDEX_DIR_BASELINE; INDEX_DIR_PREPROC; RUN_DIR; K_RESULTS; CORPUS_DIR; # Ajout CORPUS_DIR pour vérif jsonl
    queries_short; queries_long; queries_short_preprocessed; queries_long_preprocessed;
    preprocess_text;
    if not os.path.exists(INDEX_DIR_BASELINE): raise FileNotFoundError(f"Index Baseline restauré manquant: {INDEX_DIR_BASELINE}")
    if not os.path.exists(INDEX_DIR_PREPROC): raise FileNotFoundError(f"Index Preprocessed restauré manquant: {INDEX_DIR_PREPROC}")
    # Vérifier aussi que les fichiers de corpus sont là (restaurés ou recréés)
    if not os.path.exists(os.path.join(CORPUS_DIR, "ap_docs.jsonl")): raise FileNotFoundError("ap_docs.jsonl manquant.")
    if not os.path.exists(os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl")): raise FileNotFoundError("ap_docs_preprocessed.jsonl manquant.")

except NameError as e: print(f"ERREUR: Variable {e} manquante. Exécutez config complète."); raise
except FileNotFoundError as e: print(f"ERREUR: {e}"); raise

def perform_search_sequential_qld(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes (BM25 ou QLD)."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}"
    print(f"\nDébut recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', k={k}")

    all_results_list = []
    searcher = None

    try:
        print(f"  Initialisation LuceneSearcher..."); searcher = LuceneSearcher(index_path); print(f"  LuceneSearcher initialisé.")

        # Configurer similarité
        if model == 'bm25':
            print("  Configuration BM25..."); searcher.set_bm25(k1=0.9, b=0.4); print("  BM25 configuré.")
        elif model == 'qld': # Utiliser Query Likelihood Dirichlet
            print("  Configuration QLD..."); searcher.set_qld(); print("  QLD configuré.")
        else:
            print(f"Modèle '{model}' non reconnu, utilise BM25 par défaut."); searcher.set_bm25()

        # Itérer sur les requêtes
        query_errors = 0
        if 'preprocess_text' not in globals(): raise NameError("preprocess_text non définie.")

        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                if not search_text.strip(): continue # Ignorer requêtes vides

                hits = searcher.search(search_text, k=k)

                for i in range(len(hits)):
                    rank, doc_id, score = i + 1, hits[i].docid, hits[i].score
                    if doc_id is None: continue
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
            except Exception as e_query:
                query_errors += 1
                if query_errors < 5: print(f"\nErreur recherche QID {query_id}: {e_query}")
                elif query_errors == 5: print("\nPlusieurs erreurs recherche...")

        # Écrire résultats
        if all_results_list:
             # S'assurer que le dossier RUN_DIR existe avant d'écrire
             os.makedirs(os.path.dirname(output_run_file), exist_ok=True)
             with open(output_run_file, 'w', encoding='utf-8') as f_out: f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes résultats écrites dans {os.path.basename(output_run_file)}.")
        else: print("\n  Avertissement: Aucun résultat généré pour ce run.")
        if query_errors > 0: print(f"  Avertissement: {query_errors} erreurs sur requêtes.")

        end_time = time.time()
        print(f"Recherche terminée pour {run_tag}. Sauvegardé dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")
    except Exception as e_main: print(f"\nERREUR MAJEURE run {run_tag}: {e_main}"); traceback.print_exc()
    finally:
        if searcher: print(f"  Nettoyage implicite ressources {run_tag}.")

# --- Exécution des 8 configurations (BM25 et QLD) ---
print("\n--- DÉBUT DES RECHERCHES BASELINE (BM25/QLD) ---")
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt"); perform_search_sequential_qld(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")
run_file_2 = os.path.join(RUN_DIR, "baseline_short_qld.txt"); perform_search_sequential_qld(queries_short, INDEX_DIR_BASELINE, 'qld', K_RESULTS, run_file_2, "baseline_short")
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt"); perform_search_sequential_qld(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")
run_file_4 = os.path.join(RUN_DIR, "baseline_long_qld.txt"); perform_search_sequential_qld(queries_long, INDEX_DIR_BASELINE, 'qld', K_RESULTS, run_file_4, "baseline_long")
print("\n--- DÉBUT DES RECHERCHES PRÉTRAITÉES (BM25/QLD) ---")
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt"); perform_search_sequential_qld(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)
run_file_6 = os.path.join(RUN_DIR, "preproc_short_qld.txt"); perform_search_sequential_qld(queries_short_preprocessed, INDEX_DIR_PREPROC, 'qld', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt"); perform_search_sequential_qld(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)
run_file_8 = os.path.join(RUN_DIR, "preproc_long_qld.txt"); perform_search_sequential_qld(queries_long_preprocessed, INDEX_DIR_PREPROC, 'qld', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)
print("\n--- Toutes les recherches de base (BM25/QLD) sont terminées. ---")

# Vérifier si des fichiers ont été créés
print(f"\nVérification du contenu de {RUN_DIR} après les recherches...")
!ls -l {RUN_DIR}

# === Cellule de Configuration Complète (avec Stemming) ===
# Installe Java 21, configure comme défaut, installe outils build,
# pybind11, dernière Pyserini, NLTK+ressources, définit chemins,
# FONCTION preprocess_text AVEC STEMMING, parse topics.

import os
import sys
import subprocess
import time
import glob
import re
import json
import nltk # Importer nltk ici pour la partie NLTK
from tqdm.notebook import tqdm # Assurer l'import pour les fonctions
import traceback # Pour afficher les erreurs

print("--- Début de la Configuration Complète (avec Stemming) ---")
print("Cela peut prendre plusieurs minutes...")

# --- Partie 1: Installation Java 21 et Configuration ---
print("\n[1/9] Installation de OpenJDK 21...")
install_java_cmd = "apt-get update -qq > /dev/null && apt-get install -y openjdk-21-jdk-headless -qq > /dev/null"
try:
    subprocess.run(install_java_cmd, shell=True, check=True, timeout=180)
    print("OpenJDK 21 installé.")
except Exception as e:
    print(f"ERREUR lors de l'installation de Java 21: {e}")
    raise # Arrêter si Java ne s'installe pas

print("\n[2/9] Configuration de Java 21 comme défaut via update-alternatives...")
java_path_21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
if os.path.exists(java_path_21):
    try:
        subprocess.run(f"update-alternatives --install /usr/bin/java java {java_path_21} 1", shell=True, check=True)
        subprocess.run(f"update-alternatives --set java {java_path_21}", shell=True, check=True)
        print("update-alternatives configuré pour java.")
    except Exception as e:
        print(f"ERREUR lors de la configuration de update-alternatives: {e}")
        # Continuer mais avertir
else:
    print(f"ATTENTION: Chemin Java 21 non trouvé à {java_path_21}. update-alternatives non configuré.")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
if not os.path.exists(os.environ["JAVA_HOME"]):
     print(f"ATTENTION: Le chemin JAVA_HOME '{os.environ['JAVA_HOME']}' n'existe pas.")

# --- Partie 2: Installation des outils de build C++ ---
print("\n[3/9] Installation des outils de build (build-essential, cmake)...")
install_build_cmd = "apt-get update -qq > /dev/null && apt-get install -y build-essential cmake -qq > /dev/null"
try:
    subprocess.run(install_build_cmd, shell=True, check=True, timeout=180)
    print("Outils de build installés.")
except Exception as e_build:
    print(f"ERREUR lors de l'installation des outils de build: {e_build}")
    # Continuer mais avertir

# --- Partie 3: Installation de pybind11 ---
print("\n[4/9] Installation de pybind11...")
install_pybind_cmd = f"{sys.executable} -m pip install pybind11 -q" # -q peut être enlevé si ça échoue
try:
    subprocess.run(install_pybind_cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
    print("pybind11 installé avec succès.")
except Exception as e_pybind:
    print(f"ERREUR lors de l'installation de pybind11: {e_pybind}")
    # Continuer mais avertir

# --- Partie 4: Installation des Paquets Python Principaux ---
print("\n[5/9] Installation de la DERNIÈRE Pyserini, NLTK, Pytrec_eval...")
# Installer sans -q pour voir les erreurs si ça se reproduit
install_pip_cmd = f"{sys.executable} -m pip install pyserini nltk pytrec_eval"
try:
    result_pip = subprocess.run(install_pip_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600)
    print("Paquets Python principaux installés.")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: La commande pip principale a échoué avec le code {e.returncode}")
    print("Sortie STDOUT de Pip:\n", e.stdout)
    print("Sortie STDERR de Pip:\n", e.stderr)
    raise e # Arrêter si l'installation de pyserini échoue
except subprocess.TimeoutExpired as e:
    print("\nERREUR: La commande pip principale a dépassé le délai d'attente.")
    print("Sortie STDOUT de Pip (partielle):\n", e.stdout)
    print("Sortie STDERR de Pip (partielle):\n", e.stderr)
    raise e
except Exception as e_pip:
    print(f"\nERREUR inattendue lors de l'installation pip: {e_pip}")
    raise e_pip

# --- Partie 5: Téléchargement Ressources NLTK ---
print("\n[6/9] Téléchargement/Vérification des ressources NLTK...")
# S'assurer que nltk est importé
import nltk
# Liste incluant la correction pour punkt_tab
nltk_resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4', 'punkt_tab']
for resource in nltk_resources:
    try:
        # Déterminer le chemin de recherche correct pour nltk.data.find
        if resource == 'punkt' or resource == 'punkt_tab': # punkt_tab est aussi dans tokenizers
            resource_path = f'tokenizers/{resource}.zip'
        elif resource == 'omw-1.4':
             resource_path = f'corpora/{resource}.zip' # Open Multilingual Wordnet
        elif resource == 'wordnet':
             resource_path = f'corpora/{resource}.zip'
        else: # stopwords, etc.
            resource_path = f'corpora/{resource}.zip'

        # Essayer de trouver la ressource
        nltk.data.find(resource_path)
        # print(f"  Ressource NLTK '{resource}' déjà présente.")

    # Utiliser except LookupError (correction appliquée)
    except LookupError:
        print(f"  Ressource NLTK '{resource}' non trouvée. Téléchargement...")
        try:
            nltk.download(resource, quiet=True)
            print(f"  Ressource '{resource}' téléchargée.")
        except Exception as e_download:
            # Capturer les erreurs potentielles de téléchargement (réseau, etc.)
            print(f"  ERREUR lors du téléchargement de '{resource}': {e_download}")

print("Ressources NLTK prêtes.")

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")

# !!! MODIFIEZ CETTE LIGNE AVEC LE CHEMIN CORRECT VERS VOTRE DOSSIER TREC !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # <--- VÉRIFIEZ ET CORRIGEZ CE CHEMIN !

# --- Vérification et définition des autres chemins ---
if 'google.colab' in sys.modules:
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/My Drive'):
             print("  Montage de Google Drive...")
             drive.mount('/content/drive', force_remount=True)
        else:
             print("  Google Drive déjà monté.")
    except ModuleNotFoundError:
         print("ATTENTION: Google Colab non détecté ou erreur d'import.")
    except Exception as e_mount:
         print(f"ATTENTION: Erreur lors du montage de Drive: {e_mount}")

if not os.path.exists(DRIVE_PROJECT_PATH):
     raise FileNotFoundError(f"Le chemin Drive '{DRIVE_PROJECT_PATH}' n'existe pas. Vérifiez le chemin exact et le nom des dossiers.")

print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

AP_TAR_FILENAME = "AP.tar" # Nom du fichier archive (corrigé)
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, AP_TAR_FILENAME)
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence")
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed") # Sera recréé avec stemming
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
# Créer les répertoires de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP CIBLE: {AP_TAR_PATH}")
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# --- Partie 7: Définition Fonction Prétraitement (AVEC STEMMING) ---
print("\n[8/9] Définition de la fonction preprocess_text (avec Stemming)...")
# S'assurer que nltk est importé avant d'utiliser ses modules
import nltk
from nltk.corpus import stopwords
# --- Utilisation de PorterStemmer ---
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
# Utiliser des noms de variables différents pour éviter conflits potentiels
stop_words_set_global = set(stopwords.words('english'))
# --- Création de l'objet Stemmer ---
stemmer_obj_global = PorterStemmer()
def preprocess_text(text):
    """Applique tokenisation, minuscules, suppression ponctuation/non-alpha, stop words ET STEMMING (Porter)."""
    if not isinstance(text, str): return ""
    try:
        tokens = word_tokenize(text.lower())
    except LookupError as e_tok: # Gestion erreur si ressource NLTK manque
         if 'Resource' in str(e_tok) and 'not found' in str(e_tok):
              resource_name = str(e_tok).split('Resource ')[1].split(' ')[0]
              print(f"--- Tokenizer a besoin de '{resource_name}', tentative téléchargement ---")
              try:
                  nltk.download(resource_name, quiet=True)
                  print(f"--- Ressource '{resource_name}' téléchargée, nouvelle tentative de tokenisation ---")
                  tokens = word_tokenize(text.lower()) # Retenter après téléchargement
              except Exception as e_dl_tok:
                  print(f"--- Échec du téléchargement de '{resource_name}': {e_dl_tok} ---")
                  raise e_tok # Relancer l'erreur originale si le téléchargement échoue
         else:
              raise e_tok # Relancer si ce n'est pas une ressource manquante connue
    except Exception as e_tok_other:
         print(f"Erreur inattendue dans word_tokenize: {e_tok_other}")
         raise e_tok_other
    # --- Application du Stemmer ---
    filtered_tokens = [stemmer_obj_global.stem(w) for w in tokens if w.isalpha() and w not in stop_words_set_global]
    return ' '.join(filtered_tokens)
print("  Fonction preprocess_text définie avec PorterStemmer.")
# Tester la nouvelle fonction
sample_text = "This is an example showing Information Retrieval with stemming and stop words removal."
stemmed_sample = preprocess_text(sample_text)
print(f"  Exemple Stemmed: {stemmed_sample}") # Doit afficher 'thi is exampl show inform retriev with stem and stop word remov.'

# --- Partie 8: Parsing des Topics ---
print("\n[9/9] Parsing des topics...")
# S'assurer que re et glob sont importés
import re
import glob
def parse_topics(file_path):
    """Parse un fichier topic TREC standard."""
    topics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
                topic_content = top_match.group(1)
                num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE)
                if not num_match: continue
                topic_id = num_match.group(1).strip()
                title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                desc = desc_match.group(1).strip() if desc_match else ""
                if topic_id and title:
                     topics[topic_id] = {'title': title, 'desc': desc}
    except FileNotFoundError:
        print(f"  ATTENTION: Fichier topic non trouvé: {file_path}")
    except Exception as e_topic:
        print(f"  ATTENTION: Erreur lors du parsing de {file_path}: {e_topic}")
    return topics

if not os.path.exists(TOPICS_DIR):
     print(f"ATTENTION: Le dossier des topics '{TOPICS_DIR}' n'existe pas.")
     topic_files = []
else:
    topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))

all_topics = {}
if not topic_files:
     print(f"  ATTENTION: Aucun fichier topic trouvé dans {TOPICS_DIR}")
else:
    print(f"  Parsing des fichiers topics: {topic_files}")
    for tf in topic_files:
        all_topics.update(parse_topics(tf))

# Définir les dictionnaires même s'ils sont vides pour éviter NameError plus tard
# Mettre la création des dictionnaires prétraités dans un try-except
try:
    queries_short = {qid: data['title'] for qid, data in all_topics.items()}
    queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()}
    print(f"  {len(all_topics)} topics parsés.")
    print(f"  {len(queries_short)} requêtes courtes brutes créées.")
    print(f"  Prétraitement des requêtes (avec stemming)...")
    # Appliquer la NOUVELLE fonction preprocess_text (avec stemming)
    queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
    queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}
    print(f"  Prétraitement des requêtes terminé.")
except Exception as e_preproc_queries:
     print(f"\nERREUR lors du prétraitement des requêtes: {e_preproc_queries}")
     print("Les dictionnaires prétraités pourraient être incomplets ou vides.")
     # Créer des dictionnaires vides pour éviter NameError plus tard
     queries_short_preprocessed = {}
     queries_long_preprocessed = {}


# --- Vérification Finale Java ---
print("\n--- Vérification Finale de la Version Java Active ---")
java_check_cmd = "java -version"
try:
    result = subprocess.run(java_check_cmd, shell=True, check=True, capture_output=True, text=True, timeout=10)
    print("Sortie STDERR (contient souvent la version OpenJDK):\n", result.stderr)
    if "21." not in result.stderr and "21." not in result.stdout:
         print("\nATTENTION: Java 21 ne semble PAS être la version active !")
    else:
         print("\nConfirmation: Java 21 semble être la version active.")
except Exception as e:
    print(f"\nERREUR lors de la vérification Java: {e}")

# --- Vérification Finale Pyserini ---
print("\n--- Vérification Finale de la Version Pyserini Installée ---")
try:
    result_pyserini = subprocess.run(f"{sys.executable} -m pip show pyserini", shell=True, check=True, capture_output=True, text=True, timeout=30)
    print(result_pyserini.stdout)
except Exception as e_pyserini_check:
    print(f"ERREUR lors de la vérification de Pyserini: {e_pyserini_check}")

print("\n--- Configuration Complète (avec Stemming) Terminée ---")
# Ajouter un délai pour s'assurer que tout est stable avant la prochaine cellule
print("\nPause de 5 secondes...")
time.sleep(5)
print("Prêt pour la suite.")

# === Cellule 1: Extraire, Décompresser et Formater les Documents ===
# Lit AP.tar, décompresse les .gz internes, extrait <DOC>, <DOCNO>, <TEXT>
# et écrit le résultat dans ap_docs.jsonl.

import tarfile
import re
import json
import gzip # Importer le module gzip
from tqdm.notebook import tqdm
import os
import traceback

# Vérifier que les chemins sont définis (normalement fait par la cellule de config)
try:
    AP_TAR_PATH
    CORPUS_DIR
except NameError:
    print("ERREUR: Variables de chemin non définies. Exécutez la cellule de configuration complète.")
    raise

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction, Décompression et Formatage depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# Vérifier si le fichier AP.tar existe
if not os.path.exists(AP_TAR_PATH):
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé.")
else:
    tar_size = os.path.getsize(AP_TAR_PATH)
    print(f"  Taille du fichier {AP_TAR_PATH}: {tar_size} octets.") # Devrait être ~275Mo

# Regex pour extraire les infos
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

# Compteurs
doc_count = 0
file_read_count = 0
skipped_members = 0
decompression_errors = 0

try:
    # Ouvrir le fichier de sortie et l'archive TAR
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"\n{len(members)} membres trouvés dans l'archive TAR.") # Devrait être ~1051

        # Boucler sur chaque membre de l'archive
        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            # Ignorer si ce n'est pas un fichier .gz ou .Z
            if not member.isfile() or not member.name.lower().endswith(('.gz', '.z')):
                skipped_members += 1
                continue

            file_read_count += 1
            content = "" # Réinitialiser pour chaque fichier

            try:
                # Extraire le contenu compressé
                f = tar.extractfile(member)
                if f:
                    compressed_content = f.read()
                    f.close()

                    # Décompresser le contenu
                    try:
                        content_bytes = gzip.decompress(compressed_content)
                        content = content_bytes.decode('utf-8', errors='ignore') # Décoder après décompression
                    except gzip.BadGzipFile: # Gérer si ce n'est pas du gzip
                        content = compressed_content.decode('utf-8', errors='ignore')
                        decompression_errors += 1
                    except Exception as e_gzip:
                         print(f"\nErreur de décompression pour {member.name}: {e_gzip}")
                         decompression_errors += 1
                         continue # Passer au suivant

                    # Trouver tous les blocs <DOC> dans le contenu décompressé
                    doc_matches = doc_pattern.findall(content)
                    if not doc_matches: continue # Passer si aucun doc trouvé

                    # Boucler sur chaque document trouvé
                    for doc_content in doc_matches:
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match: continue
                        doc_id = docno_match.group(1).strip()

                        text_match = text_pattern.search(doc_content)
                        # Nettoyer le texte (espaces multiples)
                        doc_text = ' '.join(text_match.group(1).strip().split()) if text_match else ""

                        # Écrire la ligne JSONL
                        try:
                            json_line = json.dumps({"id": str(doc_id), "contents": str(doc_text)})
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key: print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}"); skipped_members += 1
            except EOFError: print(f"\nAvertissement: Fin fichier inattendue {member.name}."); skipped_members += 1
            except Exception as e_extract: print(f"\nErreur extraction/lecture {member.name}: {e_extract}"); skipped_members += 1

except tarfile.ReadError as e_tar: print(f"\nERREUR lecture TAR {AP_TAR_PATH}: {e_tar}"); raise e_tar
except FileNotFoundError: print(f"\nERREUR: Fichier TAR {AP_TAR_PATH} non trouvé."); raise FileNotFoundError
except Exception as e_general: print(f"\nERREUR générale traitement TAR: {e_general}"); traceback.print_exc(); raise e_general

# Afficher le résumé de l'extraction
print(f"\n--- Fin de l'Extraction et Décompression ---")
print(f"  {file_read_count} fichiers (.gz/.Z) lus.")
print(f"  {skipped_members} membres ignorés.")
if decompression_errors > 0: print(f"  {decompression_errors} erreurs/avertissements décompression.")
print(f"  {doc_count} documents écrits dans {JSONL_OUTPUT_PATH}") # Devrait être ~240k

# Vérifier la taille du fichier de sortie
if os.path.exists(JSONL_OUTPUT_PATH):
    output_size = os.path.getsize(JSONL_OUTPUT_PATH)
    print(f"  Taille finale {JSONL_OUTPUT_PATH}: {output_size} octets.") # Devrait être ~600Mo
    if output_size > 0 and doc_count > 0: print("  SUCCÈS: Fichier de sortie contient des données.")
    else: print("  ATTENTION: Fichier de sortie vide ou aucun document écrit.")
else: print(f"  ATTENTION: Fichier {JSONL_OUTPUT_PATH} non créé.")

# === Cellule 1: Extraire, Décompresser et Formater les Documents ===
# Lit AP.tar, décompresse les .gz internes, extrait <DOC>, <DOCNO>, <TEXT>
# et écrit le résultat dans ap_docs.jsonl.

import tarfile
import re
import json
import gzip # Importer le module gzip
from tqdm.notebook import tqdm
import os
import traceback
import time # Import time for potential pause

# Vérifier que les chemins sont définis (normalement fait par la cellule de config)
try:
    AP_TAR_PATH
    CORPUS_DIR
except NameError:
    print("ERREUR: Variables de chemin non définies. Exécutez la cellule de configuration complète.")
    raise

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction, Décompression et Formatage depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# --- Ajout pour débogage ---
print(f"Vérification de l'existence de {AP_TAR_PATH}...")
# Utilisez la commande shell 'ls' pour lister le contenu du répertoire parent
# et vérifier la présence du fichier AP.tar
ap_tar_dir = os.path.dirname(AP_TAR_PATH)
print(f"Contenu de {ap_tar_dir}:")
!ls -lh "{ap_tar_dir}"
# --- Fin de l'ajout pour débogage ---


# Vérifier si le fichier AP.tar existe
if not os.path.exists(AP_TAR_PATH):
    print(f"\nERREUR: Le fichier AP.tar n'a pas été trouvé à l'emplacement attendu.")
    print(f"Veuillez vérifier que le chemin '{AP_TAR_PATH}' est correct et que le fichier y est présent.")
    print("Vérifiez aussi que votre Google Drive est bien monté.")
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé.")
else:
    tar_size = os.path.getsize(AP_TAR_PATH)
    print(f"  Succès: Le fichier {AP_TAR_PATH} a été trouvé. Taille: {tar_size} octets.") # Devrait être ~275Mo

# Regex pour extraire les infos
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

# Compteurs
doc_count = 0
file_read_count = 0
skipped_members = 0
decompression_errors = 0

try:
    # Ouvrir le fichier de sortie et l'archive TAR
    # Ajouter un court délai avant d'ouvrir le TAR pour s'assurer que le FS est prêt après le ls
    time.sleep(1)
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"\n{len(members)} membres trouvés dans l'archive TAR.") # Devrait être ~1051

        # Boucler sur chaque membre de l'archive
        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            # Ignorer si ce n'est pas un fichier .gz ou .Z
            if not member.isfile() or not member.name.lower().endswith(('.gz', '.z')):
                skipped_members += 1
                continue

            file_read_count += 1
            content = "" # Réinitialiser pour chaque fichier

            try:
                # Extraire le contenu compressé
                f = tar.extractfile(member)
                if f:
                    compressed_content = f.read()
                    f.close()

                    # Décompresser le contenu
                    try:
                        content_bytes = gzip.decompress(compressed_content)
                        content = content_bytes.decode('utf-8', errors='ignore') # Décoder après décompression
                    except gzip.BadGzipFile: # Gérer si ce n'est pas du gzip
                        content = compressed_content.decode('utf-8', errors='ignore')
                        decompression_errors += 1
                    except Exception as e_gzip:
                         print(f"\nErreur de décompression pour {member.name}: {e_gzip}")
                         decompression_errors += 1
                         continue # Passer au suivant

                    # Trouver tous les blocs <DOC> dans le contenu décompressé
                    doc_matches = doc_pattern.findall(content)
                    if not doc_matches: continue # Passer si aucun doc trouvé

                    # Boucler sur chaque document trouvé
                    for doc_content in doc_matches:
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match: continue
                        doc_id = docno_match.group(1).strip()

                        text_match = text_pattern.search(doc_content)
                        # Nettoyer le texte (espaces multiples)
                        doc_text = ' '.join(text_match.group(1).strip().split()) if text_match else ""

                        # Écrire la ligne JSONL
                        try:
                            json_line = json.dumps({"id": str(doc_id), "contents": str(doc_text)})
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key: print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}"); skipped_members += 1
            except EOFError: print(f"\nAvertissement: Fin fichier inattendue {member.name}."); skipped_members += 1
            except Exception as e_extract: print(f"\nErreur extraction/lecture {member.name}: {e_extract}"); skipped_members += 1

except tarfile.ReadError as e_tar: print(f"\nERREUR lecture TAR {AP_TAR_PATH}: {e_tar}"); raise e_tar
except FileNotFoundError: print(f"\nERREUR: Fichier TAR {AP_TAR_PATH} non trouvé."); raise FileNotFoundError # Should ideally not be reached if the check above works
except Exception as e_general: print(f"\nERREUR générale traitement TAR: {e_general}"); traceback.print_exc(); raise e_general

# Afficher le résumé de l'extraction
print(f"\n--- Fin de l'Extraction et Décompression ---")
print(f"  {file_read_count} fichiers (.gz/.Z) lus.")
print(f"  {skipped_members} membres ignorés.")
if decompression_errors > 0: print(f"  {decompression_errors} erreurs/avertissements décompression.")
print(f"  {doc_count} documents écrits dans {JSONL_OUTPUT_PATH}") # Devrait être ~240k

# Vérifier la taille du fichier de sortie
if os.path.exists(JSONL_OUTPUT_PATH):
    output_size = os.path.getsize(JSONL_OUTPUT_PATH)
    print(f"  Taille finale {JSONL_OUTPUT_PATH}: {output_size} octets.") # Devrait être ~600Mo
    if output_size > 0 and doc_count > 0: print("  SUCCÈS: Fichier de sortie contient des données.")
    else: print("  ATTENTION: Fichier de sortie vide ou aucun document écrit.")
else: print(f"  ATTENTION: Fichier {JSONL_OUTPUT_PATH} non créé.")

# === Cellule 2: Indexation Baseline ===
# Crée l'index Lucene à partir de ap_docs.jsonl (sans prétraitement spécifique).

import os
import subprocess
import traceback

# Vérifier que les chemins sont définis (normalement fait par la cellule de config)
try:
    CORPUS_DIR
    INDEX_DIR_BASELINE
except NameError:
    print("ERREUR: Variables de chemin non définies. Exécutez la cellule de configuration complète.")
    raise

print(f"Début de l'indexation Baseline...")
print(f"Dossier source: {CORPUS_DIR}")
print(f"Répertoire de l'index cible: {INDEX_DIR_BASELINE}")

# Vérifier si le fichier source existe et n'est pas vide
jsonl_source_path = os.path.join(CORPUS_DIR, "ap_docs.jsonl")
if not os.path.exists(jsonl_source_path) or os.path.getsize(jsonl_source_path) == 0:
     raise FileNotFoundError(f"Le fichier source {jsonl_source_path} est manquant ou vide. L'étape d'extraction ('extract_code_tar_gzip_fixed') a échoué.")

# Commande Pyserini
# Utilise la dernière version de Pyserini installée
index_cmd_baseline = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR,
    "--index", INDEX_DIR_BASELINE,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Nombre de threads pour l'indexation
    "--storePositions", "--storeDocvectors", "--storeRaw" # Options de stockage
]

print(f"Exécution: {' '.join(index_cmd_baseline)}")
try:
    # Exécuter la commande d'indexation
    # Augmenter le timeout car cela peut être long
    result = subprocess.run(index_cmd_baseline, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 min
    print("Sortie STDOUT (fin):\n", result.stdout[-1000:]) # Afficher la fin de stdout
    print("Sortie STDERR:\n", result.stderr)
    # Vérifier si 0 document a été indexé (signe de problème)
    if "Total 0 documents indexed" in result.stdout:
         print("\nATTENTION: Pyserini indique 0 document indexé.")
    else:
         print(f"\nIndexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")
except Exception as e:
    # Gérer les erreurs potentielles
    print(f"\nERREUR pendant l'indexation Baseline: {e}")
    if isinstance(e, subprocess.CalledProcessError):
        print("Sortie STDOUT:\n", e.stdout)
        print("Sortie STDERR:\n", e.stderr)
    else:
        traceback.print_exc()
    raise e

# Vérifier la taille de l'index créé
print(f"\nVérification taille index: {INDEX_DIR_BASELINE}...")
if os.path.exists(INDEX_DIR_BASELINE):
    du_cmd = f"du -sh '{INDEX_DIR_BASELINE}'" # Commande pour taille dossier
    try:
        result_du = subprocess.run(du_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  Taille: {result_du.stdout.split()[0]}") # Afficher la taille
    except Exception as e_du:
        print(f"  Impossible de vérifier taille: {e_du}")
else:
    print("  ATTENTION: Dossier index non créé.")

# === Cellule 0: Configuration Complète (avec Stemming) ===
# Installe Java 21, configure comme défaut, installe outils build,
# pybind11, dernière Pyserini, NLTK+ressources, définit chemins,
# FONCTION preprocess_text AVEC STEMMING, parse topics.

import os
import sys
import subprocess
import time
import glob
import re
import json
import nltk # Importer nltk ici pour la partie NLTK
from tqdm.notebook import tqdm # Assurer l'import pour les fonctions
import traceback # Pour afficher les erreurs

print("--- Début de la Configuration Complète (avec Stemming) ---")
print("Cela peut prendre plusieurs minutes...")

# --- Partie 1: Installation Java 21 et Configuration ---
print("\n[1/9] Installation de OpenJDK 21...")
install_java_cmd = "apt-get update -qq > /dev/null && apt-get install -y openjdk-21-jdk-headless -qq > /dev/null"
try: subprocess.run(install_java_cmd, shell=True, check=True, timeout=180); print("OpenJDK 21 installé.")
except Exception as e: print(f"ERREUR installation Java 21: {e}"); raise

print("\n[2/9] Configuration de Java 21 comme défaut via update-alternatives...")
java_path_21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
if os.path.exists(java_path_21):
    try: subprocess.run(f"update-alternatives --install /usr/bin/java java {java_path_21} 1", shell=True, check=True); subprocess.run(f"update-alternatives --set java {java_path_21}", shell=True, check=True); print("update-alternatives configuré.")
    except Exception as e: print(f"ERREUR config update-alternatives: {e}")
else: print(f"ATTENTION: Chemin Java 21 non trouvé: {java_path_21}.")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
if not os.path.exists(os.environ["JAVA_HOME"]): print(f"ATTENTION: Chemin JAVA_HOME inexistant.")

# --- Partie 2: Installation des outils de build C++ ---
print("\n[3/9] Installation des outils de build (build-essential, cmake)...")
install_build_cmd = "apt-get update -qq > /dev/null && apt-get install -y build-essential cmake -qq > /dev/null"
try: subprocess.run(install_build_cmd, shell=True, check=True, timeout=180); print("Outils de build installés.")
except Exception as e_build: print(f"ERREUR installation outils de build: {e_build}")

# --- Partie 3: Installation de pybind11 ---
print("\n[4/9] Installation de pybind11...")
install_pybind_cmd = f"{sys.executable} -m pip install pybind11 -q"
try: subprocess.run(install_pybind_cmd, shell=True, check=True, timeout=60); print("pybind11 installé.")
except Exception as e_pybind: print(f"ERREUR installation pybind11: {e_pybind}")

# --- Partie 4: Installation des Paquets Python Principaux ---
print("\n[5/9] Installation de la DERNIÈRE Pyserini, NLTK, Pytrec_eval...")
install_pip_cmd = f"{sys.executable} -m pip install pyserini nltk pytrec_eval"
try: result_pip = subprocess.run(install_pip_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600); print("Paquets Python principaux installés.")
except Exception as e_pip: print(f"ERREUR installation pip: {e_pip}"); raise

# --- Partie 5: Téléchargement Ressources NLTK ---
print("\n[6/9] Téléchargement/Vérification des ressources NLTK...")
import nltk
nltk_resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4', 'punkt_tab'] # Liste corrigée
for resource in nltk_resources:
    try:
        if resource == 'punkt' or resource == 'punkt_tab': resource_path = f'tokenizers/{resource}.zip'
        elif resource == 'omw-1.4': resource_path = f'corpora/{resource}.zip'
        elif resource == 'wordnet': resource_path = f'corpora/{resource}.zip'
        else: resource_path = f'corpora/{resource}.zip'
        nltk.data.find(resource_path)
    except LookupError:
        print(f"  Ressource NLTK '{resource}' non trouvée. Téléchargement...")
        try: nltk.download(resource, quiet=True); print(f"  Ressource '{resource}' téléchargée.")
        except Exception as e_download: print(f"  ERREUR téléchargement '{resource}': {e_download}")
print("Ressources NLTK prêtes.")

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")
# !!! MODIFIEZ CETTE LIGNE AVEC LE CHEMIN CORRECT VERS VOTRE DOSSIER TREC !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # <--- VÉRIFIEZ ET CORRIGEZ CE CHEMIN !

if 'google.colab' in sys.modules:
    try: from google.colab import drive; drive.mount('/content/drive', force_remount=True); print("  Google Drive monté.")
    except Exception as e_mount: print(f"ATTENTION: Erreur montage Drive: {e_mount}")
if not os.path.exists(DRIVE_PROJECT_PATH): raise FileNotFoundError(f"Chemin Drive '{DRIVE_PROJECT_PATH}' inexistant.")
print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

AP_TAR_FILENAME = "AP.tar"
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, AP_TAR_FILENAME)
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence")
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed") # Sera recréé avec stemming
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(INDEX_DIR_BASELINE, exist_ok=True); os.makedirs(INDEX_DIR_PREPROC, exist_ok=True);
os.makedirs(CORPUS_DIR, exist_ok=True); os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP CIBLE: {AP_TAR_PATH}")
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# --- Partie 7: Définition Fonction Prétraitement (AVEC STEMMING) ---
print("\n[8/9] Définition de la fonction preprocess_text (avec Stemming)...")
import nltk
from nltk.corpus import stopwords
# --- Utilisation de PorterStemmer ---
from nltk.stem import PorterStemmer # Import du stemmer
from nltk.tokenize import word_tokenize
import string
stop_words_set_global = set(stopwords.words('english'))
# --- Création de l'objet Stemmer ---
stemmer_obj_global = PorterStemmer() # Création de l'objet
def preprocess_text(text):
    """Applique tokenisation, minuscules, suppression ponctuation/non-alpha, stop words ET STEMMING (Porter)."""
    if not isinstance(text, str): return ""
    try: tokens = word_tokenize(text.lower())
    except LookupError as e_tok: # Gestion erreur si ressource NLTK manque
         if 'Resource' in str(e_tok) and 'not found' in str(e_tok):
              resource_name = str(e_tok).split('Resource ')[1].split(' ')[0]; print(f"--- Tokenizer a besoin de '{resource_name}', tentative téléchargement ---")
              try: nltk.download(resource_name, quiet=True); print(f"--- Ressource '{resource_name}' téléchargée ---"); tokens = word_tokenize(text.lower())
              except Exception as e_dl_tok: print(f"--- Échec téléchargement '{resource_name}': {e_dl_tok} ---"); raise e_tok
         else: raise e_tok
    except Exception as e_tok_other: print(f"Erreur word_tokenize: {e_tok_other}"); raise e_tok_other
    # --- Application du Stemmer ---
    filtered_tokens = [stemmer_obj_global.stem(w) for w in tokens if w.isalpha() and w not in stop_words_set_global]
    return ' '.join(filtered_tokens)
print("  Fonction preprocess_text définie avec PorterStemmer.")
# Tester la nouvelle fonction
sample_text = "This is an example showing Information Retrieval with stemming and stop words removal."
stemmed_sample = preprocess_text(sample_text)
print(f"  Exemple Stemmed: {stemmed_sample}") # Doit afficher 'thi is exampl show inform retriev with stem and stop word remov.'

# --- Partie 8: Parsing des Topics ---
print("\n[9/9] Parsing des topics...")
import re
import glob
def parse_topics(file_path):
    """Parse un fichier topic TREC standard."""
    topics = {};
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
            topic_content = top_match.group(1)
            num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE); topic_id = num_match.group(1).strip() if num_match else None
            title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL); title = title_match.group(1).strip() if title_match else ""
            desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL); desc = desc_match.group(1).strip() if desc_match else ""
            if topic_id and title: topics[topic_id] = {'title': title, 'desc': desc}
    except Exception as e_topic: print(f"  ATTENTION: Erreur parsing {file_path}: {e_topic}")
    return topics

if not os.path.exists(TOPICS_DIR): print(f"ATTENTION: Dossier topics '{TOPICS_DIR}' inexistant."); topic_files = []
else: topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))
all_topics = {}
if not topic_files: print(f"  ATTENTION: Aucun fichier topic trouvé dans {TOPICS_DIR}")
else: print(f"  Parsing fichiers topics: {topic_files}"); [all_topics.update(parse_topics(tf)) for tf in topic_files]

try:
    queries_short = {qid: data['title'] for qid, data in all_topics.items()}
    queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()}
    print(f"  {len(all_topics)} topics parsés."); print(f"  {len(queries_short)} requêtes courtes brutes créées.")
    print(f"  Prétraitement des requêtes (avec stemming)...")
    # Appliquer la NOUVELLE fonction preprocess_text (avec stemming)
    queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
    queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}
    print(f"  Prétraitement des requêtes terminé.")
except Exception as e_preproc_queries: print(f"\nERREUR prétraitement requêtes: {e_preproc_queries}"); queries_short_preprocessed, queries_long_preprocessed = {}, {}

# --- Vérification Finale Java ---
print("\n--- Vérification Finale Version Java Active ---")
try: result = subprocess.run("java -version", shell=True, check=True, capture_output=True, text=True, timeout=10); print("STDERR:\n", result.stderr); print("\nConfirmation: Java 21 OK." if "21." in result.stderr else "\nATTENTION: Java 21 NON ACTIF ?!")
except Exception as e: print(f"\nERREUR vérification Java: {e}")
# --- Vérification Finale Pyserini ---
print("\n--- Vérification Finale Version Pyserini Installée ---")
try: result_pyserini = subprocess.run(f"{sys.executable} -m pip show pyserini", shell=True, check=True, capture_output=True, text=True, timeout=30); print(result_pyserini.stdout)
except Exception as e: print(f"ERREUR vérification Pyserini: {e}")

print("\n--- Configuration Complète (avec Stemming) Terminée ---")
print("\nPause..."); time.sleep(2); print("Prêt.")

# === Cellule 1: Extraire, Décompresser et Formater les Documents ===
# Lit AP.tar, décompresse les .gz internes, extrait <DOC>, <DOCNO>, <TEXT>
# et écrit le résultat dans ap_docs.jsonl.

import tarfile
import re
import json
import gzip # Importer le module gzip
from tqdm.notebook import tqdm
import os
import traceback

# Vérifier que les chemins sont définis (normalement fait par la cellule de config)
try:
    AP_TAR_PATH
    CORPUS_DIR
except NameError:
    print("ERREUR: Variables de chemin non définies. Exécutez la cellule de configuration complète.")
    raise

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction, Décompression et Formatage depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# Vérifier si le fichier AP.tar existe
if not os.path.exists(AP_TAR_PATH):
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé.")
else:
    tar_size = os.path.getsize(AP_TAR_PATH)
    print(f"  Taille du fichier {AP_TAR_PATH}: {tar_size} octets.") # Devrait être ~275Mo

# Regex pour extraire les infos
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

# Compteurs
doc_count = 0
file_read_count = 0
skipped_members = 0
decompression_errors = 0

try:
    # Ouvrir le fichier de sortie et l'archive TAR
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"\n{len(members)} membres trouvés dans l'archive TAR.") # Devrait être ~1051

        # Boucler sur chaque membre de l'archive
        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            # Ignorer si ce n'est pas un fichier .gz ou .Z
            if not member.isfile() or not member.name.lower().endswith(('.gz', '.z')):
                skipped_members += 1
                continue

            file_read_count += 1
            content = "" # Réinitialiser pour chaque fichier

            try:
                # Extraire le contenu compressé
                f = tar.extractfile(member)
                if f:
                    compressed_content = f.read()
                    f.close()

                    # Décompresser le contenu
                    try:
                        content_bytes = gzip.decompress(compressed_content)
                        content = content_bytes.decode('utf-8', errors='ignore') # Décoder après décompression
                    except gzip.BadGzipFile: # Gérer si ce n'est pas du gzip
                        content = compressed_content.decode('utf-8', errors='ignore')
                        decompression_errors += 1
                    except Exception as e_gzip:
                         print(f"\nErreur de décompression pour {member.name}: {e_gzip}")
                         decompression_errors += 1
                         continue # Passer au suivant

                    # Trouver tous les blocs <DOC> dans le contenu décompressé
                    doc_matches = doc_pattern.findall(content)
                    if not doc_matches: continue # Passer si aucun doc trouvé

                    # Boucler sur chaque document trouvé
                    for doc_content in doc_matches:
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match: continue
                        doc_id = docno_match.group(1).strip()

                        text_match = text_pattern.search(doc_content)
                        # Nettoyer le texte (espaces multiples)
                        doc_text = ' '.join(text_match.group(1).strip().split()) if text_match else ""

                        # Écrire la ligne JSONL
                        try:
                            json_line = json.dumps({"id": str(doc_id), "contents": str(doc_text)})
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key: print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}"); skipped_members += 1
            except EOFError: print(f"\nAvertissement: Fin fichier inattendue {member.name}."); skipped_members += 1
            except Exception as e_extract: print(f"\nErreur extraction/lecture {member.name}: {e_extract}"); skipped_members += 1

except tarfile.ReadError as e_tar: print(f"\nERREUR lecture TAR {AP_TAR_PATH}: {e_tar}"); raise e_tar
except FileNotFoundError: print(f"\nERREUR: Fichier TAR {AP_TAR_PATH} non trouvé."); raise FileNotFoundError
except Exception as e_general: print(f"\nERREUR générale traitement TAR: {e_general}"); traceback.print_exc(); raise e_general

# Afficher le résumé de l'extraction
print(f"\n--- Fin de l'Extraction et Décompression ---")
print(f"  {file_read_count} fichiers (.gz/.Z) lus.")
print(f"  {skipped_members} membres ignorés.")
if decompression_errors > 0: print(f"  {decompression_errors} erreurs/avertissements décompression.")
print(f"  {doc_count} documents écrits dans {JSONL_OUTPUT_PATH}") # Devrait être ~240k

# Vérifier la taille du fichier de sortie
if os.path.exists(JSONL_OUTPUT_PATH):
    output_size = os.path.getsize(JSONL_OUTPUT_PATH)
    print(f"  Taille finale {JSONL_OUTPUT_PATH}: {output_size} octets.") # Devrait être ~600Mo
    if output_size > 0 and doc_count > 0: print("  SUCCÈS: Fichier de sortie contient des données.")
    else: print("  ATTENTION: Fichier de sortie vide ou aucun document écrit.")
else: print(f"  ATTENTION: Fichier {JSONL_OUTPUT_PATH} non créé.")

# === Cellule 2: Indexation Baseline ===
# Crée l'index Lucene à partir de ap_docs.jsonl (sans prétraitement spécifique).

import os
import subprocess
import traceback

# Vérifier que les chemins sont définis (normalement fait par la cellule de config)
try:
    CORPUS_DIR
    INDEX_DIR_BASELINE
except NameError:
    print("ERREUR: Variables de chemin non définies. Exécutez la cellule de configuration complète.")
    raise

print(f"Début de l'indexation Baseline...")
print(f"Dossier source: {CORPUS_DIR}")
print(f"Répertoire de l'index cible: {INDEX_DIR_BASELINE}")

# Vérifier si le fichier source existe et n'est pas vide
jsonl_source_path = os.path.join(CORPUS_DIR, "ap_docs.jsonl")
if not os.path.exists(jsonl_source_path) or os.path.getsize(jsonl_source_path) == 0:
     raise FileNotFoundError(f"Le fichier source {jsonl_source_path} est manquant ou vide. L'étape d'extraction a échoué.")

# Commande Pyserini
# Utilise la dernière version de Pyserini installée
index_cmd_baseline = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR,
    "--index", INDEX_DIR_BASELINE,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Nombre de threads pour l'indexation
    "--storePositions", "--storeDocvectors", "--storeRaw" # Options de stockage
]

print(f"Exécution: {' '.join(index_cmd_baseline)}")
try:
    # Exécuter la commande d'indexation
    # Augmentation possible du timeout si l'indexation est très longue
    result = subprocess.run(index_cmd_baseline, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 min
    print("Sortie STDOUT (fin):\n", result.stdout[-1000:]) # Afficher la fin de stdout
    print("Sortie STDERR:\n", result.stderr)
    # Vérifier si 0 document a été indexé (signe de problème)
    if "Total 0 documents indexed" in result.stdout:
         print("\nATTENTION: Pyserini indique 0 document indexé.")
    else:
         print(f"\nIndexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")
except Exception as e:
    # Gérer les erreurs potentielles
    print(f"\nERREUR pendant l'indexation Baseline: {e}")
    if isinstance(e, subprocess.CalledProcessError):
        print("Sortie STDOUT:\n", e.stdout)
        print("Sortie STDERR:\n", e.stderr)
    else:
        traceback.print_exc()
    raise e

# Vérifier la taille de l'index créé
print(f"\nVérification taille index: {INDEX_DIR_BASELINE}...")
if os.path.exists(INDEX_DIR_BASELINE):
    du_cmd = f"du -sh '{INDEX_DIR_BASELINE}'" # Commande pour taille dossier
    try:
        result_du = subprocess.run(du_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  Taille: {result_du.stdout.split()[0]}") # Afficher la taille
    except Exception as e_du:
        print(f"  Impossible de vérifier taille: {e_du}")
else:
    print("  ATTENTION: Dossier index non créé.")

# === Cellule de Vérification du Contenu du Dossier Runs ===
# Utilise les commandes shell de Colab préfixées par '!'

# Chemin exact où les résultats de recherche sont attendus
# (Défini dans la cellule de configuration complète)
RUN_DIR_PATH="/content/ap_output/runs/"

# Utiliser '!' pour exécuter la commande shell 'echo'
print(f"Vérification du contenu de : {RUN_DIR_PATH}")

# Utiliser '!' pour exécuter la commande shell 'ls -l'
# Mettre le chemin entre guillemets pour gérer les espaces potentiels (même s'il n'y en a pas ici)
!ls -lh "{RUN_DIR_PATH}"

# === Monter Google Drive ===
    from google.colab import drive
    import os

    try:
        print("Tentative de montage de Google Drive...")
        drive.mount('/content/drive', force_remount=True) # force_remount=True est utile en cas de problème antérieur

        # Vérifier si le point de montage de base existe après la tentative
        if os.path.exists('/content/drive/My Drive'):
            print("\nGoogle Drive monté avec succès sur /content/drive !")
        else:
            print("\nATTENTION: Le montage semble avoir échoué (vérifiez les messages ci-dessus et la fenêtre d'autorisation).")

    except Exception as e:
        print(f"\nUne erreur s'est produite lors du montage de Drive: {e}")

from multiprocessing import Pool
from google.colab import drive
import os

drive.mount("/content/drive")

# Create the directory if it doesn't exist
target_dir = "/content/drive/MyDrive/Projet_RI"  # Changed 'myDrive' to 'MyDrive'
if not os.path.exists(target_dir):
    try:
        os.makedirs(target_dir, exist_ok=True)  # Use exist_ok to avoid error if directory exists
        print(f"Directory '{target_dir}' created.")
    except FileExistsError:
        print(f"Directory '{target_dir}' already exists.")
else:
    print(f"Directory '{target_dir}' already exists.")

os.chdir(target_dir)

def process_file(file):
    # Votre code de prétraitement ici
    # Example: Assuming you want to read the file and return its content
    file_path = os.path.join("AP_Final", file) # Construct the full file path
    # Specify the encoding when opening the file
    with open(file_path, 'r', encoding='latin-1') as f:  # Try 'latin-1' or 'cp1252'
        preprocessed_text = f.read()  # Assign a value to preprocessed_text
    return preprocessed_text

if __name__ == "__main__":
    files = os.listdir("AP_Final")
    with Pool(os.cpu_count()) as p:  # Utilise tous les cœurs
        results = p.map(process_file, files)

# === Cellule 3.1 (Modifiée): Fonction de Recherche et Sauvegarde (Séquentielle d'abord) ===
from pyserini.search.lucene import LuceneSearcher
import time
from tqdm.notebook import tqdm # Toujours utile pour la progression
import traceback # Pour afficher les erreurs détaillées

# --- Configuration des modèles de similarité ---
from jnius import autoclass, JavaException
ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')

def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}" # Ex: baseline_short_bm25
    print(f"Début recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', PréprocReq={use_preprocessed_query}, k={k}")

    all_results_list = []
    searcher = None # Initialiser à None

    try:
        # Initialiser le searcher UNE SEULE FOIS pour toutes les requêtes de ce run
        print(f"  Initialisation de LuceneSearcher pour {run_tag}...")
        searcher = LuceneSearcher(index_path)
        print(f"  LuceneSearcher initialisé.")

        # Configurer le modèle de similarité
        if model == 'bm25':
            print("  Configuration de BM25...")
            searcher.set_bm25(k1=0.9, b=0.4)
            print("  BM25 configuré.")
        elif model == 'tfidf':
            print("  Configuration de ClassicSimilarity (TF-IDF)...")
            try:
                 searcher.set_similarity(ClassicSimilarity())
                 print("  ClassicSimilarity configurée.")
            except JavaException as e:
                 print(f"ERREUR Java lors de la configuration de ClassicSimilarity: {e}")
                 print(traceback.format_exc()) # Affiche la trace complète de l'erreur Java
                 raise # Arrête l'exécution pour ce run si la similarité ne peut être définie
        else:
            print("  Configuration BM25 par défaut...")
            searcher.set_bm25()
            print("  BM25 par défaut configuré.")

        # Itérer sur les requêtes séquentiellement
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                hits = searcher.search(search_text, k=k)

                # Formater les résultats pour cette requête
                for i in range(len(hits)):
                    rank = i + 1
                    doc_id = hits[i].docid
                    score = hits[i].score
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            except Exception as e_query:
                print(f"\nErreur lors de la recherche pour QID {query_id} avec {run_tag}: {e_query}")
                # Continue avec la requête suivante

        # Écrire les résultats dans le fichier de run TREC
        with open(output_run_file, 'w') as f_out:
           f_out.writelines(all_results_list)

        end_time = time.time()
        print(f"Recherche SÉQUENTIELLE terminée pour {run_tag}. Résultats sauvegardés dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.\n")

    except Exception as e_main:
        print(f"\nERREUR MAJEURE pendant l'exécution de {run_tag}: {e_main}")
        print(traceback.format_exc()) # Affiche la trace complète de l'erreur
    finally:
        # Important: Fermer le searcher pour libérer les ressources Java, même en cas d'erreur
        if searcher:
             try:
                 # Note: Pyserini ne semble pas avoir de méthode close() explicite sur LuceneSearcher
                 # La JVM devrait se nettoyer, mais c'est une bonne pratique si disponible
                 # searcher.close() # Décommentez si une telle méthode existe dans votre version
                 print(f"  Nettoyage implicite des ressources pour {run_tag}.")
                 pass
             except Exception as e_close:
                 print(f"  Erreur lors de la tentative de fermeture du searcher pour {run_tag}: {e_close}")


# --- Exécution des différentes configurations (en mode séquentiel) ---
K_RESULTS = 1000 # Nombre de documents à retourner par requête

# 1. Index Baseline + Requêtes Courtes + BM25
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")

# 2. Index Baseline + Requêtes Courtes + TF-IDF
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")

# 3. Index Baseline + Requêtes Longues + BM25
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")

# 4. Index Baseline + Requêtes Longues + TF-IDF
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")

# --- Recherches sur l'index prétraité ---
# 5. Index Preprocessed + Requêtes Courtes (Prétraitées) + BM25
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)

# 6. Index Preprocessed + Requêtes Courtes (Prétraitées) + TF-IDF
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)

# 7. Index Preprocessed + Requêtes Longues (Prétraitées) + BM25
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)

# 8. Index Preprocessed + Requêtes Longues (Prétraitées) + TF-IDF
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)

print("Toutes les recherches de base (mode séquentiel) sont terminées.")

# --- Note importante ---
# Si cette cellule s'exécute sans planter (même si c'est lent),
# le problème est probablement lié à la parallélisation (mémoire/conflits JVM).
# Si elle plante encore, surtout lors des runs 'tfidf',
# le problème pourrait être lié à ClassicSimilarity ou à l'environnement Java lui-même.

# === Cellule 3.1 (Modifiée): Fonction de Recherche et Sauvegarde (BM25 Séquentiel Uniquement) ===
from pyserini.search.lucene import LuceneSearcher
import time
from tqdm.notebook import tqdm # Toujours utile pour la progression
import traceback # Pour afficher les erreurs détaillées

# --- Configuration des modèles de similarité ---
# On importe toujours ClassicSimilarity au cas où, mais on ne l'utilisera pas dans ce test
from jnius import autoclass, JavaException
ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')

def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}" # Ex: baseline_short_bm25
    print(f"Début recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', PréprocReq={use_preprocessed_query}, k={k}")

    # --- Vérification ajoutée : Ne traiter que BM25 pour ce test ---
    if model != 'bm25':
        print(f"--- Run '{run_tag}' ignoré (Test BM25 uniquement) ---")
        return # Ne rien faire si ce n'est pas BM25

    all_results_list = []
    searcher = None # Initialiser à None

    try:
        # Initialiser le searcher UNE SEULE FOIS pour toutes les requêtes de ce run
        print(f"  Initialisation de LuceneSearcher pour {run_tag}...")
        searcher = LuceneSearcher(index_path)
        print(f"  LuceneSearcher initialisé.")

        # Configurer le modèle de similarité (seulement BM25 ici)
        print("  Configuration de BM25...")
        searcher.set_bm25(k1=0.9, b=0.4)
        print("  BM25 configuré.")

        # Itérer sur les requêtes séquentiellement
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                hits = searcher.search(search_text, k=k)

                # Formater les résultats pour cette requête
                for i in range(len(hits)):
                    rank = i + 1
                    doc_id = hits[i].docid
                    score = hits[i].score
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            except Exception as e_query:
                print(f"\nErreur lors de la recherche pour QID {query_id} avec {run_tag}: {e_query}")
                # Continue avec la requête suivante

        # Écrire les résultats dans le fichier de run TREC
        with open(output_run_file, 'w') as f_out:
           f_out.writelines(all_results_list)

        end_time = time.time()
        print(f"Recherche SÉQUENTIELLE terminée pour {run_tag}. Résultats sauvegardés dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.\n")

    except Exception as e_main:
        print(f"\nERREUR MAJEURE pendant l'exécution de {run_tag}: {e_main}")
        print(traceback.format_exc()) # Affiche la trace complète de l'erreur
    finally:
        if searcher:
             try:
                 print(f"  Nettoyage implicite des ressources pour {run_tag}.")
                 pass
             except Exception as e_close:
                 print(f"  Erreur lors de la tentative de fermeture du searcher pour {run_tag}: {e_close}")


# --- Exécution des différentes configurations (BM25 seulement) ---
K_RESULTS = 1000 # Nombre de documents à retourner par requête

# 1. Index Baseline + Requêtes Courtes + BM25
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")

# 2. Index Baseline + Requêtes Courtes + TF-IDF (IGNORÉ DANS CETTE VERSION)
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")

# 3. Index Baseline + Requêtes Longues + BM25
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")

# 4. Index Baseline + Requêtes Longues + TF-IDF (IGNORÉ DANS CETTE VERSION)
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")

# --- Recherches sur l'index prétraité ---
# 5. Index Preprocessed + Requêtes Courtes (Prétraitées) + BM25
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)

# 6. Index Preprocessed + Requêtes Courtes (Prétraitées) + TF-IDF (IGNORÉ DANS CETTE VERSION)
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)

# 7. Index Preprocessed + Requêtes Longues (Prétraitées) + BM25
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)

# 8. Index Preprocessed + Requêtes Longues (Prétraitées) + TF-IDF (IGNORÉ DANS CETTE VERSION)
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)

print("Toutes les recherches de base (mode séquentiel, BM25 uniquement) sont terminées.")

# --- Note importante ---
# Si cette cellule s'exécute sans planter, le problème est très probablement lié
# à l'utilisation de ClassicSimilarity (TF-IDF) dans l'environnement Java actuel.
# Si elle plante encore, le problème est plus profond avec l'initialisation de LuceneSearcher.

# === Cellule 3.1 (Modifiée): Fonction de Recherche et Sauvegarde (Séquentielle - BM25 & TF-IDF) ===
# Utilise Pyserini 0.23.0
from pyserini.search.lucene import LuceneSearcher
import time
from tqdm.notebook import tqdm # Toujours utile pour la progression
import traceback # Pour afficher les erreurs détaillées

# --- Configuration des modèles de similarité ---
from jnius import autoclass, JavaException
ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')

def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}" # Ex: baseline_short_bm25 or baseline_short_tfidf
    print(f"Début recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', PréprocReq={use_preprocessed_query}, k={k}")

    all_results_list = []
    searcher = None # Initialiser à None

    try:
        # Initialiser le searcher UNE SEULE FOIS pour toutes les requêtes de ce run
        print(f"  Initialisation de LuceneSearcher pour {run_tag}...")
        searcher = LuceneSearcher(index_path)
        print(f"  LuceneSearcher initialisé.")

        # Configurer le modèle de similarité
        if model == 'bm25':
            print("  Configuration de BM25...")
            searcher.set_bm25(k1=0.9, b=0.4) # Utilise les paramètres BM25 par défaut de Pyserini
            print("  BM25 configuré.")
        elif model == 'tfidf':
            print("  Configuration de ClassicSimilarity (TF-IDF)...")
            try:
                 # Tentative de configuration de ClassicSimilarity
                 searcher.set_similarity(ClassicSimilarity())
                 print("  ClassicSimilarity configurée.")
            except JavaException as e:
                 print(f"ERREUR Java lors de la configuration de ClassicSimilarity: {e}")
                 print(traceback.format_exc()) # Affiche la trace complète de l'erreur Java
                 print(f"--- ABANDON du run {run_tag} à cause de l'erreur de configuration TF-IDF ---")
                 return # Arrête l'exécution pour ce run spécifique si TF-IDF échoue
            except Exception as e_other:
                 print(f"ERREUR Inattendue lors de la configuration de ClassicSimilarity: {e_other}")
                 print(traceback.format_exc())
                 print(f"--- ABANDON du run {run_tag} à cause d'une erreur TF-IDF ---")
                 return # Arrête l'exécution pour ce run spécifique
        else:
            # Sécurité : si le modèle n'est ni bm25 ni tfidf, utilise bm25 par défaut
            print(f"Modèle '{model}' non reconnu, utilisation de BM25 par défaut...")
            searcher.set_bm25()
            print("  BM25 par défaut configuré.")

        # Itérer sur les requêtes séquentiellement
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                hits = searcher.search(search_text, k=k)

                # Formater les résultats pour cette requête
                for i in range(len(hits)):
                    rank = i + 1
                    doc_id = hits[i].docid
                    score = hits[i].score
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            except Exception as e_query:
                print(f"\nErreur lors de la recherche pour QID {query_id} avec {run_tag}: {e_query}")
                # Continue avec la requête suivante même si une échoue

        # Écrire les résultats dans le fichier de run TREC (seulement si aucune erreur majeure n'est survenue avant la boucle)
        with open(output_run_file, 'w') as f_out:
           f_out.writelines(all_results_list)

        end_time = time.time()
        print(f"Recherche SÉQUENTIELLE terminée pour {run_tag}. Résultats sauvegardés dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.\n")

    except Exception as e_main:
        # Erreur pendant l'initialisation du searcher ou configuration BM25 (peu probable maintenant)
        print(f"\nERREUR MAJEURE pendant l'exécution de {run_tag}: {e_main}")
        print(traceback.format_exc()) # Affiche la trace complète de l'erreur
    finally:
        # Nettoyage implicite (Pyserini gère la fermeture de la JVM)
        if searcher:
             print(f"  Nettoyage implicite des ressources pour {run_tag}.")
             pass


# --- Exécution des différentes configurations (Séquentiel - BM25 & TF-IDF) ---
K_RESULTS = 1000 # Nombre de documents à retourner par requête

# 1. Index Baseline + Requêtes Courtes + BM25
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")

# 2. Index Baseline + Requêtes Courtes + TF-IDF
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")

# 3. Index Baseline + Requêtes Longues + BM25
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")

# 4. Index Baseline + Requêtes Longues + TF-IDF
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")

# --- Recherches sur l'index prétraité ---
# 5. Index Preprocessed + Requêtes Courtes (Prétraitées) + BM25
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)

# 6. Index Preprocessed + Requêtes Courtes (Prétraitées) + TF-IDF
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)

# 7. Index Preprocessed + Requêtes Longues (Prétraitées) + BM25
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)

# 8. Index Preprocessed + Requêtes Longues (Prétraitées) + TF-IDF
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)

print("Toutes les recherches de base (mode séquentiel - BM25 & TF-IDF tentative) sont terminées.")

# --- Note importante ---
# Surveillez la sortie lors de l'exécution des runs 'tfidf'.
# Si vous voyez des erreurs Java ou si le kernel plante à nouveau,
# cela signifie que ClassicSimilarity est toujours problématique.

# === Cellule 0.3: Définir les chemins ===
import os # Assurez-vous que os est importé

# !!! ADAPTEZ CE CHEMIN VERS VOTRE DOSSIER SUR GOOGLE DRIVE SI NÉCESSAIRE !!!
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Projet_RI/TREC/"

# Vérification que le chemin existe
if not os.path.exists(DRIVE_PROJECT_PATH):
    raise FileNotFoundError(f"Le chemin spécifié n'existe pas : {DRIVE_PROJECT_PATH}. Vérifiez le chemin.")

AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, "AP.tar") # Assumant que c'est un .tar.gz, sinon ajustez
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence") # Définit QRELS_DIR

# Chemins pour les sorties (index, résultats, etc.) dans l'environnement Colab
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus") # Pour les documents extraits/formatés
RUN_DIR = os.path.join(OUTPUT_DIR, "runs") # Définit RUN_DIR
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval") # Définit EVAL_DIR

# Créer les répertoires de sortie s'ils n'existent pas déjà
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"Chemin du projet Drive: {DRIVE_PROJECT_PATH}")
print(f"Répertoire de sortie Colab: {OUTPUT_DIR}")
print(f"Chemin Qrels: {QRELS_DIR}") # Vérifie que QRELS_DIR est défini
print(f"Chemin Runs: {RUN_DIR}")
print(f"Chemin Eval: {EVAL_DIR}")

# === Cellule 4.1 & 4.2: Préparation Qrels et Évaluation des Runs ===
import pandas as pd
import glob
import pytrec_eval
import os # Assurez-vous que os est importé
import traceback # Pour afficher les erreurs détaillées

# --- 4.1: Préparer le Fichier Qrels Combiné ---

# Chemins définis précédemment dans la Cellule 0.3 (qui vient d'être exécutée avec succès)
# QRELS_DIR, RUN_DIR, EVAL_DIR devraient être définis

print(f"Préparation des Qrels depuis: {QRELS_DIR}")
qrels_files = sorted(glob.glob(os.path.join(QRELS_DIR, "qrels.*.txt")))
if not qrels_files:
    print(f"ATTENTION: Aucun fichier Qrels trouvé dans {QRELS_DIR}. Vérifiez le chemin.")
else:
    print(f"Fichiers Qrels trouvés: {qrels_files}")

all_qrels_data = []
for qf in qrels_files:
    try:
        # Lire le fichier qrels: query_id unused doc_id relevance
        # S'assurer que les IDs sont lus comme des chaînes de caractères
        qrels_df = pd.read_csv(qf, sep='\s+', names=['query_id', 'unused', 'doc_id', 'relevance'],
                               dtype={'query_id': str, 'unused': str, 'doc_id': str, 'relevance': int})
        all_qrels_data.append(qrels_df[['query_id', 'doc_id', 'relevance']]) # Garder seulement les colonnes utiles
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier Qrels {qf}: {e}")


if not all_qrels_data:
     print("ERREUR: Impossible de lire les données Qrels. Vérifiez les fichiers et les chemins.")
     # Arrêter ici si les qrels ne peuvent pas être chargés
     raise ValueError("Données Qrels non chargées.")
else:
    combined_qrels_df = pd.concat(all_qrels_data, ignore_index=True)

    # Convertir en dictionnaire format pytrec_eval: {query_id: {doc_id: relevance}}
    qrels_dict = {}
    for _, row in combined_qrels_df.iterrows():
        qid = row['query_id']
        did = row['doc_id']
        # Assurer que la pertinence est bien un entier
        try:
            rel = int(row['relevance'])
        except ValueError:
            print(f"Avertissement: Valeur de pertinence non entière ignorée pour qid={qid}, did={did}: {row['relevance']}")
            continue

        # Filtrer les jugements non binaires si nécessaire (garder 0 et 1, ou > 0 pour pertinent)
        if rel < 0: # Ignorer les jugements négatifs si présents
             continue

        if qid not in qrels_dict:
            qrels_dict[qid] = {}
        # Stocker la pertinence (pytrec_eval gère différents niveaux, mais ici 0=non pertinent, >0=pertinent)
        qrels_dict[qid][did] = rel

    print(f"Total de {len(qrels_dict)} requêtes avec jugements dans le fichier Qrels combiné.")
    qrels_doc_count = sum(len(docs) for docs in qrels_dict.values())
    print(f"Nombre total de jugements pertinents/non pertinents chargés: {qrels_doc_count}")


    # --- 4.2: Évaluation des Runs ---

    # Mesures à calculer (standard TREC)
    measures = {'map', 'P_10'} # MAP (mean average precision), Precision at 10

    # Initialiser l'évaluateur avec les qrels et les mesures
    # Utiliser seulement les query_ids présents dans les qrels pour l'évaluation
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, measures)

    # Trouver tous les fichiers de run générés dans RUN_DIR
    run_files = sorted(glob.glob(os.path.join(RUN_DIR, "*.txt")))
    print(f"\nFichiers de run à évaluer trouvés dans {RUN_DIR}: {len(run_files)}")
    # print(run_files) # Décommentez pour voir la liste exacte

    results_summary = [] # Pour stocker les résultats pour le tableau final

    if not run_files:
        print(f"ATTENTION: Aucun fichier de run (.txt) trouvé dans {RUN_DIR}. Vérifiez que l'étape 3 a bien généré des fichiers.")
    else:
        for run_file in run_files:
            run_name = os.path.basename(run_file)
            print(f"\n--- Évaluation de: {run_name} ---")

            # Charger le fichier de run au format TREC
            # pytrec_eval attend un dictionnaire: {query_id: {doc_id: score}}
            run_dict = {}
            line_count = 0
            error_count = 0
            try:
                with open(run_file, 'r') as f_run:
                    for line in f_run:
                        line_count += 1
                        parts = line.strip().split()
                        if len(parts) != 6:
                             # print(f"Ligne mal formatée ignorée dans {run_name} (ligne {line_count}): {line.strip()}")
                             error_count += 1
                             continue
                        qid, _, did, rank, score, _ = parts
                        # Assurer que l'ID de requête est une chaîne, comme dans qrels_dict
                        qid = str(qid)
                        # Assurer que le score est un float
                        try:
                            score = float(score)
                        except ValueError:
                            # print(f"Score non flottant ignoré dans {run_name} (ligne {line_count}): {score}")
                            error_count += 1
                            continue

                        if qid not in run_dict:
                            run_dict[qid] = {}
                        run_dict[qid][did] = score

                if error_count > 0:
                    print(f"  Avertissement: {error_count} lignes mal formatées ignorées dans {run_name}.")

                # Filtrer le run_dict pour ne garder que les query_ids présents dans qrels_dict
                filtered_run_dict = {qid: docs for qid, docs in run_dict.items() if qid in qrels_dict}
                ignored_queries = len(run_dict) - len(filtered_run_dict)
                if ignored_queries > 0:
                    print(f"  Avertissement: {ignored_queries} requêtes du run ignorées car absentes des Qrels.")

                if not filtered_run_dict:
                     print("  Erreur: Aucune requête du run ne correspond aux Qrels. Impossible d'évaluer.")
                     continue

                # Effectuer l'évaluation sur les données filtrées
                eval_results = evaluator.evaluate(filtered_run_dict)

                # Calculer les moyennes sur toutes les requêtes évaluées
                # Gérer le cas où une métrique pourrait manquer pour une requête (peu probable avec MAP, P@10)
                all_maps = [q_res.get("map", 0) for q_res in eval_results.values()]
                all_p10s = [q_res.get("P_10", 0) for q_res in eval_results.values()]

                # Éviter la division par zéro si aucune requête n'a pu être évaluée
                avg_map = sum(all_maps) / len(all_maps) if all_maps else 0
                avg_p10 = sum(all_p10s) / len(all_p10s) if all_p10s else 0

                print(f"  MAP: {avg_map:.4f}")
                print(f"  P@10: {avg_p10:.4f}")
                print("-" * (20 + len(run_name)))

                # Extraire les informations pour le tableau récapitulatif
                # Gère les noms de fichiers comme 'baseline_short_bm25.txt' ou 'preproc_long_tfidf.txt'
                parts = run_name.replace('.txt','').split('_')
                if len(parts) >= 3:
                    index_type = parts[0] # baseline ou preproc
                    query_type = parts[1] # short ou long
                    model_type = parts[2] # bm25 ou tfidf
                    # Gérer le cas RM3 si on l'ajoute plus tard
                    if len(parts) > 3 and parts[3] == 'rm3':
                         model_type += "+RM3"

                    results_summary.append({
                        "Run Name": run_name,
                        "Index": index_type,
                        "Query Type": query_type.capitalize(), # Met la première lettre en majuscule
                        "Weighting Scheme": model_type.upper(), # Met en majuscules (BM25, TFIDF)
                        "MAP": avg_map,
                        "P@10": avg_p10
                    })
                else:
                     print(f"  Avertissement: Impossible de parser le nom du run '{run_name}' pour le résumé.")

            except FileNotFoundError:
                 print(f"  Erreur: Fichier run non trouvé: {run_file}")
            except Exception as e:
                 print(f"  Erreur lors de l'évaluation de {run_name}: {e}")
                 print(traceback.format_exc())

        # Afficher le tableau récapitulatif si des résultats ont été collectés
        if results_summary:
            print("\n\n=== Tableau Récapitulatif des Résultats (Partie 1) ===")
            results_df = pd.DataFrame(results_summary)

            # Pivoter pour obtenir le format demandé (plus ou moins)
            try:
                pivot_map = results_df.pivot_table(index=['Query Type', 'Weighting Scheme'], columns='Index', values='MAP')
                print("\n--- MAP (Moyenne des Précisions Moyennes) ---")
                print(pivot_map.to_markdown(floatfmt=".4f"))
            except Exception as e_pivot_map:
                 print(f"\nErreur lors de la création du tableau pivot MAP: {e_pivot_map}")
                 print("Affichage du DataFrame brut MAP:")
                 print(results_df[['Query Type', 'Weighting Scheme', 'Index', 'MAP']].to_markdown(index=False, floatfmt=".4f"))


            try:
                pivot_p10 = results_df.pivot_table(index=['Query Type', 'Weighting Scheme'], columns='Index', values='P@10')
                print("\n--- P@10 (Précision aux 10 premiers documents) ---")
                print(pivot_p10.to_markdown(floatfmt=".4f"))
            except Exception as e_pivot_p10:
                 print(f"\nErreur lors de la création du tableau pivot P@10: {e_pivot_p10}")
                 print("Affichage du DataFrame brut P@10:")
                 print(results_df[['Query Type', 'Weighting Scheme', 'Index', 'P@10']].to_markdown(index=False, floatfmt=".4f"))


            # Sauvegarder le DataFrame pour une utilisation ultérieure (ex: rapport)
            summary_file_path = os.path.join(EVAL_DIR, "evaluation_summary_part1.csv")
            try:
                 results_df.to_csv(summary_file_path, index=False)
                 print(f"\nTableau récapitulatif sauvegardé dans {summary_file_path}")
            except Exception as e_save:
                 print(f"\nErreur lors de la sauvegarde du résumé dans {summary_file_path}: {e_save}")

        else:
            print("\nAucun résultat d'évaluation à afficher ou sauvegarder.")

# === Cellule de Vérification Java (à exécuter JUSTE AVANT la Cellule 5.1 / rm3_run_code) ===
    # Ceci vérifie quelle version de Java le kernel Python voit ACTUELLEMENT
    print("--- Vérification de la version Java vue par le kernel ACTUEL ---")
    !java -version
    print("-------------------------------------------------------------")

# === Cellule de Configuration Complète (Chemins Corrigés) ===
import os
import sys
import subprocess
import time

print("--- Début de la Configuration Complète ---")

# --- Partie 1: Installation Java 21 et Configuration ---
print("\n[1/9] Installation de OpenJDK 21...")
install_java_cmd = "apt-get update -qq > /dev/null && apt-get install -y openjdk-21-jdk-headless -qq > /dev/null"
subprocess.run(install_java_cmd, shell=True, check=True)
print("OpenJDK 21 installé.")

print("\n[2/9] Configuration de Java 21 comme défaut via update-alternatives...")
java_path_21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
if os.path.exists(java_path_21):
    subprocess.run(f"update-alternatives --install /usr/bin/java java {java_path_21} 1", shell=True, check=True)
    subprocess.run(f"update-alternatives --set java {java_path_21}", shell=True, check=True)
    print("update-alternatives configuré pour java.")
else:
    print(f"ATTENTION: Chemin Java 21 non trouvé à {java_path_21}. update-alternatives non configuré.")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
if not os.path.exists(os.environ["JAVA_HOME"]):
     print(f"ATTENTION: Le chemin JAVA_HOME '{os.environ['JAVA_HOME']}' n'existe pas.")

# --- Partie 2: Installation des outils de build C++ ---
print("\n[3/9] Installation des outils de build (build-essential, cmake)...")
install_build_cmd = "apt-get update -qq > /dev/null && apt-get install -y build-essential cmake -qq > /dev/null"
try:
    subprocess.run(install_build_cmd, shell=True, check=True, timeout=180)
    print("Outils de build installés.")
except Exception as e_build:
    print(f"ERREUR lors de l'installation des outils de build: {e_build}")

# --- Partie 3: Installation de pybind11 ---
print("\n[4/9] Installation de pybind11...")
install_pybind_cmd = f"{sys.executable} -m pip install pybind11"
try:
    subprocess.run(install_pybind_cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
    print("pybind11 installé avec succès.")
except Exception as e_pybind:
    print(f"ERREUR lors de l'installation de pybind11: {e_pybind}")

# --- Partie 4: Installation des Paquets Python Principaux ---
print("\n[5/9] Installation de la DERNIÈRE Pyserini, NLTK, Pytrec_eval...")
install_pip_cmd = f"{sys.executable} -m pip install pyserini nltk pytrec_eval"
try:
    result_pip = subprocess.run(install_pip_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600)
    print("Paquets Python principaux installés.")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: La commande pip principale a échoué avec le code {e.returncode}")
    print("Sortie STDOUT de Pip:\n", e.stdout)
    print("Sortie STDERR de Pip:\n", e.stderr)
    raise e
except subprocess.TimeoutExpired as e:
    print("\nERREUR: La commande pip principale a dépassé le délai d'attente.")
    print("Sortie STDOUT de Pip (partielle):\n", e.stdout)
    print("Sortie STDERR de Pip (partielle):\n", e.stderr)
    raise e

# --- Partie 5: Téléchargement Ressources NLTK ---
print("\n[6/9] Téléchargement/Vérification des ressources NLTK...")
import nltk
nltk_resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}.zip') if resource != 'punkt' else nltk.data.find(f'tokenizers/{resource}.zip')
    except nltk.downloader.DownloadError:
        print(f"  Téléchargement de la ressource NLTK '{resource}'...")
        nltk.download(resource, quiet=True)
print("Ressources NLTK prêtes.")

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")

# !!! CHEMIN CORRIGÉ SELON VOS INDICATIONS !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # Chemin vers le sous-dossier TREC

# --- Le reste du code vérifie le chemin et définit les autres variables ---
if not os.path.exists(DRIVE_PROJECT_PATH):
    try:
        from google.colab import drive
        print("  Montage de Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        if not os.path.exists(DRIVE_PROJECT_PATH):
             raise FileNotFoundError(f"Le chemin Drive '{DRIVE_PROJECT_PATH}' n'existe TOUJOURS PAS après montage. Vérifiez le chemin exact et le nom des dossiers.")
    except ModuleNotFoundError:
         raise FileNotFoundError(f"Google Colab Drive non trouvé et chemin '{DRIVE_PROJECT_PATH}' inexistant.")
    except Exception as e_mount:
         raise FileNotFoundError(f"Erreur lors du montage de Drive ou chemin '{DRIVE_PROJECT_PATH}' inexistant: {e_mount}")

print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

# !!! NOM DE FICHIER CORRIGÉ SELON VOS INDICATIONS !!!
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, "AP.tar") # Utilise AP.tar au lieu de AP.tar.gz
# Note: Pensez à modifier la Cellule 0.4 (extraction) pour ouvrir avec "r:" au lieu de "r:gz"

TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence")
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP: {AP_TAR_PATH}")
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# --- Partie 7: Définition Fonction Prétraitement ---
print("\n[8/9] Définition de la fonction preprocess_text...")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    if not isinstance(text, str): return ""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(filtered_tokens)
print("  Fonction preprocess_text définie.")


# --- Partie 8: Parsing des Topics ---
print("\n[9/9] Parsing des topics...")
import glob
import re
def parse_topics(file_path):
    topics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
                topic_content = top_match.group(1)
                num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE)
                if not num_match: continue
                topic_id = num_match.group(1).strip()
                title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                desc = desc_match.group(1).strip() if desc_match else ""
                if topic_id and title:
                     topics[topic_id] = {'title': title, 'desc': desc}
    except FileNotFoundError:
        print(f"  ATTENTION: Fichier topic non trouvé: {file_path}")
    except Exception as e_topic:
        print(f"  ATTENTION: Erreur lors du parsing de {file_path}: {e_topic}")
    return topics
topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))
all_topics = {}
if not topic_files:
     print(f"  ATTENTION: Aucun fichier topic trouvé dans {TOPICS_DIR}")
else:
    for tf in topic_files:
        all_topics.update(parse_topics(tf))
queries_short = {qid: data['title'] for qid, data in all_topics.items()}
queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()}
queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}
print(f"  {len(all_topics)} topics parsés.")
print(f"  {len(queries_short)} requêtes courtes créées.")


# --- Vérification Finale Java ---
print("\n--- Vérification Finale de la Version Java Active ---")
java_check_cmd = "java -version"
try:
    result = subprocess.run(java_check_cmd, shell=True, check=True, capture_output=True, text=True, timeout=10)
    print("Sortie STDERR:\n", result.stderr) # Version souvent sur stderr
    if "21." not in result.stderr and "21." not in result.stdout:
         print("\nATTENTION: Java 21 ne semble PAS être la version active !")
    else:
         print("\nConfirmation: Java 21 semble être la version active.")
except Exception as e:
    print(f"\nERREUR lors de la vérification Java: {e}")


# --- Vérification Finale Pyserini ---
print("\n--- Vérification Finale de la Version Pyserini Installée ---")
try:
    result_pyserini = subprocess.run(f"{sys.executable} -m pip show pyserini", shell=True, check=True, capture_output=True, text=True, timeout=30)
    print(result_pyserini.stdout)
except ImportError:
    print("ERREUR: Impossible d'importer Pyserini après l'installation.")
except Exception as e_pyserini_check:
    print(f"ERREUR lors de la vérification de Pyserini: {e_pyserini_check}")


print("\n--- Configuration Complète Terminée ---")

# === Cellule de Configuration Complète (Chemins Corrigés) ===
# ... (début de la cellule inchangé : installation Java, build tools, pip, etc.) ...

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")

# !!! VÉRIFIEZ CE CHEMIN VERS LE DOSSIER CONTENANT AP.tar !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # Est-ce le bon dossier ?

# --- Le reste du code vérifie le chemin et définit les autres variables ---
if not os.path.exists(DRIVE_PROJECT_PATH):
    try:
        from google.colab import drive
        print("  Montage de Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        if not os.path.exists(DRIVE_PROJECT_PATH):
             raise FileNotFoundError(f"Le chemin Drive '{DRIVE_PROJECT_PATH}' n'existe TOUJOURS PAS après montage. Vérifiez le chemin exact et le nom des dossiers.")
    except ModuleNotFoundError:
         raise FileNotFoundError(f"Google Colab Drive non trouvé et chemin '{DRIVE_PROJECT_PATH}' inexistant.")
    except Exception as e_mount:
         raise FileNotFoundError(f"Erreur lors du montage de Drive ou chemin '{DRIVE_PROJECT_PATH}' inexistant: {e_mount}")

print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

# !!! VÉRIFIEZ CE NOM DE FICHIER EXACT !!!
AP_TAR_FILENAME = "AP.tar" # Est-ce bien 'AP.tar' ? Ou 'ap.tar' ? Autre ?
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, AP_TAR_FILENAME)
# Note: Pensez à modifier la Cellule 0.4 (extraction) pour ouvrir avec "r:" au lieu de "r:gz"

TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes") # Ces sous-dossiers existent-ils ?
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence") # Ces sous-dossiers existent-ils ?
OUTPUT_DIR = "/content/ap_output"
# ... (définition des autres chemins inchangée) ...
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP CIBLE: {AP_TAR_PATH}") # Affiche le chemin complet qui sera vérifié
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# ... (reste de la cellule inchangé : définition preprocess_text, parsing topics, vérifications finales) ...

# === Cellule 0.4 (Modifiée): Extraire et Formater les Documents depuis AP.tar ===
import tarfile
import re
import json
from tqdm.notebook import tqdm # Barre de progression
import os # Assurer que os est importé
import traceback # Pour afficher les erreurs

# Chemins définis dans la cellule précédente (combined_setup_paths_fixed)
# AP_TAR_PATH devrait pointer vers ".../AP.tar"
# CORPUS_DIR devrait être défini

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction et formatage des documents depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# Vérifier si le fichier AP.tar existe (devrait être bon maintenant, mais double vérification)
if not os.path.exists(AP_TAR_PATH):
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé. Vérifiez le chemin et le nom du fichier dans la cellule de configuration.")

# Regex pour extraire DOCNO et TEXT
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

doc_count = 0
file_read_count = 0
skipped_members = 0

# Ouvrir/créer le fichier JSONL de sortie
# Utiliser le mode "r:" pour un fichier .tar non compressé
try:
    # Utiliser encoding='utf-8' pour l'écriture du fichier JSONL
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"  {len(members)} membres trouvés dans l'archive TAR.")
        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            # Ignorer les dossiers ou les fichiers non réguliers
            if not member.isfile():
                skipped_members += 1
                continue

            file_read_count += 1
            # Extraire le contenu du fichier
            try:
                f = tar.extractfile(member)
                if f: # S'assurer que l'extraction a réussi
                    # Lire et décoder avec gestion des erreurs
                    content = f.read().decode('utf-8', errors='ignore')
                    f.close()

                    # Trouver tous les documents (<DOC>...</DOC>) dans le fichier actuel
                    for doc_match in doc_pattern.finditer(content):
                        doc_content = doc_match.group(1)

                        # Extraire DOCNO
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match:
                            continue
                        doc_id = docno_match.group(1).strip()

                        # Extraire TEXT
                        text_match = text_pattern.search(doc_content)
                        if text_match:
                           doc_text = text_match.group(1).strip()
                           doc_text = ' '.join(doc_text.split()) # Nettoyage espaces
                        else:
                            doc_text = ""

                        # Écrire l'entrée JSONL
                        try:
                            json_line = json.dumps({"id": doc_id, "contents": doc_text})
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur lors de l'écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key:
                # Peut arriver si le membre est listé mais inaccessible
                print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}")
                skipped_members += 1
            except EOFError:
                 print(f"\nAvertissement: Fin de fichier inattendue lors de la lecture de {member.name}.")
                 skipped_members += 1
            except Exception as e_extract:
                print(f"\nErreur inattendue lors de l'extraction/lecture du membre {member.name}: {e_extract}")
                skipped_members += 1

except tarfile.ReadError as e_tar:
    print(f"\nERREUR: Impossible de lire le fichier TAR {AP_TAR_PATH}. Est-il corrompu ou n'est-ce pas un fichier TAR valide? Erreur: {e_tar}")
    raise e_tar # Arrêter si le TAR est illisible
except FileNotFoundError:
     print(f"\nERREUR: Le fichier TAR {AP_TAR_PATH} n'a pas été trouvé au moment de l'ouverture.")
     raise FileNotFoundError
except Exception as e_general:
     print(f"\nERREUR générale lors du traitement du fichier TAR: {e_general}")
     traceback.print_exc()
     raise e_general


print(f"\nTraitement terminé.")
print(f"  {file_read_count} fichiers lus depuis l'archive.")
print(f"  {skipped_members} membres ignorés (dossiers ou erreurs).")
print(f"  {doc_count} documents formatés et écrits dans {JSONL_OUTPUT_PATH}")
if doc_count < 100000: # Seuil arbitraire pour AP
     print("  ATTENTION: Le nombre de documents extraits semble faible. Vérifiez le fichier TAR et les regex.")

# === Cellule 1.2: Indexation Baseline ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini

# Chemins définis précédemment
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")
# INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
# CORPUS_DIR contient le fichier JSONL

print(f"Début de l'indexation Baseline (sans prétraitement explicite)...")
print(f"Collection source: {JSONL_OUTPUT_PATH}")
print(f"Répertoire de l'index: {INDEX_DIR_BASELINE}")

# Commande Pyserini pour l'indexation
# -input: dossier contenant les fichiers JSONL (ici CORPUS_DIR)
# -collection: type de collection (JsonCollection pour nos fichiers .jsonl)
# -generator: comment traiter les fichiers (DefaultLuceneDocumentGenerator crée un document Lucene par ligne JSON)
# -index: chemin où sauvegarder l'index
# -threads: nombre de threads à utiliser (ajustez si besoin, 4 est raisonnable pour Colab)
# -storePositions -storeDocvectors -storeRaw: stocke informations supplémentaires utiles
index_cmd_baseline = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR,
    "--index", INDEX_DIR_BASELINE,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté (ex: 2 ou 8 selon les ressources Colab)
    "--storePositions", "--storeDocvectors", "--storeRaw"
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_baseline)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion des erreurs/sorties
    result = subprocess.run(index_cmd_baseline, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    print(f"\nIndexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Baseline a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    # Vous pouvez décider de lever l'erreur pour arrêter ou juste afficher un message
    # raise e
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Baseline a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    # raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Baseline: {e}")
    import traceback
    traceback.print_exc()
    # raise e

# === Cellule 1.3: Préparer les Données Prétraitées ===
import json
from tqdm.notebook import tqdm
import os
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")
# CORPUS_DIR

JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl")

print(f"Préparation des données prétraitées depuis {JSONL_OUTPUT_PATH} vers {JSONL_PREPROC_PATH}...")

# S'assurer que la fonction preprocess_text est définie (normalement fait dans la cellule de setup)
if 'preprocess_text' not in globals():
    print("Erreur: La fonction 'preprocess_text' n'est pas définie. Ré-exécutez la cellule de configuration.")
    # Optionnel: Redéfinir ici si nécessaire (copier depuis la cellule de setup)
    raise NameError("preprocess_text non définie")
else:
    doc_count_preproc = 0
    error_count = 0
    # Lire le fichier JSONL original et écrire le fichier prétraité
    try:
        # Utiliser utf-8 pour lire et écrire
        with open(JSONL_OUTPUT_PATH, 'r', encoding='utf-8') as infile, \
             open(JSONL_PREPROC_PATH, 'w', encoding='utf-8') as outfile:

            # Itérer sur le fichier d'entrée
            # Utiliser tqdm pour la barre de progression
            for line in tqdm(infile, desc="Prétraitement des documents"):
                try:
                    data = json.loads(line)
                    # Utiliser .get pour la robustesse si 'id' ou 'contents' manque
                    doc_id = data.get('id', None)
                    original_contents = data.get('contents', '')

                    if doc_id is None:
                        # print("Avertissement: Ligne JSON sans 'id', ignorée.")
                        error_count += 1
                        continue

                    # Appliquer le prétraitement
                    preprocessed_contents = preprocess_text(original_contents)

                    # Écrire la nouvelle ligne JSONL
                    # S'assurer que l'ID est une chaîne et le contenu aussi
                    json_line = json.dumps({"id": str(doc_id), "contents": str(preprocessed_contents)})
                    outfile.write(json_line + '\n')
                    doc_count_preproc += 1

                except json.JSONDecodeError:
                    # print(f"Avertissement: Erreur de décodage JSON sur une ligne, ignorée.")
                    error_count += 1
                except Exception as e_line:
                    print(f"\nErreur inattendue lors du prétraitement d'une ligne (id={data.get('id', 'inconnu')}): {e_line}")
                    error_count += 1
                    # Optionnel: Afficher la trace pour débugger des erreurs spécifiques
                    # traceback.print_exc()


        print(f"\nTerminé.")
        print(f"  {doc_count_preproc} documents prétraités et écrits dans {JSONL_PREPROC_PATH}")
        if error_count > 0:
             print(f"  {error_count} lignes ignorées à cause d'erreurs.")

    except FileNotFoundError:
        print(f"ERREUR: Le fichier d'entrée {JSONL_OUTPUT_PATH} n'a pas été trouvé.")
        raise
    except Exception as e_main:
        print(f"ERREUR générale lors de la préparation des données prétraitées: {e_main}")
        traceback.print_exc()
        raise

# === Cellule 1.4: Indexation Avec Prétraitement ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment
# JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl") # Fichier source
# INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed") # Dossier cible pour l'index
# CORPUS_DIR contient le fichier JSONL prétraité

print(f"Début de l'indexation avec Prétraitement...")
# Note: Pyserini s'attend à un dossier en entrée pour JsonCollection,
# il trouvera ap_docs_preprocessed.jsonl dans CORPUS_DIR.
print(f"Collection source (dossier): {CORPUS_DIR}")
print(f"Fichier JSONL prétraité attendu: {JSONL_PREPROC_PATH}")
print(f"Répertoire de l'index cible: {INDEX_DIR_PREPROC}")

# Vérifier si le fichier prétraité existe
if not os.path.exists(JSONL_PREPROC_PATH):
    raise FileNotFoundError(f"Le fichier de données prétraitées {JSONL_PREPROC_PATH} n'a pas été trouvé. Assurez-vous que l'étape précédente (1.3) s'est bien terminée.")

# Commande Pyserini pour l'indexation prétraitée
index_cmd_preproc = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR, # Pointeur vers le dossier contenant les jsonl
    "--index", INDEX_DIR_PREPROC,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté
    "--storePositions", "--storeDocvectors", "--storeRaw",
    "--pretokenized" # Important: Indique que le texte est déjà tokenisé/traité
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_preproc)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion
    result = subprocess.run(index_cmd_preproc, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    print(f"\nIndexation avec Prétraitement terminée. Index créé dans {INDEX_DIR_PREPROC}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Prétraitée a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    raise e
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Prétraitée a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Prétraitée: {e}")
    traceback.print_exc()
    raise e

# === Cellule 3.1: Exécuter les Recherches (Séquentielles - BM25 & TF-IDF) ===
# Utilise la dernière Pyserini et Java 21
# Assurez-vous que les variables d'index et de requêtes sont définies par la cellule de config
# INDEX_DIR_BASELINE, INDEX_DIR_PREPROC
# queries_short, queries_long, queries_short_preprocessed, queries_long_preprocessed
# K_RESULTS devrait aussi être défini (sinon, on le mettra à 1000)

from pyserini.search.lucene import LuceneSearcher
import time
from tqdm.notebook import tqdm # Toujours utile pour la progression
import traceback # Pour afficher les erreurs détaillées
import os # Assurer que os est importé
from jnius import autoclass, JavaException # Importer pour TF-IDF

# Essayer de définir K_RESULTS si ce n'est pas déjà fait
try:
    K_RESULTS
except NameError:
    print("Définition de K_RESULTS (nombre de résultats) à 1000...")
    K_RESULTS = 1000

# --- Configuration des modèles de similarité ---
# Charger la classe Java pour TF-IDF (ClassicSimilarity)
# Mettre dans un try-except au cas où l'import échouerait encore (peu probable avec Java 21)
try:
    ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')
    print("Classe ClassicSimilarity (pour TF-IDF) chargée avec succès.")
except JavaException as e_load_class:
    print(f"ERREUR Java lors du chargement de ClassicSimilarity: {e_load_class}")
    print("Les recherches TF-IDF échoueront probablement.")
    ClassicSimilarity = None # Mettre à None pour pouvoir vérifier plus tard
except Exception as e_load_gen:
     print(f"ERREUR inattendue lors du chargement de ClassicSimilarity: {e_load_gen}")
     ClassicSimilarity = None


def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}" # Ex: baseline_short_bm25 or baseline_short_tfidf
    print(f"\nDébut recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', PréprocReq={use_preprocessed_query}, k={k}")

    all_results_list = []
    searcher = None # Initialiser à None

    try:
        # Initialiser le searcher UNE SEULE FOIS pour toutes les requêtes de ce run
        print(f"  Initialisation de LuceneSearcher pour {run_tag}...")
        # Assurer que LuceneSearcher est importé
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(index_path)
        print(f"  LuceneSearcher initialisé.")

        # Configurer le modèle de similarité
        if model == 'bm25':
            print("  Configuration de BM25...")
            searcher.set_bm25(k1=0.9, b=0.4)
            print("  BM25 configuré.")
        elif model == 'tfidf':
            if ClassicSimilarity is None:
                 print("ERREUR: Classe ClassicSimilarity non chargée. Impossible de configurer TF-IDF.")
                 print(f"--- ABANDON du run {run_tag} ---")
                 return # Ne pas continuer si la classe n'a pas pu être chargée

            print("  Configuration de ClassicSimilarity (TF-IDF)...")
            try:
                 searcher.set_similarity(ClassicSimilarity())
                 print("  ClassicSimilarity configurée.")
            except JavaException as e:
                 print(f"ERREUR Java lors de la configuration de ClassicSimilarity: {e}")
                 print(traceback.format_exc())
                 print(f"--- ABANDON du run {run_tag} à cause de l'erreur de configuration TF-IDF ---")
                 return
            except Exception as e_other:
                 print(f"ERREUR Inattendue lors de la configuration de ClassicSimilarity: {e_other}")
                 print(traceback.format_exc())
                 print(f"--- ABANDON du run {run_tag} à cause d'une erreur TF-IDF ---")
                 return
        else:
            print(f"Modèle '{model}' non reconnu, utilisation de BM25 par défaut...")
            searcher.set_bm25()
            print("  BM25 par défaut configuré.")

        # Itérer sur les requêtes séquentiellement
        query_errors = 0
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                # S'assurer que preprocess_text est défini
                if 'preprocess_text' not in globals():
                     raise NameError("La fonction preprocess_text n'est pas définie.")

                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                hits = searcher.search(search_text, k=k)

                # Formater les résultats pour cette requête
                for i in range(len(hits)):
                    rank = i + 1
                    doc_id = hits[i].docid
                    score = hits[i].score
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            except Exception as e_query:
                # Compter les erreurs par requête mais continuer
                query_errors += 1
                if query_errors < 10: # Limiter l'affichage des erreurs par requête
                     print(f"\nErreur lors de la recherche pour QID {query_id} avec {run_tag}: {e_query}")
                elif query_errors == 10:
                     print("\nPlusieurs erreurs de recherche pour ce run, messages suivants masqués...")


        # Écrire les résultats dans le fichier de run TREC
        if all_results_list:
             with open(output_run_file, 'w', encoding='utf-8') as f_out:
                f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes de résultats écrites.")
        else:
            print("\n  Avertissement: Aucun résultat généré pour ce run.")

        if query_errors > 0:
            print(f"  Avertissement: {query_errors} erreurs rencontrées lors de la recherche sur les requêtes individuelles.")

        end_time = time.time()
        print(f"Recherche SÉQUENTIELLE terminée pour {run_tag}. Résultats sauvegardés dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")

    except Exception as e_main:
        # Erreur pendant l'initialisation du searcher ou configuration BM25
        print(f"\nERREUR MAJEURE pendant l'exécution de {run_tag}: {e_main}")
        print(traceback.format_exc())
    finally:
        if searcher:
             print(f"  Nettoyage implicite des ressources pour {run_tag}.")
             pass


# --- Exécution des 8 configurations de recherche (Séquentiel) ---

print("\n--- DÉBUT DES RECHERCHES BASELINE ---")
# 1. Index Baseline + Requêtes Courtes + BM25
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")

# 2. Index Baseline + Requêtes Courtes + TF-IDF
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")

# 3. Index Baseline + Requêtes Longues + BM25
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")

# 4. Index Baseline + Requêtes Longues + TF-IDF
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")

print("\n--- DÉBUT DES RECHERCHES PRÉTRAITÉES ---")
# 5. Index Preprocessed + Requêtes Courtes (Prétraitées) + BM25
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)

# 6. Index Preprocessed + Requêtes Courtes (Prétraitées) + TF-IDF
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)

# 7. Index Preprocessed + Requêtes Longues (Prétraitées) + BM25
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)

# 8. Index Preprocessed + Requêtes Longues (Prétraitées) + TF-IDF
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)

print("\n--- Toutes les recherches de base (mode séquentiel) sont terminées. ---")

# === Cellule de Configuration Complète (Tout-en-un) ===
# Réunit toutes les étapes de setup nécessaires

import os
import sys
import subprocess
import time
import glob
import re
import json
import nltk
from tqdm.notebook import tqdm # Assurer l'import pour les fonctions

print("--- Début de la Configuration Complète ---")
print("Cela peut prendre plusieurs minutes...")

# --- Partie 1: Installation Java 21 et Configuration ---
print("\n[1/9] Installation de OpenJDK 21...")
install_java_cmd = "apt-get update -qq > /dev/null && apt-get install -y openjdk-21-jdk-headless -qq > /dev/null"
try:
    subprocess.run(install_java_cmd, shell=True, check=True, timeout=180)
    print("OpenJDK 21 installé.")
except Exception as e:
    print(f"ERREUR lors de l'installation de Java 21: {e}")
    raise # Arrêter si Java ne s'installe pas

print("\n[2/9] Configuration de Java 21 comme défaut via update-alternatives...")
java_path_21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
if os.path.exists(java_path_21):
    try:
        subprocess.run(f"update-alternatives --install /usr/bin/java java {java_path_21} 1", shell=True, check=True)
        subprocess.run(f"update-alternatives --set java {java_path_21}", shell=True, check=True)
        print("update-alternatives configuré pour java.")
    except Exception as e:
        print(f"ERREUR lors de la configuration de update-alternatives: {e}")
        # Continuer mais avertir
else:
    print(f"ATTENTION: Chemin Java 21 non trouvé à {java_path_21}. update-alternatives non configuré.")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
if not os.path.exists(os.environ["JAVA_HOME"]):
     print(f"ATTENTION: Le chemin JAVA_HOME '{os.environ['JAVA_HOME']}' n'existe pas.")

# --- Partie 2: Installation des outils de build C++ ---
print("\n[3/9] Installation des outils de build (build-essential, cmake)...")
install_build_cmd = "apt-get update -qq > /dev/null && apt-get install -y build-essential cmake -qq > /dev/null"
try:
    subprocess.run(install_build_cmd, shell=True, check=True, timeout=180)
    print("Outils de build installés.")
except Exception as e_build:
    print(f"ERREUR lors de l'installation des outils de build: {e_build}")
    # Continuer mais avertir

# --- Partie 3: Installation de pybind11 ---
print("\n[4/9] Installation de pybind11...")
install_pybind_cmd = f"{sys.executable} -m pip install pybind11 -q" # -q peut être enlevé si ça échoue
try:
    subprocess.run(install_pybind_cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
    print("pybind11 installé avec succès.")
except Exception as e_pybind:
    print(f"ERREUR lors de l'installation de pybind11: {e_pybind}")
    # Continuer mais avertir

# --- Partie 4: Installation des Paquets Python Principaux ---
print("\n[5/9] Installation de la DERNIÈRE Pyserini, NLTK, Pytrec_eval...")
# Installer sans -q pour voir les erreurs si ça se reproduit
install_pip_cmd = f"{sys.executable} -m pip install pyserini nltk pytrec_eval"
try:
    result_pip = subprocess.run(install_pip_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600)
    print("Paquets Python principaux installés.")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: La commande pip principale a échoué avec le code {e.returncode}")
    print("Sortie STDOUT de Pip:\n", e.stdout)
    print("Sortie STDERR de Pip:\n", e.stderr)
    raise e # Arrêter si l'installation de pyserini échoue
except subprocess.TimeoutExpired as e:
    print("\nERREUR: La commande pip principale a dépassé le délai d'attente.")
    print("Sortie STDOUT de Pip (partielle):\n", e.stdout)
    print("Sortie STDERR de Pip (partielle):\n", e.stderr)
    raise e
except Exception as e_pip:
    print(f"\nERREUR inattendue lors de l'installation pip: {e_pip}")
    raise e_pip

# --- Partie 5: Téléchargement Ressources NLTK ---
print("\n[6/9] Téléchargement/Vérification des ressources NLTK...")
nltk_resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4']
for resource in nltk_resources:
    try:
        # Essayer de trouver la ressource pour éviter le re-téléchargement inutile
        if resource == 'punkt':
            nltk.data.find(f'tokenizers/{resource}.zip')
        elif resource == 'omw-1.4':
             nltk.data.find(f'corpora/{resource}.zip')
        else:
            nltk.data.find(f'corpora/{resource}.zip')
        # print(f"  Ressource NLTK '{resource}' déjà présente.")
    except nltk.downloader.DownloadError:
        print(f"  Téléchargement de la ressource NLTK '{resource}'...")
        nltk.download(resource, quiet=True)
print("Ressources NLTK prêtes.")

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")

# !!! MODIFIEZ CETTE LIGNE AVEC LE CHEMIN CORRECT VERS VOTRE DOSSIER TREC !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # <--- VÉRIFIEZ ET CORRIGEZ CE CHEMIN !

# --- Vérification et définition des autres chemins ---
if 'google.colab' in sys.modules:
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/My Drive'):
             print("  Montage de Google Drive...")
             drive.mount('/content/drive', force_remount=True)
        else:
             print("  Google Drive déjà monté.")
    except ModuleNotFoundError:
         print("ATTENTION: Google Colab non détecté ou erreur d'import.")
    except Exception as e_mount:
         print(f"ATTENTION: Erreur lors du montage de Drive: {e_mount}")

if not os.path.exists(DRIVE_PROJECT_PATH):
     raise FileNotFoundError(f"Le chemin Drive '{DRIVE_PROJECT_PATH}' n'existe pas. Vérifiez le chemin exact et le nom des dossiers.")

print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

AP_TAR_FILENAME = "AP.tar" # Nom du fichier archive
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, AP_TAR_FILENAME)
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence")
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
# Créer les répertoires de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP CIBLE: {AP_TAR_PATH}")
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# --- Partie 7: Définition Fonction Prétraitement ---
print("\n[8/9] Définition de la fonction preprocess_text...")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
stop_words_set = set(stopwords.words('english'))
lemmatizer_obj = WordNetLemmatizer()
def preprocess_text(text):
    if not isinstance(text, str): return ""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer_obj.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words_set]
    return ' '.join(filtered_tokens)
print("  Fonction preprocess_text définie.")

# --- Partie 8: Parsing des Topics ---
print("\n[9/9] Parsing des topics...")
def parse_topics(file_path):
    topics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
                topic_content = top_match.group(1)
                num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE)
                if not num_match: continue
                topic_id = num_match.group(1).strip()
                title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                desc = desc_match.group(1).strip() if desc_match else ""
                if topic_id and title:
                     topics[topic_id] = {'title': title, 'desc': desc}
    except FileNotFoundError:
        print(f"  ATTENTION: Fichier topic non trouvé: {file_path}")
    except Exception as e_topic:
        print(f"  ATTENTION: Erreur lors du parsing de {file_path}: {e_topic}")
    return topics

if not os.path.exists(TOPICS_DIR):
     print(f"ATTENTION: Le dossier des topics '{TOPICS_DIR}' n'existe pas.")
     topic_files = []
else:
    topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))

all_topics = {}
if not topic_files:
     print(f"  ATTENTION: Aucun fichier topic trouvé dans {TOPICS_DIR}")
else:
    for tf in topic_files:
        all_topics.update(parse_topics(tf))

# Définir les dictionnaires même s'ils sont vides pour éviter NameError plus tard
queries_short = {qid: data['title'] for qid, data in all_topics.items()}
queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()}
queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}
print(f"  {len(all_topics)} topics parsés.")
print(f"  {len(queries_short)} requêtes courtes créées.")

# --- Vérification Finale Java ---
print("\n--- Vérification Finale de la Version Java Active ---")
java_check_cmd = "java -version"
try:
    result = subprocess.run(java_check_cmd, shell=True, check=True, capture_output=True, text=True, timeout=10)
    print("Sortie STDERR (contient souvent la version OpenJDK):\n", result.stderr)
    if "21." not in result.stderr and "21." not in result.stdout:
         print("\nATTENTION: Java 21 ne semble PAS être la version active !")
    else:
         print("\nConfirmation: Java 21 semble être la version active.")
except Exception as e:
    print(f"\nERREUR lors de la vérification Java: {e}")

# --- Vérification Finale Pyserini ---
print("\n--- Vérification Finale de la Version Pyserini Installée ---")
try:
    result_pyserini = subprocess.run(f"{sys.executable} -m pip show pyserini", shell=True, check=True, capture_output=True, text=True, timeout=30)
    print(result_pyserini.stdout)
except Exception as e_pyserini_check:
    print(f"ERREUR lors de la vérification de Pyserini: {e_pyserini_check}")

print("\n--- Configuration Complète Terminée ---")
# Ajouter un délai pour s'assurer que tout est stable avant la prochaine cellule
print("\nPause de 5 secondes...")
time.sleep(5)
print("Prêt pour la suite.")

# === Cellule 0.4 (Modifiée): Extraire et Formater les Documents depuis AP.tar ===
import tarfile
import re
import json
from tqdm.notebook import tqdm # Barre de progression
import os # Assurer que os est importé
import traceback # Pour afficher les erreurs

# Chemins définis dans la cellule précédente (full_setup_code)
# AP_TAR_PATH devrait pointer vers ".../AP.tar"
# CORPUS_DIR devrait être défini

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction et formatage des documents depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# Vérifier si le fichier AP.tar existe (devrait être bon maintenant)
if not os.path.exists(AP_TAR_PATH):
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé. Vérifiez le chemin et le nom du fichier dans la cellule de configuration.")

# Regex pour extraire DOCNO et TEXT
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

doc_count = 0
file_read_count = 0
skipped_members = 0

# Ouvrir/créer le fichier JSONL de sortie
# Utiliser le mode "r:" pour un fichier .tar non compressé
try:
    # Utiliser encoding='utf-8' pour l'écriture du fichier JSONL
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"  {len(members)} membres trouvés dans l'archive TAR.")
        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            # Ignorer les dossiers ou les fichiers non réguliers
            if not member.isfile():
                skipped_members += 1
                continue

            file_read_count += 1
            # Extraire le contenu du fichier
            try:
                f = tar.extractfile(member)
                if f: # S'assurer que l'extraction a réussi
                    # Lire et décoder avec gestion des erreurs
                    content = f.read().decode('utf-8', errors='ignore')
                    f.close()

                    # Trouver tous les documents (<DOC>...</DOC>) dans le fichier actuel
                    for doc_match in doc_pattern.finditer(content):
                        doc_content = doc_match.group(1)

                        # Extraire DOCNO
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match:
                            continue
                        doc_id = docno_match.group(1).strip()

                        # Extraire TEXT
                        text_match = text_pattern.search(doc_content)
                        if text_match:
                           doc_text = text_match.group(1).strip()
                           doc_text = ' '.join(doc_text.split()) # Nettoyage espaces
                        else:
                            doc_text = ""

                        # Écrire l'entrée JSONL
                        try:
                            json_line = json.dumps({"id": str(doc_id), "contents": str(doc_text)}) # Assurer str
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur lors de l'écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key:
                # Peut arriver si le membre est listé mais inaccessible
                print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}")
                skipped_members += 1
            except EOFError:
                 print(f"\nAvertissement: Fin de fichier inattendue lors de la lecture de {member.name}.")
                 skipped_members += 1
            except Exception as e_extract:
                print(f"\nErreur inattendue lors de l'extraction/lecture du membre {member.name}: {e_extract}")
                skipped_members += 1

except tarfile.ReadError as e_tar:
    print(f"\nERREUR: Impossible de lire le fichier TAR {AP_TAR_PATH}. Est-il corrompu ou n'est-ce pas un fichier TAR valide? Erreur: {e_tar}")
    raise e_tar # Arrêter si le TAR est illisible
except FileNotFoundError:
     print(f"\nERREUR: Le fichier TAR {AP_TAR_PATH} n'a pas été trouvé au moment de l'ouverture.")
     raise FileNotFoundError
except Exception as e_general:
     print(f"\nERREUR générale lors du traitement du fichier TAR: {e_general}")
     traceback.print_exc()
     raise e_general


print(f"\nTraitement terminé.")
print(f"  {file_read_count} fichiers lus depuis l'archive.")
print(f"  {skipped_members} membres ignorés (dossiers ou erreurs).")
print(f"  {doc_count} documents formatés et écrits dans {JSONL_OUTPUT_PATH}")
if doc_count < 100000: # Seuil arbitraire pour AP
     print("  ATTENTION: Le nombre de documents extraits semble faible. Vérifiez le fichier TAR et les regex.")

# === Cellule 1.2: Indexation Baseline ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment dans la cellule de configuration complète
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl") # Fichier source
# INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline") # Dossier cible
# CORPUS_DIR contient le fichier JSONL

print(f"Début de l'indexation Baseline (sans prétraitement explicite)...")
# Pyserini utilise le dossier CORPUS_DIR comme entrée pour JsonCollection
print(f"Dossier source contenant ap_docs.jsonl: {CORPUS_DIR}")
print(f"Répertoire de l'index cible: {INDEX_DIR_BASELINE}")

# Commande Pyserini pour l'indexation
# Utilise la dernière version de Pyserini installée
index_cmd_baseline = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR,
    "--index", INDEX_DIR_BASELINE,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté (ex: 2 ou 8 selon les ressources Colab)
    "--storePositions", "--storeDocvectors", "--storeRaw" # Options utiles pour certaines techniques avancées
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_baseline)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion des erreurs/sorties
    # Augmentation possible du timeout si l'indexation est très longue
    result = subprocess.run(index_cmd_baseline, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    print(f"\nIndexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Baseline a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    raise e # Arrêter si l'indexation échoue
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Baseline a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Baseline: {e}")
    traceback.print_exc()
    raise e

# === Cellule 1.2: Indexation Baseline ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment dans la cellule de configuration complète
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl") # Fichier source
# INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline") # Dossier cible
# CORPUS_DIR contient le fichier JSONL

print(f"Début de l'indexation Baseline (sans prétraitement explicite)...")
# Pyserini utilise le dossier CORPUS_DIR comme entrée pour JsonCollection
print(f"Dossier source contenant ap_docs.jsonl: {CORPUS_DIR}")
print(f"Répertoire de l'index cible: {INDEX_DIR_BASELINE}")

# Commande Pyserini pour l'indexation
# Utilise la dernière version de Pyserini installée
index_cmd_baseline = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR,
    "--index", INDEX_DIR_BASELINE,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté (ex: 2 ou 8 selon les ressources Colab)
    "--storePositions", "--storeDocvectors", "--storeRaw" # Options utiles pour certaines techniques avancées
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_baseline)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion des erreurs/sorties
    # Augmentation possible du timeout si l'indexation est très longue
    result = subprocess.run(index_cmd_baseline, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    print(f"\nIndexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Baseline a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    raise e # Arrêter si l'indexation échoue
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Baseline a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Baseline: {e}")
    traceback.print_exc()
    raise e

# === Cellule de Vérification et Nettoyage du Corpus ===
import os
import subprocess

print("--- Vérification du contenu du dossier Corpus ---")

# Redéfinir CORPUS_DIR au cas où (normalement défini dans la config)
OUTPUT_DIR = "/content/ap_output"
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")

# Vérifier si le dossier CORPUS_DIR existe
if not os.path.exists(CORPUS_DIR):
    print(f"ERREUR: Le dossier {CORPUS_DIR} n'existe pas. L'étape d'extraction a peut-être échoué.")
else:
    print(f"Contenu du dossier : {CORPUS_DIR}")
    # Utiliser !ls pour lister le contenu
    !ls -lh {CORPUS_DIR}
    print("-" * 30)

    print("\n--- Vérification du format de ap_docs.jsonl ---")
    jsonl_path = os.path.join(CORPUS_DIR, "ap_docs.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"ERREUR: Le fichier {jsonl_path} n'existe pas. L'étape d'extraction a échoué.")
    else:
        print(f"Affichage des 3 premières lignes de : {jsonl_path}")
        # Utiliser !head pour afficher les premières lignes
        !head -n 3 {jsonl_path}
        print("-" * 30)

    print("\n--- Vérification et Nettoyage potentiel ---")
    preproc_jsonl_path = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl")
    if os.path.exists(preproc_jsonl_path):
        print(f"Le fichier prétraité {preproc_jsonl_path} existe.")
        print("Il va être supprimé pour éviter les interférences avec l'indexation baseline.")
        try:
            # Utiliser !rm pour supprimer le fichier
            rm_cmd = f"rm '{preproc_jsonl_path}'" # Mettre des guillemets au cas où il y aurait des espaces
            print(f"Exécution de : {rm_cmd}")
            subprocess.run(rm_cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"Fichier {preproc_jsonl_path} supprimé avec succès.")
            # Vérifier à nouveau le contenu du dossier
            print("\nNouveau contenu du dossier :")
            !ls -lh {CORPUS_DIR}
        except subprocess.CalledProcessError as e:
            print(f"ERREUR lors de la suppression de {preproc_jsonl_path}: {e}")
            print("Sortie STDERR:", e.stderr)
        except Exception as e:
            print(f"ERREUR inattendue lors de la suppression: {e}")
    else:
        print(f"Le fichier prétraité {preproc_jsonl_path} n'existe pas. Aucun nettoyage nécessaire.")
    print("-" * 30)

print("\n--- Vérification et Nettoyage Terminés ---")

# === Cellule 0.4 (Modifiée): Extraire et Formater les Documents depuis AP.tar (Avec Debug) ===
import tarfile
import re
import json
from tqdm.notebook import tqdm # Barre de progression
import os # Assurer que os est importé
import traceback # Pour afficher les erreurs

# Chemins définis précédemment
# AP_TAR_PATH devrait pointer vers ".../AP.tar"
# CORPUS_DIR devrait être défini

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction et formatage des documents depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")
print("--- AJOUT DE DEBUG ---")

# Vérifier si le fichier AP.tar existe
if not os.path.exists(AP_TAR_PATH):
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé. Vérifiez le chemin et le nom du fichier.")
else:
    # Afficher la taille du fichier TAR
    tar_size = os.path.getsize(AP_TAR_PATH)
    print(f"  Taille du fichier {AP_TAR_PATH}: {tar_size} octets.")
    if tar_size < 1024 * 1024: # Moins de 1 Mo, suspect pour AP
        print("  ATTENTION: La taille du fichier TAR semble très petite !")


# Regex pour extraire DOCNO et TEXT
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

doc_count = 0
file_read_count = 0
skipped_members = 0
docs_found_in_files = 0
first_doc_id_found = None
first_doc_text_sample = None

# Ouvrir/créer le fichier JSONL de sortie
# Utiliser le mode "r:" pour un fichier .tar non compressé
try:
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"\nDEBUG: {len(members)} membres trouvés dans l'archive TAR.")
        if not members:
             print("ATTENTION: Aucun membre trouvé dans l'archive TAR. Le fichier est peut-être vide ou corrompu.")

        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            if not member.isfile():
                skipped_members += 1
                # print(f"DEBUG: Membre ignoré (pas un fichier): {member.name}")
                continue

            file_read_count += 1
            if file_read_count % 50 == 0: # Afficher un message tous les 50 fichiers lus
                 print(f"DEBUG: Lecture du fichier {file_read_count}/{len(members)}: {member.name}")

            try:
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode('utf-8', errors='ignore')
                    f.close()

                    # DEBUG: Vérifier si des balises <DOC> sont trouvées
                    doc_matches = doc_pattern.findall(content)
                    num_docs_in_file = len(doc_matches)
                    if num_docs_in_file > 0:
                        docs_found_in_files += 1
                        # print(f"DEBUG: Trouvé {num_docs_in_file} <DOC> dans {member.name}")
                    # elif file_read_count <= 10: # Afficher pour les 10 premiers fichiers si aucun doc trouvé
                         # print(f"DEBUG: Trouvé 0 <DOC> dans {member.name}")


                    for doc_content in doc_matches:
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match:
                            continue
                        doc_id = docno_match.group(1).strip()

                        text_match = text_pattern.search(doc_content)
                        if text_match:
                           doc_text = text_match.group(1).strip()
                           doc_text = ' '.join(doc_text.split())
                        else:
                            doc_text = ""

                        # DEBUG: Sauvegarder le premier ID et extrait de texte trouvés
                        if first_doc_id_found is None:
                            first_doc_id_found = doc_id
                            first_doc_text_sample = doc_text[:100] + "..." # Extrait

                        try:
                            json_line = json.dumps({"id": str(doc_id), "contents": str(doc_text)})
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur lors de l'écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key:
                print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}")
                skipped_members += 1
            except EOFError:
                 print(f"\nAvertissement: Fin de fichier inattendue lors de la lecture de {member.name}.")
                 skipped_members += 1
            except Exception as e_extract:
                print(f"\nErreur inattendue lors de l'extraction/lecture du membre {member.name}: {e_extract}")
                skipped_members += 1

except tarfile.ReadError as e_tar:
    print(f"\nERREUR: Impossible de lire le fichier TAR {AP_TAR_PATH}. Est-il corrompu ou n'est-ce pas un fichier TAR valide? Erreur: {e_tar}")
    raise e_tar
except FileNotFoundError:
     print(f"\nERREUR: Le fichier TAR {AP_TAR_PATH} n'a pas été trouvé au moment de l'ouverture.")
     raise FileNotFoundError
except Exception as e_general:
     print(f"\nERREUR générale lors du traitement du fichier TAR: {e_general}")
     traceback.print_exc()
     raise e_general


print(f"\n--- Fin de l'Extraction (Avec Debug) ---")
print(f"  {file_read_count} fichiers lus depuis l'archive.")
print(f"  {skipped_members} membres ignorés.")
print(f"  {docs_found_in_files} fichiers contenaient au moins une balise <DOC>.")
print(f"  {doc_count} documents au total ont été formatés et écrits dans {JSONL_OUTPUT_PATH}")
if first_doc_id_found:
    print(f"  Premier Doc ID trouvé: {first_doc_id_found}")
    print(f"  Extrait du premier texte: {first_doc_text_sample}")
else:
    print("  Aucun document avec ID et Texte n'a été trouvé/extrait.")

if doc_count == 0 and file_read_count > 0:
     print("\n*** PROBLEME MAJEUR: Aucun document n'a été extrait ! Vérifiez les regex ou la structure interne des fichiers dans AP.tar. ***")
elif doc_count < 100000 and file_read_count > 0:
     print("\n  ATTENTION: Le nombre de documents extraits semble faible.")

# Vérifier la taille du fichier de sortie
if os.path.exists(JSONL_OUTPUT_PATH):
    output_size = os.path.getsize(JSONL_OUTPUT_PATH)
    print(f"  Taille finale de {JSONL_OUTPUT_PATH}: {output_size} octets.")
    if output_size == 0 and doc_count == 0:
        print("  CONFIRMATION: Le fichier de sortie est vide.")

# === Cellule 0.4 (Modifiée): Extraire, Décompresser et Formater les Documents ===
import tarfile
import re
import json
import gzip # Importer le module gzip
from tqdm.notebook import tqdm
import os
import traceback

# Chemins définis précédemment
# AP_TAR_PATH devrait pointer vers ".../AP.tar"
# CORPUS_DIR devrait être défini

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction, Décompression et Formatage depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# Vérifier si le fichier AP.tar existe
if not os.path.exists(AP_TAR_PATH):
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé.")
else:
    tar_size = os.path.getsize(AP_TAR_PATH)
    print(f"  Taille du fichier {AP_TAR_PATH}: {tar_size} octets.")

# Regex (inchangées)
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

doc_count = 0
file_read_count = 0
skipped_members = 0
decompression_errors = 0

# Ouvrir/créer le fichier JSONL de sortie
try:
    # Utiliser encoding='utf-8' pour l'écriture
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"\n{len(members)} membres trouvés dans l'archive TAR.")

        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            if not member.isfile() or not member.name.endswith(('.gz', '.Z')): # Traiter seulement les fichiers .gz ou .Z
                skipped_members += 1
                continue

            file_read_count += 1
            content = "" # Initialiser content

            try:
                f = tar.extractfile(member)
                if f:
                    compressed_content = f.read()
                    f.close()

                    # --- AJOUT : Décompression Gzip ---
                    try:
                        # Décompresser le contenu lu
                        content_bytes = gzip.decompress(compressed_content)
                        # Décoder en texte APRES décompression
                        content = content_bytes.decode('utf-8', errors='ignore')
                    except gzip.BadGzipFile:
                        # print(f"Avertissement: Fichier {member.name} n'est pas un fichier gzip valide, tentative de lecture directe.")
                        # Essayer de décoder directement si ce n'était pas du gzip (moins probable vu les noms)
                        content = compressed_content.decode('utf-8', errors='ignore')
                        decompression_errors += 1
                    except Exception as e_gzip:
                         print(f"\nErreur de décompression pour {member.name}: {e_gzip}")
                         decompression_errors += 1
                         continue # Passer au fichier suivant si la décompression échoue
                    # --- FIN AJOUT ---

                    # Chercher les documents dans le contenu décompressé et décodé
                    doc_matches = doc_pattern.findall(content)
                    if not doc_matches:
                         # Si aucun <DOC> trouvé, passer au membre suivant
                         continue

                    for doc_content in doc_matches:
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match:
                            continue
                        doc_id = docno_match.group(1).strip()

                        text_match = text_pattern.search(doc_content)
                        if text_match:
                           doc_text = text_match.group(1).strip()
                           doc_text = ' '.join(doc_text.split())
                        else:
                            doc_text = ""

                        try:
                            json_line = json.dumps({"id": str(doc_id), "contents": str(doc_text)})
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur lors de l'écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key:
                print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}")
                skipped_members += 1
            except EOFError:
                 print(f"\nAvertissement: Fin de fichier inattendue lors de la lecture de {member.name}.")
                 skipped_members += 1
            except Exception as e_extract:
                print(f"\nErreur inattendue lors de l'extraction/lecture du membre {member.name}: {e_extract}")
                skipped_members += 1

except tarfile.ReadError as e_tar:
    print(f"\nERREUR: Impossible de lire le fichier TAR {AP_TAR_PATH}. Erreur: {e_tar}")
    raise e_tar
except FileNotFoundError:
     print(f"\nERREUR: Le fichier TAR {AP_TAR_PATH} n'a pas été trouvé.")
     raise FileNotFoundError
except Exception as e_general:
     print(f"\nERREUR générale lors du traitement: {e_general}")
     traceback.print_exc()
     raise e_general

print(f"\n--- Fin de l'Extraction et Décompression ---")
print(f"  {file_read_count} fichiers (.gz/.Z) lus depuis l'archive.")
print(f"  {skipped_members} membres ignorés (pas .gz/.Z ou erreur lecture).")
if decompression_errors > 0:
    print(f"  {decompression_errors} erreurs de décompression rencontrées.")
print(f"  {doc_count} documents au total ont été formatés et écrits dans {JSONL_OUTPUT_PATH}")

if doc_count == 0 and file_read_count > 0:
     print("\n*** PROBLEME MAJEUR: Aucun document n'a été extrait même après tentative de décompression ! Vérifiez les regex ou la structure interne des fichiers décompressés. ***")
elif doc_count < 100000 and file_read_count > 0:
     print("\n  ATTENTION: Le nombre de documents extraits semble faible.")

# Vérifier la taille du fichier de sortie
if os.path.exists(JSONL_OUTPUT_PATH):
    output_size = os.path.getsize(JSONL_OUTPUT_PATH)
    print(f"  Taille finale de {JSONL_OUTPUT_PATH}: {output_size} octets.")
    if output_size == 0 and doc_count == 0:
        print("  CONFIRMATION: Le fichier de sortie est vide.")
    elif output_size > 0 and doc_count > 0:
         print("  SUCCÈS: Le fichier de sortie contient des données.")

# === Cellule 1.2: Indexation Baseline ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment dans la cellule de configuration complète
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl") # Fichier source (maintenant non vide)
# INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline") # Dossier cible
# CORPUS_DIR contient le fichier JSONL

print(f"Début de l'indexation Baseline (sans prétraitement explicite)...")
# Pyserini utilise le dossier CORPUS_DIR comme entrée pour JsonCollection
print(f"Dossier source contenant ap_docs.jsonl: {CORPUS_DIR}")
print(f"Répertoire de l'index cible: {INDEX_DIR_BASELINE}")

# Vérifier si le fichier source existe et n'est pas vide
jsonl_source_path = os.path.join(CORPUS_DIR, "ap_docs.jsonl")
if not os.path.exists(jsonl_source_path) or os.path.getsize(jsonl_source_path) == 0:
     raise FileNotFoundError(f"Le fichier source {jsonl_source_path} est manquant ou vide. L'étape d'extraction a échoué.")

# Commande Pyserini pour l'indexation
# Utilise la dernière version de Pyserini installée
index_cmd_baseline = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR,
    "--index", INDEX_DIR_BASELINE,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté (ex: 2 ou 8 selon les ressources Colab)
    "--storePositions", "--storeDocvectors", "--storeRaw" # Options utiles pour certaines techniques avancées
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_baseline)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion des erreurs/sorties
    # Augmentation possible du timeout si l'indexation est très longue
    result = subprocess.run(index_cmd_baseline, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    # Vérifier si la sortie indique un nombre non nul de documents indexés
    if "Total 0 documents indexed" in result.stdout:
         print("\nATTENTION: Pyserini indique que 0 document a été indexé malgré un fichier source non vide. Problème potentiel.")
    else:
         print(f"\nIndexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Baseline a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    raise e # Arrêter si l'indexation échoue
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Baseline a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Baseline: {e}")
    traceback.print_exc()
    raise e

# Vérification finale de l'index (taille)
print(f"\nVérification de la taille de l'index créé dans {INDEX_DIR_BASELINE}...")
if os.path.exists(INDEX_DIR_BASELINE):
    # Commande pour obtenir la taille totale du dossier
    du_cmd = f"du -sh '{INDEX_DIR_BASELINE}'"
    try:
        result_du = subprocess.run(du_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  Taille de l'index: {result_du.stdout.split()[0]}")
    except Exception as e_du:
        print(f"  Impossible de vérifier la taille de l'index: {e_du}")
else:
    print("  ATTENTION: Le dossier de l'index n'a pas été créé.")

# === Cellule 1.3: Préparer les Données Prétraitées ===
import json
from tqdm.notebook import tqdm
import os
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl") # Fichier source (non vide)
# CORPUS_DIR

JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl")

print(f"Préparation des données prétraitées depuis {JSONL_OUTPUT_PATH} vers {JSONL_PREPROC_PATH}...")

# S'assurer que la fonction preprocess_text est définie (normalement fait dans la cellule de setup)
if 'preprocess_text' not in globals():
    print("Erreur: La fonction 'preprocess_text' n'est pas définie. Ré-exécutez la cellule de configuration.")
    raise NameError("preprocess_text non définie")
else:
    doc_count_preproc = 0
    error_count = 0
    # Lire le fichier JSONL original et écrire le fichier prétraité
    try:
        # Utiliser utf-8 pour lire et écrire
        with open(JSONL_OUTPUT_PATH, 'r', encoding='utf-8') as infile, \
             open(JSONL_PREPROC_PATH, 'w', encoding='utf-8') as outfile:

            # Itérer sur le fichier d'entrée
            # Utiliser tqdm pour la barre de progression
            for line in tqdm(infile, desc="Prétraitement des documents"):
                try:
                    data = json.loads(line)
                    # Utiliser .get pour la robustesse si 'id' ou 'contents' manque
                    doc_id = data.get('id', None)
                    original_contents = data.get('contents', '')

                    if doc_id is None:
                        error_count += 1
                        continue

                    # Appliquer le prétraitement
                    preprocessed_contents = preprocess_text(original_contents)

                    # Écrire la nouvelle ligne JSONL
                    json_line = json.dumps({"id": str(doc_id), "contents": str(preprocessed_contents)})
                    outfile.write(json_line + '\n')
                    doc_count_preproc += 1

                except json.JSONDecodeError:
                    error_count += 1
                except Exception as e_line:
                    print(f"\nErreur inattendue lors du prétraitement d'une ligne (id={data.get('id', 'inconnu')}): {e_line}")
                    error_count += 1

        print(f"\nTerminé.")
        print(f"  {doc_count_preproc} documents prétraités et écrits dans {JSONL_PREPROC_PATH}")
        if error_count > 0:
             print(f"  {error_count} lignes ignorées à cause d'erreurs.")

        # Vérifier la taille du fichier de sortie
        if os.path.exists(JSONL_PREPROC_PATH):
            output_size = os.path.getsize(JSONL_PREPROC_PATH)
            print(f"  Taille finale de {JSONL_PREPROC_PATH}: {output_size} octets.")
            if output_size == 0 and doc_count_preproc > 0:
                 print("  ATTENTION: 0 octet écrit malgré le traitement de documents. Problème ?")
        else:
            print(f"  ATTENTION: Le fichier de sortie {JSONL_PREPROC_PATH} n'a pas été créé.")


    except FileNotFoundError:
        print(f"ERREUR: Le fichier d'entrée {JSONL_OUTPUT_PATH} n'a pas été trouvé.")
        raise
    except Exception as e_main:
        print(f"ERREUR générale lors de la préparation des données prétraitées: {e_main}")
        traceback.print_exc()
        raise

# === Cellule 1.4: Indexation Avec Prétraitement ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment
# JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl") # Fichier source
# INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed") # Dossier cible pour l'index
# CORPUS_DIR contient le fichier JSONL prétraité

print(f"Début de l'indexation avec Prétraitement...")
# Note: Pyserini s'attend à un dossier en entrée pour JsonCollection,
# il trouvera ap_docs_preprocessed.jsonl dans CORPUS_DIR.
print(f"Collection source (dossier): {CORPUS_DIR}")
JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl") # Chemin complet pour vérification
print(f"Fichier JSONL prétraité attendu: {JSONL_PREPROC_PATH}")
print(f"Répertoire de l'index cible: {INDEX_DIR_PREPROC}")

# Vérifier si le fichier prétraité existe et n'est pas vide
if not os.path.exists(JSONL_PREPROC_PATH) or os.path.getsize(JSONL_PREPROC_PATH) == 0:
    raise FileNotFoundError(f"Le fichier de données prétraitées {JSONL_PREPROC_PATH} est manquant ou vide. L'étape précédente (1.3) a échoué.")

# Commande Pyserini pour l'indexation prétraitée
index_cmd_preproc = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR, # Pointeur vers le dossier contenant les jsonl
    "--index", INDEX_DIR_PREPROC,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté
    "--storePositions", "--storeDocvectors", "--storeRaw",
    "--pretokenized" # Important: Indique que le texte est déjà tokenisé/traité
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_preproc)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion
    result = subprocess.run(index_cmd_preproc, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    # Vérifier si la sortie indique un nombre non nul de documents indexés
    if "Total 0 documents indexed" in result.stdout:
         print("\nATTENTION: Pyserini indique que 0 document a été indexé. Problème potentiel avec l'indexation prétraitée.")
    else:
        print(f"\nIndexation avec Prétraitement terminée. Index créé dans {INDEX_DIR_PREPROC}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Prétraitée a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    raise e
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Prétraitée a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Prétraitée: {e}")
    traceback.print_exc()
    raise e

# Vérification finale de l'index (taille)
print(f"\nVérification de la taille de l'index créé dans {INDEX_DIR_PREPROC}...")
if os.path.exists(INDEX_DIR_PREPROC):
    # Commande pour obtenir la taille totale du dossier
    du_cmd = f"du -sh '{INDEX_DIR_PREPROC}'"
    try:
        result_du = subprocess.run(du_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  Taille de l'index: {result_du.stdout.split()[0]}")
    except Exception as e_du:
        print(f"  Impossible de vérifier la taille de l'index: {e_du}")
else:
    print("  ATTENTION: Le dossier de l'index n'a pas été créé.")

# === Cellule 3.1: Exécuter les Recherches (Séquentielles - BM25 & TF-IDF) ===
# Utilise la dernière Pyserini et Java 21
# Assurez-vous que les variables d'index et de requêtes sont définies par la cellule de config
# INDEX_DIR_BASELINE, INDEX_DIR_PREPROC
# queries_short, queries_long, queries_short_preprocessed, queries_long_preprocessed
# K_RESULTS devrait aussi être défini (sinon, on le mettra à 1000)

from pyserini.search.lucene import LuceneSearcher
import time
from tqdm.notebook import tqdm # Toujours utile pour la progression
import traceback # Pour afficher les erreurs détaillées
import os # Assurer que os est importé
from jnius import autoclass, JavaException # Importer pour TF-IDF

# Essayer de définir K_RESULTS si ce n'est pas déjà fait
try:
    K_RESULTS
except NameError:
    print("Définition de K_RESULTS (nombre de résultats) à 1000...")
    K_RESULTS = 1000

# --- Configuration des modèles de similarité ---
# Charger la classe Java pour TF-IDF (ClassicSimilarity)
# Mettre dans un try-except au cas où l'import échouerait (peu probable avec Java 21)
try:
    ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')
    print("Classe ClassicSimilarity (pour TF-IDF) chargée avec succès.")
except JavaException as e_load_class:
    print(f"ERREUR Java lors du chargement de ClassicSimilarity: {e_load_class}")
    print("Les recherches TF-IDF échoueront probablement.")
    ClassicSimilarity = None # Mettre à None pour pouvoir vérifier plus tard
except Exception as e_load_gen:
     print(f"ERREUR inattendue lors du chargement de ClassicSimilarity: {e_load_gen}")
     ClassicSimilarity = None


def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}" # Ex: baseline_short_bm25 or baseline_short_tfidf
    print(f"\nDébut recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', PréprocReq={use_preprocessed_query}, k={k}")

    all_results_list = []
    searcher = None # Initialiser à None

    try:
        # Initialiser le searcher UNE SEULE FOIS pour toutes les requêtes de ce run
        print(f"  Initialisation de LuceneSearcher pour {run_tag}...")
        # Assurer que LuceneSearcher est importé
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(index_path)
        print(f"  LuceneSearcher initialisé.")

        # Configurer le modèle de similarité
        if model == 'bm25':
            print("  Configuration de BM25...")
            searcher.set_bm25(k1=0.9, b=0.4)
            print("  BM25 configuré.")
        elif model == 'tfidf':
            if ClassicSimilarity is None:
                 print("ERREUR: Classe ClassicSimilarity non chargée. Impossible de configurer TF-IDF.")
                 print(f"--- ABANDON du run {run_tag} ---")
                 return # Ne pas continuer si la classe n'a pas pu être chargée

            print("  Configuration de ClassicSimilarity (TF-IDF)...")
            try:
                 searcher.set_similarity(ClassicSimilarity())
                 print("  ClassicSimilarity configurée.")
            except JavaException as e:
                 print(f"ERREUR Java lors de la configuration de ClassicSimilarity: {e}")
                 print(traceback.format_exc())
                 print(f"--- ABANDON du run {run_tag} à cause de l'erreur de configuration TF-IDF ---")
                 return
            except Exception as e_other:
                 print(f"ERREUR Inattendue lors de la configuration de ClassicSimilarity: {e_other}")
                 print(traceback.format_exc())
                 print(f"--- ABANDON du run {run_tag} à cause d'une erreur TF-IDF ---")
                 return
        else:
            print(f"Modèle '{model}' non reconnu, utilisation de BM25 par défaut...")
            searcher.set_bm25()
            print("  BM25 par défaut configuré.")

        # Itérer sur les requêtes séquentiellement
        query_errors = 0
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                # S'assurer que preprocess_text est défini
                if 'preprocess_text' not in globals():
                     raise NameError("La fonction preprocess_text n'est pas définie.")

                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                # Vérifier si la requête traitée est vide
                if not search_text.strip():
                     # print(f"  Avertissement: Requête QID {query_id} est vide après traitement, ignorée.")
                     continue # Ignorer les requêtes vides

                hits = searcher.search(search_text, k=k)

                # Formater les résultats pour cette requête
                for i in range(len(hits)):
                    rank = i + 1
                    doc_id = hits[i].docid
                    score = hits[i].score
                    # S'assurer que doc_id n'est pas None (peut arriver dans de rares cas)
                    if doc_id is None:
                        # print(f"  Avertissement: Doc ID est None pour QID {query_id} au rang {rank}, ignoré.")
                        continue
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            except Exception as e_query:
                # Compter les erreurs par requête mais continuer
                query_errors += 1
                if query_errors < 10: # Limiter l'affichage des erreurs par requête
                     print(f"\nErreur lors de la recherche pour QID {query_id} avec {run_tag}: {e_query}")
                elif query_errors == 10:
                     print("\nPlusieurs erreurs de recherche pour ce run, messages suivants masqués...")


        # Écrire les résultats dans le fichier de run TREC
        if all_results_list:
             # Utiliser encoding='utf-8' pour l'écriture
             with open(output_run_file, 'w', encoding='utf-8') as f_out:
                f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes de résultats écrites.")
        else:
            print("\n  Avertissement: Aucun résultat généré pour ce run.")

        if query_errors > 0:
            print(f"  Avertissement: {query_errors} erreurs rencontrées lors de la recherche sur les requêtes individuelles.")

        end_time = time.time()
        print(f"Recherche SÉQUENTIELLE terminée pour {run_tag}. Résultats sauvegardés dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")

    except Exception as e_main:
        # Erreur pendant l'initialisation du searcher ou configuration BM25
        print(f"\nERREUR MAJEURE pendant l'exécution de {run_tag}: {e_main}")
        print(traceback.format_exc())
    finally:
        # En théorie, Pyserini/jnius gère la fermeture de la JVM, pas besoin de fermer le searcher explicitement
        if searcher:
             print(f"  Nettoyage implicite des ressources pour {run_tag}.")
             pass


# --- Exécution des 8 configurations de recherche (Séquentiel) ---

print("\n--- DÉBUT DES RECHERCHES BASELINE ---")
# 1. Index Baseline + Requêtes Courtes + BM25
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")

# 2. Index Baseline + Requêtes Courtes + TF-IDF
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")

# 3. Index Baseline + Requêtes Longues + BM25
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")

# 4. Index Baseline + Requêtes Longues + TF-IDF
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")

print("\n--- DÉBUT DES RECHERCHES PRÉTRAITÉES ---")
# 5. Index Preprocessed + Requêtes Courtes (Prétraitées) + BM25
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)

# 6. Index Preprocessed + Requêtes Courtes (Prétraitées) + TF-IDF
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)

# 7. Index Preprocessed + Requêtes Longues (Prétraitées) + BM25
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)

# 8. Index Preprocessed + Requêtes Longues (Prétraitées) + TF-IDF
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)

print("\n--- Toutes les recherches de base (mode séquentiel) sont terminées. ---")

# === Cellule de Configuration Complète (Tout-en-un) ===
# Réunit toutes les étapes de setup nécessaires

import os
import sys
import subprocess
import time
import glob
import re
import json
import nltk
from tqdm.notebook import tqdm # Assurer l'import pour les fonctions
import traceback # Pour afficher les erreurs

print("--- Début de la Configuration Complète ---")
print("Cela peut prendre plusieurs minutes...")

# --- Partie 1: Installation Java 21 et Configuration ---
print("\n[1/9] Installation de OpenJDK 21...")
install_java_cmd = "apt-get update -qq > /dev/null && apt-get install -y openjdk-21-jdk-headless -qq > /dev/null"
try:
    subprocess.run(install_java_cmd, shell=True, check=True, timeout=180)
    print("OpenJDK 21 installé.")
except Exception as e:
    print(f"ERREUR lors de l'installation de Java 21: {e}")
    raise # Arrêter si Java ne s'installe pas

print("\n[2/9] Configuration de Java 21 comme défaut via update-alternatives...")
java_path_21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
if os.path.exists(java_path_21):
    try:
        subprocess.run(f"update-alternatives --install /usr/bin/java java {java_path_21} 1", shell=True, check=True)
        subprocess.run(f"update-alternatives --set java {java_path_21}", shell=True, check=True)
        print("update-alternatives configuré pour java.")
    except Exception as e:
        print(f"ERREUR lors de la configuration de update-alternatives: {e}")
        # Continuer mais avertir
else:
    print(f"ATTENTION: Chemin Java 21 non trouvé à {java_path_21}. update-alternatives non configuré.")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
if not os.path.exists(os.environ["JAVA_HOME"]):
     print(f"ATTENTION: Le chemin JAVA_HOME '{os.environ['JAVA_HOME']}' n'existe pas.")

# --- Partie 2: Installation des outils de build C++ ---
print("\n[3/9] Installation des outils de build (build-essential, cmake)...")
install_build_cmd = "apt-get update -qq > /dev/null && apt-get install -y build-essential cmake -qq > /dev/null"
try:
    subprocess.run(install_build_cmd, shell=True, check=True, timeout=180)
    print("Outils de build installés.")
except Exception as e_build:
    print(f"ERREUR lors de l'installation des outils de build: {e_build}")
    # Continuer mais avertir

# --- Partie 3: Installation de pybind11 ---
print("\n[4/9] Installation de pybind11...")
install_pybind_cmd = f"{sys.executable} -m pip install pybind11 -q" # -q peut être enlevé si ça échoue
try:
    subprocess.run(install_pybind_cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
    print("pybind11 installé avec succès.")
except Exception as e_pybind:
    print(f"ERREUR lors de l'installation de pybind11: {e_pybind}")
    # Continuer mais avertir

# --- Partie 4: Installation des Paquets Python Principaux ---
print("\n[5/9] Installation de la DERNIÈRE Pyserini, NLTK, Pytrec_eval...")
# Installer sans -q pour voir les erreurs si ça se reproduit
install_pip_cmd = f"{sys.executable} -m pip install pyserini nltk pytrec_eval"
try:
    result_pip = subprocess.run(install_pip_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600)
    print("Paquets Python principaux installés.")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: La commande pip principale a échoué avec le code {e.returncode}")
    print("Sortie STDOUT de Pip:\n", e.stdout)
    print("Sortie STDERR de Pip:\n", e.stderr)
    raise e # Arrêter si l'installation de pyserini échoue
except subprocess.TimeoutExpired as e:
    print("\nERREUR: La commande pip principale a dépassé le délai d'attente.")
    print("Sortie STDOUT de Pip (partielle):\n", e.stdout)
    print("Sortie STDERR de Pip (partielle):\n", e.stderr)
    raise e
except Exception as e_pip:
    print(f"\nERREUR inattendue lors de l'installation pip: {e_pip}")
    raise e_pip

# --- Partie 5: Téléchargement Ressources NLTK ---
print("\n[6/9] Téléchargement/Vérification des ressources NLTK...")
nltk_resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4']
for resource in nltk_resources:
    try:
        # Essayer de trouver la ressource pour éviter le re-téléchargement inutile
        if resource == 'punkt':
            nltk.data.find(f'tokenizers/{resource}.zip')
        elif resource == 'omw-1.4':
             nltk.data.find(f'corpora/{resource}.zip')
        else:
            nltk.data.find(f'corpora/{resource}.zip')
        # print(f"  Ressource NLTK '{resource}' déjà présente.")
    except nltk.downloader.DownloadError:
        print(f"  Téléchargement de la ressource NLTK '{resource}'...")
        nltk.download(resource, quiet=True)
print("Ressources NLTK prêtes.")

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")

# !!! MODIFIEZ CETTE LIGNE AVEC LE CHEMIN CORRECT VERS VOTRE DOSSIER TREC !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # <--- VÉRIFIEZ ET CORRIGEZ CE CHEMIN !

# --- Vérification et définition des autres chemins ---
if 'google.colab' in sys.modules:
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/My Drive'):
             print("  Montage de Google Drive...")
             drive.mount('/content/drive', force_remount=True)
        else:
             print("  Google Drive déjà monté.")
    except ModuleNotFoundError:
         print("ATTENTION: Google Colab non détecté ou erreur d'import.")
    except Exception as e_mount:
         print(f"ATTENTION: Erreur lors du montage de Drive: {e_mount}")

if not os.path.exists(DRIVE_PROJECT_PATH):
     raise FileNotFoundError(f"Le chemin Drive '{DRIVE_PROJECT_PATH}' n'existe pas. Vérifiez le chemin exact et le nom des dossiers.")

print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

AP_TAR_FILENAME = "AP.tar" # Nom du fichier archive
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, AP_TAR_FILENAME)
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence")
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
# Créer les répertoires de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP CIBLE: {AP_TAR_PATH}")
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# --- Partie 7: Définition Fonction Prétraitement ---
print("\n[8/9] Définition de la fonction preprocess_text...")
# S'assurer que nltk est importé avant d'utiliser ses modules
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
# Utiliser des noms de variables différents pour éviter conflits potentiels
stop_words_set_global = set(stopwords.words('english'))
lemmatizer_obj_global = WordNetLemmatizer()
def preprocess_text(text):
    if not isinstance(text, str): return ""
    # Utiliser les objets globaux définis ici
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer_obj_global.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words_set_global]
    return ' '.join(filtered_tokens)
print("  Fonction preprocess_text définie.")

# --- Partie 8: Parsing des Topics ---
print("\n[9/9] Parsing des topics...")
# S'assurer que re et glob sont importés
import re
import glob
def parse_topics(file_path):
    topics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
                topic_content = top_match.group(1)
                num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE)
                if not num_match: continue
                topic_id = num_match.group(1).strip()
                title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                desc = desc_match.group(1).strip() if desc_match else ""
                if topic_id and title:
                     topics[topic_id] = {'title': title, 'desc': desc}
    except FileNotFoundError:
        print(f"  ATTENTION: Fichier topic non trouvé: {file_path}")
    except Exception as e_topic:
        print(f"  ATTENTION: Erreur lors du parsing de {file_path}: {e_topic}")
    return topics

if not os.path.exists(TOPICS_DIR):
     print(f"ATTENTION: Le dossier des topics '{TOPICS_DIR}' n'existe pas.")
     topic_files = []
else:
    topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))

all_topics = {}
if not topic_files:
     print(f"  ATTENTION: Aucun fichier topic trouvé dans {TOPICS_DIR}")
else:
    for tf in topic_files:
        all_topics.update(parse_topics(tf))

# Définir les dictionnaires même s'ils sont vides pour éviter NameError plus tard
queries_short = {qid: data['title'] for qid, data in all_topics.items()}
queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()}
queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}
print(f"  {len(all_topics)} topics parsés.")
print(f"  {len(queries_short)} requêtes courtes créées.")

# --- Vérification Finale Java ---
print("\n--- Vérification Finale de la Version Java Active ---")
java_check_cmd = "java -version"
try:
    result = subprocess.run(java_check_cmd, shell=True, check=True, capture_output=True, text=True, timeout=10)
    print("Sortie STDERR (contient souvent la version OpenJDK):\n", result.stderr)
    if "21." not in result.stderr and "21." not in result.stdout:
         print("\nATTENTION: Java 21 ne semble PAS être la version active !")
    else:
         print("\nConfirmation: Java 21 semble être la version active.")
except Exception as e:
    print(f"\nERREUR lors de la vérification Java: {e}")

# --- Vérification Finale Pyserini ---
print("\n--- Vérification Finale de la Version Pyserini Installée ---")
try:
    result_pyserini = subprocess.run(f"{sys.executable} -m pip show pyserini", shell=True, check=True, capture_output=True, text=True, timeout=30)
    print(result_pyserini.stdout)
except Exception as e_pyserini_check:
    print(f"ERREUR lors de la vérification de Pyserini: {e_pyserini_check}")

print("\n--- Configuration Complète Terminée ---")
# Ajouter un délai pour s'assurer que tout est stable avant la prochaine cellule
print("\nPause de 5 secondes...")
time.sleep(5)
print("Prêt pour la suite.")

# === Cellule 0.4 (Modifiée): Extraire, Décompresser et Formater les Documents ===
import tarfile
import re
import json
import gzip # Importer le module gzip
from tqdm.notebook import tqdm
import os
import traceback

# Chemins définis dans la cellule précédente (full_setup_code)
# AP_TAR_PATH devrait pointer vers ".../AP.tar"
# CORPUS_DIR devrait être défini

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction, Décompression et Formatage depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# Vérifier si le fichier AP.tar existe
if not os.path.exists(AP_TAR_PATH):
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé.")
else:
    tar_size = os.path.getsize(AP_TAR_PATH)
    print(f"  Taille du fichier {AP_TAR_PATH}: {tar_size} octets.")

# Regex (inchangées)
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

doc_count = 0
file_read_count = 0
skipped_members = 0
decompression_errors = 0

# Ouvrir/créer le fichier JSONL de sortie
try:
    # Utiliser encoding='utf-8' pour l'écriture
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"\n{len(members)} membres trouvés dans l'archive TAR.")

        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            # Traiter seulement les fichiers se terminant par .gz ou .Z (typique pour TREC)
            # Ignorer les dossiers ou les fichiers non réguliers
            if not member.isfile() or not member.name.lower().endswith(('.gz', '.z')):
                skipped_members += 1
                continue

            file_read_count += 1
            content = "" # Initialiser content

            try:
                f = tar.extractfile(member)
                if f:
                    compressed_content = f.read()
                    f.close()

                    # --- AJOUT : Décompression Gzip ---
                    try:
                        # Décompresser le contenu lu
                        content_bytes = gzip.decompress(compressed_content)
                        # Décoder en texte APRES décompression
                        content = content_bytes.decode('utf-8', errors='ignore')
                    except gzip.BadGzipFile:
                        # print(f"Avertissement: Fichier {member.name} n'est pas un fichier gzip valide, tentative de lecture directe.")
                        # Essayer de décoder directement si ce n'était pas du gzip
                        content = compressed_content.decode('utf-8', errors='ignore')
                        decompression_errors += 1
                    except Exception as e_gzip:
                         print(f"\nErreur de décompression pour {member.name}: {e_gzip}")
                         decompression_errors += 1
                         continue # Passer au fichier suivant si la décompression échoue
                    # --- FIN AJOUT ---

                    # Chercher les documents dans le contenu décompressé et décodé
                    doc_matches = doc_pattern.findall(content)
                    if not doc_matches:
                         # Si aucun <DOC> trouvé, passer au membre suivant
                         continue

                    for doc_content in doc_matches:
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match:
                            continue
                        doc_id = docno_match.group(1).strip()

                        text_match = text_pattern.search(doc_content)
                        if text_match:
                           doc_text = text_match.group(1).strip()
                           doc_text = ' '.join(doc_text.split())
                        else:
                            doc_text = ""

                        try:
                            json_line = json.dumps({"id": str(doc_id), "contents": str(doc_text)})
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur lors de l'écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key:
                print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}")
                skipped_members += 1
            except EOFError:
                 print(f"\nAvertissement: Fin de fichier inattendue lors de la lecture de {member.name}.")
                 skipped_members += 1
            except Exception as e_extract:
                print(f"\nErreur inattendue lors de l'extraction/lecture du membre {member.name}: {e_extract}")
                skipped_members += 1

except tarfile.ReadError as e_tar:
    print(f"\nERREUR: Impossible de lire le fichier TAR {AP_TAR_PATH}. Erreur: {e_tar}")
    raise e_tar
except FileNotFoundError:
     print(f"\nERREUR: Le fichier TAR {AP_TAR_PATH} n'a pas été trouvé.")
     raise FileNotFoundError
except Exception as e_general:
     print(f"\nERREUR générale lors du traitement: {e_general}")
     traceback.print_exc()
     raise e_general

print(f"\n--- Fin de l'Extraction et Décompression ---")
print(f"  {file_read_count} fichiers (.gz/.Z) lus depuis l'archive.")
print(f"  {skipped_members} membres ignorés.")
if decompression_errors > 0:
    print(f"  {decompression_errors} erreurs ou avertissements de décompression rencontrés.")
print(f"  {doc_count} documents au total ont été formatés et écrits dans {JSONL_OUTPUT_PATH}")

if doc_count == 0 and file_read_count > 0:
     print("\n*** PROBLEME MAJEUR: Aucun document n'a été extrait ! Vérifiez les regex ou la structure interne des fichiers décompressés. ***")
elif doc_count < 100000 and file_read_count > 0:
     print("\n  ATTENTION: Le nombre de documents extraits semble faible.")

# Vérifier la taille du fichier de sortie
if os.path.exists(JSONL_OUTPUT_PATH):
    output_size = os.path.getsize(JSONL_OUTPUT_PATH)
    print(f"  Taille finale de {JSONL_OUTPUT_PATH}: {output_size} octets.")
    if output_size == 0 and doc_count == 0:
        print("  CONFIRMATION: Le fichier de sortie est vide.")
    elif output_size > 0 and doc_count > 0:
         print("  SUCCÈS: Le fichier de sortie contient des données.")

# === Cellule 1.2: Indexation Baseline ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment dans la cellule de configuration complète
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl") # Fichier source (maintenant non vide)
# INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline") # Dossier cible
# CORPUS_DIR contient le fichier JSONL

print(f"Début de l'indexation Baseline (sans prétraitement explicite)...")
# Pyserini utilise le dossier CORPUS_DIR comme entrée pour JsonCollection
print(f"Dossier source contenant ap_docs.jsonl: {CORPUS_DIR}")
print(f"Répertoire de l'index cible: {INDEX_DIR_BASELINE}")

# Vérifier si le fichier source existe et n'est pas vide
jsonl_source_path = os.path.join(CORPUS_DIR, "ap_docs.jsonl")
if not os.path.exists(jsonl_source_path) or os.path.getsize(jsonl_source_path) == 0:
     raise FileNotFoundError(f"Le fichier source {jsonl_source_path} est manquant ou vide. L'étape d'extraction a échoué.")

# Commande Pyserini pour l'indexation
# Utilise la dernière version de Pyserini installée
index_cmd_baseline = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR,
    "--index", INDEX_DIR_BASELINE,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté (ex: 2 ou 8 selon les ressources Colab)
    "--storePositions", "--storeDocvectors", "--storeRaw" # Options utiles pour certaines techniques avancées
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_baseline)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion des erreurs/sorties
    # Augmentation possible du timeout si l'indexation est très longue
    result = subprocess.run(index_cmd_baseline, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    # Vérifier si la sortie indique un nombre non nul de documents indexés
    if "Total 0 documents indexed" in result.stdout:
         print("\nATTENTION: Pyserini indique que 0 document a été indexé malgré un fichier source non vide. Problème potentiel.")
    else:
         print(f"\nIndexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Baseline a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    raise e # Arrêter si l'indexation échoue
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Baseline a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Baseline: {e}")
    traceback.print_exc()
    raise e

# Vérification finale de l'index (taille)
print(f"\nVérification de la taille de l'index créé dans {INDEX_DIR_BASELINE}...")
if os.path.exists(INDEX_DIR_BASELINE):
    # Commande pour obtenir la taille totale du dossier
    du_cmd = f"du -sh '{INDEX_DIR_BASELINE}'"
    try:
        result_du = subprocess.run(du_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  Taille de l'index: {result_du.stdout.split()[0]}")
    except Exception as e_du:
        print(f"  Impossible de vérifier la taille de l'index: {e_du}")
else:
    print("  ATTENTION: Le dossier de l'index n'a pas été créé.")

# === Sauvegarde des fichiers générés vers Google Drive ===
    import os
    import subprocess

    # Redéfinir le chemin de base sur Drive (adaptez si nécessaire)
    # Assurez-vous que ce chemin pointe vers le dossier où vous voulez sauvegarder,
    # par exemple, le dossier Projet_RI
    # DRIVE_SAVE_BASE_PATH = "/content/drive/My Drive/Projet_RI" # Exemple
    # Ou utiliser le chemin du projet TREC si vous voulez sauvegarder dedans
    DRIVE_SAVE_BASE_PATH = DRIVE_PROJECT_PATH # Sauvegarde dans le dossier TREC

    # Chemin source dans Colab
    SOURCE_DIR = "/content/ap_output"

    # Chemin cible sur Google Drive
    # Crée un sous-dossier 'colab_output_backup' pour ne pas mélanger
    # avec vos fichiers originaux.
    TARGET_DIR_ON_DRIVE = os.path.join(DRIVE_SAVE_BASE_PATH, "colab_output_backup")

    print(f"Source à copier : {SOURCE_DIR}")
    print(f"Cible sur Drive : {TARGET_DIR_ON_DRIVE}")

    # Vérifier si le dossier source existe
    if os.path.exists(SOURCE_DIR):
        # Créer le dossier cible sur Drive s'il n'existe pas
        os.makedirs(TARGET_DIR_ON_DRIVE, exist_ok=True)
        print("\nCopie des fichiers en cours... (Cela peut prendre quelques minutes)")
        # Utiliser cp -r (récursif) et -v (verbeux)
        copy_cmd = f"cp -r -v '{SOURCE_DIR}/.' '{TARGET_DIR_ON_DRIVE}/'" # Copie le contenu de SOURCE_DIR
        try:
            # Utiliser subprocess pour voir la sortie en temps réel (peut être long)
            # Ou simplement !cp -r ...
            process = subprocess.Popen(copy_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                print("\nCopie terminée avec succès !")
                print(f"Les fichiers de {SOURCE_DIR} ont été copiés dans {TARGET_DIR_ON_DRIVE}")
            else:
                print(f"\nERREUR lors de la copie. Code de retour: {process.returncode}")
                print("STDOUT:", stdout.decode())
                print("STDERR:", stderr.decode())
        except Exception as e:
            print(f"\nERREUR inattendue lors de la copie: {e}")
    else:
        print(f"Le dossier source {SOURCE_DIR} n'existe pas, aucune copie effectuée.")

# === Restauration des fichiers depuis Google Drive ===
    import os
    import subprocess
    import time

    # Chemin où les fichiers ont été sauvegardés sur Drive
    # (Doit correspondre au TARGET_DIR_ON_DRIVE de la cellule save_output_code)
    # Assurez-vous que DRIVE_PROJECT_PATH est défini par la cellule de setup précédente
    try:
        DRIVE_PROJECT_PATH
    except NameError:
        print("ERREUR: La variable DRIVE_PROJECT_PATH n'est pas définie. Exécutez d'abord la cellule de configuration complète.")
        # Optionnel: Redéfinir ici si nécessaire, mais il vaut mieux exécuter la cellule de setup
        # DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC"
        raise

    DRIVE_BACKUP_DIR = os.path.join(DRIVE_PROJECT_PATH, "colab_output_backup")

    # Chemin cible dans Colab (où Pyserini s'attend à les trouver)
    TARGET_RESTORE_DIR = "/content/ap_output"

    print(f"Source sur Drive : {DRIVE_BACKUP_DIR}")
    print(f"Cible dans Colab : {TARGET_RESTORE_DIR}")

    # Vérifier si le dossier de sauvegarde existe sur Drive
    if os.path.exists(DRIVE_BACKUP_DIR):
        # Créer le dossier cible dans Colab s'il n'existe pas
        # (La cellule de setup l'a peut-être déjà créé, mais `exist_ok=True` gère cela)
        os.makedirs(TARGET_RESTORE_DIR, exist_ok=True)

        print("\nRestauration des fichiers en cours... (Cela peut prendre quelques minutes)")
        # Utiliser cp -r (récursif) et -v (verbeux)
        # Copie le contenu de DRIVE_BACKUP_DIR dans TARGET_RESTORE_DIR
        # L'option -T peut être utile si TARGET_RESTORE_DIR existe déjà pour éviter de créer un sous-dossier
        # Mais copier le contenu avec '/.' est généralement plus sûr.
        copy_cmd = f"cp -r -v '{DRIVE_BACKUP_DIR}/.' '{TARGET_RESTORE_DIR}/'"
        try:
            # Exécuter et attendre la fin
            process = subprocess.run(copy_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600) # Timeout 10 minutes
            # Afficher stdout/stderr peut être très long, afficher seulement si erreur?
            # print("STDOUT:", process.stdout)
            # print("STDERR:", process.stderr)
            print("\nRestauration terminée avec succès !")
            print(f"Les fichiers de {DRIVE_BACKUP_DIR} ont été copiés dans {TARGET_RESTORE_DIR}")
            # Vérifier le contenu restauré
            print("\nContenu du dossier restauré (partiel):")
            !ls -lR {TARGET_RESTORE_DIR} | head -n 20 # Afficher une partie du contenu
        except subprocess.CalledProcessError as e:
             print(f"\nERREUR lors de la restauration. Code de retour: {e.returncode}")
             print("STDOUT:", e.stdout)
             print("STDERR:", e.stderr)
             print("\nVérifiez que le dossier de sauvegarde existe et contient les bons fichiers/dossiers (corpus, indexes/baseline).")
             raise e
        except subprocess.TimeoutExpired as e:
            print(f"\nERREUR: La restauration a dépassé le délai d'attente.")
            raise e
        except Exception as e:
            print(f"\nERREUR inattendue lors de la restauration: {e}")
            raise e
    else:
        print(f"ERREUR: Le dossier de sauvegarde {DRIVE_BACKUP_DIR} n'existe pas sur Google Drive.")
        print("Impossible de restaurer les fichiers. Vous devrez relancer les étapes d'extraction et d'indexation baseline.")
        # Optionnel: lever une exception pour arrêter
        # raise FileNotFoundError(f"Dossier de sauvegarde non trouvé: {DRIVE_BACKUP_DIR}")

# === Cellule de Configuration Complète (Pour Reprendre) ===
# Réunit toutes les étapes de setup nécessaires

import os
import sys
import subprocess
import time
import glob
import re
import json
import nltk
from tqdm.notebook import tqdm # Assurer l'import pour les fonctions
import traceback # Pour afficher les erreurs

print("--- Début de la Configuration Complète ---")
print("Cela peut prendre plusieurs minutes...")

# --- Partie 1: Installation Java 21 et Configuration ---
print("\n[1/9] Installation de OpenJDK 21...")
install_java_cmd = "apt-get update -qq > /dev/null && apt-get install -y openjdk-21-jdk-headless -qq > /dev/null"
try:
    subprocess.run(install_java_cmd, shell=True, check=True, timeout=180)
    print("OpenJDK 21 installé.")
except Exception as e:
    print(f"ERREUR lors de l'installation de Java 21: {e}")
    raise # Arrêter si Java ne s'installe pas

print("\n[2/9] Configuration de Java 21 comme défaut via update-alternatives...")
java_path_21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
if os.path.exists(java_path_21):
    try:
        subprocess.run(f"update-alternatives --install /usr/bin/java java {java_path_21} 1", shell=True, check=True)
        subprocess.run(f"update-alternatives --set java {java_path_21}", shell=True, check=True)
        print("update-alternatives configuré pour java.")
    except Exception as e:
        print(f"ERREUR lors de la configuration de update-alternatives: {e}")
        # Continuer mais avertir
else:
    print(f"ATTENTION: Chemin Java 21 non trouvé à {java_path_21}. update-alternatives non configuré.")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
if not os.path.exists(os.environ["JAVA_HOME"]):
     print(f"ATTENTION: Le chemin JAVA_HOME '{os.environ['JAVA_HOME']}' n'existe pas.")

# --- Partie 2: Installation des outils de build C++ ---
print("\n[3/9] Installation des outils de build (build-essential, cmake)...")
install_build_cmd = "apt-get update -qq > /dev/null && apt-get install -y build-essential cmake -qq > /dev/null"
try:
    subprocess.run(install_build_cmd, shell=True, check=True, timeout=180)
    print("Outils de build installés.")
except Exception as e_build:
    print(f"ERREUR lors de l'installation des outils de build: {e_build}")
    # Continuer mais avertir

# --- Partie 3: Installation de pybind11 ---
print("\n[4/9] Installation de pybind11...")
install_pybind_cmd = f"{sys.executable} -m pip install pybind11 -q" # -q peut être enlevé si ça échoue
try:
    subprocess.run(install_pybind_cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
    print("pybind11 installé avec succès.")
except Exception as e_pybind:
    print(f"ERREUR lors de l'installation de pybind11: {e_pybind}")
    # Continuer mais avertir

# --- Partie 4: Installation des Paquets Python Principaux ---
print("\n[5/9] Installation de la DERNIÈRE Pyserini, NLTK, Pytrec_eval...")
# Installer sans -q pour voir les erreurs si ça se reproduit
install_pip_cmd = f"{sys.executable} -m pip install pyserini nltk pytrec_eval"
try:
    result_pip = subprocess.run(install_pip_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600)
    print("Paquets Python principaux installés.")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: La commande pip principale a échoué avec le code {e.returncode}")
    print("Sortie STDOUT de Pip:\n", e.stdout)
    print("Sortie STDERR de Pip:\n", e.stderr)
    raise e # Arrêter si l'installation de pyserini échoue
except subprocess.TimeoutExpired as e:
    print("\nERREUR: La commande pip principale a dépassé le délai d'attente.")
    print("Sortie STDOUT de Pip (partielle):\n", e.stdout)
    print("Sortie STDERR de Pip (partielle):\n", e.stderr)
    raise e
except Exception as e_pip:
    print(f"\nERREUR inattendue lors de l'installation pip: {e_pip}")
    raise e_pip

# --- Partie 5: Téléchargement Ressources NLTK ---
print("\n[6/9] Téléchargement/Vérification des ressources NLTK...")
nltk_resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4']
for resource in nltk_resources:
    try:
        # Essayer de trouver la ressource pour éviter le re-téléchargement inutile
        if resource == 'punkt':
            nltk.data.find(f'tokenizers/{resource}.zip')
        elif resource == 'omw-1.4':
             nltk.data.find(f'corpora/{resource}.zip')
        else:
            nltk.data.find(f'corpora/{resource}.zip')
        # print(f"  Ressource NLTK '{resource}' déjà présente.")
    except nltk.downloader.DownloadError:
        print(f"  Téléchargement de la ressource NLTK '{resource}'...")
        nltk.download(resource, quiet=True)
print("Ressources NLTK prêtes.")

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")

# !!! MODIFIEZ CETTE LIGNE AVEC LE CHEMIN CORRECT VERS VOTRE DOSSIER TREC !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # <--- VÉRIFIEZ ET CORRIGEZ CE CHEMIN !

# --- Vérification et définition des autres chemins ---
if 'google.colab' in sys.modules:
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/My Drive'):
             print("  Montage de Google Drive...")
             drive.mount('/content/drive', force_remount=True)
        else:
             print("  Google Drive déjà monté.")
    except ModuleNotFoundError:
         print("ATTENTION: Google Colab non détecté ou erreur d'import.")
    except Exception as e_mount:
         print(f"ATTENTION: Erreur lors du montage de Drive: {e_mount}")

if not os.path.exists(DRIVE_PROJECT_PATH):
     raise FileNotFoundError(f"Le chemin Drive '{DRIVE_PROJECT_PATH}' n'existe pas. Vérifiez le chemin exact et le nom des dossiers.")

print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

AP_TAR_FILENAME = "AP.tar" # Nom du fichier archive
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, AP_TAR_FILENAME)
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence")
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
# Créer les répertoires de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP CIBLE: {AP_TAR_PATH}")
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# --- Partie 7: Définition Fonction Prétraitement ---
print("\n[8/9] Définition de la fonction preprocess_text...")
# S'assurer que nltk est importé avant d'utiliser ses modules
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
# Utiliser des noms de variables différents pour éviter conflits potentiels
stop_words_set_global = set(stopwords.words('english'))
lemmatizer_obj_global = WordNetLemmatizer()
def preprocess_text(text):
    if not isinstance(text, str): return ""
    # Utiliser les objets globaux définis ici
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer_obj_global.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words_set_global]
    return ' '.join(filtered_tokens)
print("  Fonction preprocess_text définie.")

# --- Partie 8: Parsing des Topics ---
print("\n[9/9] Parsing des topics...")
# S'assurer que re et glob sont importés
import re
import glob
def parse_topics(file_path):
    topics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
                topic_content = top_match.group(1)
                num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE)
                if not num_match: continue
                topic_id = num_match.group(1).strip()
                title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                desc = desc_match.group(1).strip() if desc_match else ""
                if topic_id and title:
                     topics[topic_id] = {'title': title, 'desc': desc}
    except FileNotFoundError:
        print(f"  ATTENTION: Fichier topic non trouvé: {file_path}")
    except Exception as e_topic:
        print(f"  ATTENTION: Erreur lors du parsing de {file_path}: {e_topic}")
    return topics

if not os.path.exists(TOPICS_DIR):
     print(f"ATTENTION: Le dossier des topics '{TOPICS_DIR}' n'existe pas.")
     topic_files = []
else:
    topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))

all_topics = {}
if not topic_files:
     print(f"  ATTENTION: Aucun fichier topic trouvé dans {TOPICS_DIR}")
else:
    for tf in topic_files:
        all_topics.update(parse_topics(tf))

# Définir les dictionnaires même s'ils sont vides pour éviter NameError plus tard
queries_short = {qid: data['title'] for qid, data in all_topics.items()}
queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()}
queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}
print(f"  {len(all_topics)} topics parsés.")
print(f"  {len(queries_short)} requêtes courtes créées.")

# --- Vérification Finale Java ---
print("\n--- Vérification Finale de la Version Java Active ---")
java_check_cmd = "java -version"
try:
    result = subprocess.run(java_check_cmd, shell=True, check=True, capture_output=True, text=True, timeout=10)
    print("Sortie STDERR (contient souvent la version OpenJDK):\n", result.stderr)
    if "21." not in result.stderr and "21." not in result.stdout:
         print("\nATTENTION: Java 21 ne semble PAS être la version active !")
    else:
         print("\nConfirmation: Java 21 semble être la version active.")
except Exception as e:
    print(f"\nERREUR lors de la vérification Java: {e}")

# --- Vérification Finale Pyserini ---
print("\n--- Vérification Finale de la Version Pyserini Installée ---")
try:
    result_pyserini = subprocess.run(f"{sys.executable} -m pip show pyserini", shell=True, check=True, capture_output=True, text=True, timeout=30)
    print(result_pyserini.stdout)
except Exception as e_pyserini_check:
    print(f"ERREUR lors de la vérification de Pyserini: {e_pyserini_check}")

print("\n--- Configuration Complète Terminée ---")
# Ajouter un délai pour s'assurer que tout est stable avant la prochaine cellule
print("\nPause de 5 secondes...")
time.sleep(5)
print("Prêt pour la suite.")

# === Cellule de Configuration Complète (Correction NLTK) ===
# Réunit toutes les étapes de setup nécessaires

import os
import sys
import subprocess
import time
import glob
import re
import json
import nltk # Importer nltk ici pour la partie NLTK
from tqdm.notebook import tqdm # Assurer l'import pour les fonctions
import traceback # Pour afficher les erreurs

print("--- Début de la Configuration Complète ---")
print("Cela peut prendre plusieurs minutes...")

# --- Partie 1: Installation Java 21 et Configuration ---
print("\n[1/9] Installation de OpenJDK 21...")
install_java_cmd = "apt-get update -qq > /dev/null && apt-get install -y openjdk-21-jdk-headless -qq > /dev/null"
try:
    subprocess.run(install_java_cmd, shell=True, check=True, timeout=180)
    print("OpenJDK 21 installé.")
except Exception as e:
    print(f"ERREUR lors de l'installation de Java 21: {e}")
    raise # Arrêter si Java ne s'installe pas

print("\n[2/9] Configuration de Java 21 comme défaut via update-alternatives...")
java_path_21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
if os.path.exists(java_path_21):
    try:
        subprocess.run(f"update-alternatives --install /usr/bin/java java {java_path_21} 1", shell=True, check=True)
        subprocess.run(f"update-alternatives --set java {java_path_21}", shell=True, check=True)
        print("update-alternatives configuré pour java.")
    except Exception as e:
        print(f"ERREUR lors de la configuration de update-alternatives: {e}")
        # Continuer mais avertir
else:
    print(f"ATTENTION: Chemin Java 21 non trouvé à {java_path_21}. update-alternatives non configuré.")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
if not os.path.exists(os.environ["JAVA_HOME"]):
     print(f"ATTENTION: Le chemin JAVA_HOME '{os.environ['JAVA_HOME']}' n'existe pas.")

# --- Partie 2: Installation des outils de build C++ ---
print("\n[3/9] Installation des outils de build (build-essential, cmake)...")
install_build_cmd = "apt-get update -qq > /dev/null && apt-get install -y build-essential cmake -qq > /dev/null"
try:
    subprocess.run(install_build_cmd, shell=True, check=True, timeout=180)
    print("Outils de build installés.")
except Exception as e_build:
    print(f"ERREUR lors de l'installation des outils de build: {e_build}")
    # Continuer mais avertir

# --- Partie 3: Installation de pybind11 ---
print("\n[4/9] Installation de pybind11...")
install_pybind_cmd = f"{sys.executable} -m pip install pybind11 -q" # -q peut être enlevé si ça échoue
try:
    subprocess.run(install_pybind_cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
    print("pybind11 installé avec succès.")
except Exception as e_pybind:
    print(f"ERREUR lors de l'installation de pybind11: {e_pybind}")
    # Continuer mais avertir

# --- Partie 4: Installation des Paquets Python Principaux ---
print("\n[5/9] Installation de la DERNIÈRE Pyserini, NLTK, Pytrec_eval...")
# Installer sans -q pour voir les erreurs si ça se reproduit
install_pip_cmd = f"{sys.executable} -m pip install pyserini nltk pytrec_eval"
try:
    result_pip = subprocess.run(install_pip_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600)
    print("Paquets Python principaux installés.")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: La commande pip principale a échoué avec le code {e.returncode}")
    print("Sortie STDOUT de Pip:\n", e.stdout)
    print("Sortie STDERR de Pip:\n", e.stderr)
    raise e # Arrêter si l'installation de pyserini échoue
except subprocess.TimeoutExpired as e:
    print("\nERREUR: La commande pip principale a dépassé le délai d'attente.")
    print("Sortie STDOUT de Pip (partielle):\n", e.stdout)
    print("Sortie STDERR de Pip (partielle):\n", e.stderr)
    raise e
except Exception as e_pip:
    print(f"\nERREUR inattendue lors de l'installation pip: {e_pip}")
    raise e_pip

# --- Partie 5: Téléchargement Ressources NLTK ---
print("\n[6/9] Téléchargement/Vérification des ressources NLTK...")
# S'assurer que nltk est importé
import nltk
nltk_resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4']
for resource in nltk_resources:
    try:
        # Déterminer le chemin de recherche correct pour nltk.data.find
        if resource == 'punkt':
            resource_path = f'tokenizers/{resource}.zip'
        elif resource == 'omw-1.4':
             resource_path = f'corpora/{resource}.zip' # Open Multilingual Wordnet
        elif resource == 'wordnet':
             resource_path = f'corpora/{resource}.zip'
        else: # stopwords, etc.
            resource_path = f'corpora/{resource}.zip'

        # Essayer de trouver la ressource
        nltk.data.find(resource_path)
        # print(f"  Ressource NLTK '{resource}' déjà présente.")

    # --- CORRECTION ICI: Utiliser except LookupError ---
    except LookupError:
    # --------------------------------------------------
        print(f"  Ressource NLTK '{resource}' non trouvée. Téléchargement...")
        try:
            nltk.download(resource, quiet=True)
            print(f"  Ressource '{resource}' téléchargée.")
        except Exception as e_download:
            # Capturer les erreurs potentielles de téléchargement (réseau, etc.)
            print(f"  ERREUR lors du téléchargement de '{resource}': {e_download}")
            # Optionnel: arrêter si une ressource critique manque
            # if resource in ['punkt', 'stopwords', 'wordnet']: raise

print("Ressources NLTK prêtes.")

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")

# !!! MODIFIEZ CETTE LIGNE AVEC LE CHEMIN CORRECT VERS VOTRE DOSSIER TREC !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # <--- VÉRIFIEZ ET CORRIGEZ CE CHEMIN !

# --- Vérification et définition des autres chemins ---
if 'google.colab' in sys.modules:
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/My Drive'):
             print("  Montage de Google Drive...")
             drive.mount('/content/drive', force_remount=True)
        else:
             print("  Google Drive déjà monté.")
    except ModuleNotFoundError:
         print("ATTENTION: Google Colab non détecté ou erreur d'import.")
    except Exception as e_mount:
         print(f"ATTENTION: Erreur lors du montage de Drive: {e_mount}")

if not os.path.exists(DRIVE_PROJECT_PATH):
     raise FileNotFoundError(f"Le chemin Drive '{DRIVE_PROJECT_PATH}' n'existe pas. Vérifiez le chemin exact et le nom des dossiers.")

print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

AP_TAR_FILENAME = "AP.tar" # Nom du fichier archive
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, AP_TAR_FILENAME)
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence")
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
# Créer les répertoires de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP CIBLE: {AP_TAR_PATH}")
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# --- Partie 7: Définition Fonction Prétraitement ---
print("\n[8/9] Définition de la fonction preprocess_text...")
# S'assurer que nltk est importé avant d'utiliser ses modules
# (Déjà fait plus haut, mais redondance sans danger)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
# Utiliser des noms de variables différents pour éviter conflits potentiels
stop_words_set_global = set(stopwords.words('english'))
lemmatizer_obj_global = WordNetLemmatizer()
def preprocess_text(text):
    if not isinstance(text, str): return ""
    # Utiliser les objets globaux définis ici
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer_obj_global.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words_set_global]
    return ' '.join(filtered_tokens)
print("  Fonction preprocess_text définie.")

# --- Partie 8: Parsing des Topics ---
print("\n[9/9] Parsing des topics...")
# S'assurer que re et glob sont importés
import re
import glob
def parse_topics(file_path):
    topics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
                topic_content = top_match.group(1)
                num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE)
                if not num_match: continue
                topic_id = num_match.group(1).strip()
                title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                desc = desc_match.group(1).strip() if desc_match else ""
                if topic_id and title:
                     topics[topic_id] = {'title': title, 'desc': desc}
    except FileNotFoundError:
        print(f"  ATTENTION: Fichier topic non trouvé: {file_path}")
    except Exception as e_topic:
        print(f"  ATTENTION: Erreur lors du parsing de {file_path}: {e_topic}")
    return topics

if not os.path.exists(TOPICS_DIR):
     print(f"ATTENTION: Le dossier des topics '{TOPICS_DIR}' n'existe pas.")
     topic_files = []
else:
    topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))

all_topics = {}
if not topic_files:
     print(f"  ATTENTION: Aucun fichier topic trouvé dans {TOPICS_DIR}")
else:
    for tf in topic_files:
        all_topics.update(parse_topics(tf))

# Définir les dictionnaires même s'ils sont vides pour éviter NameError plus tard
queries_short = {qid: data['title'] for qid, data in all_topics.items()}
queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()}
queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}
print(f"  {len(all_topics)} topics parsés.")
print(f"  {len(queries_short)} requêtes courtes créées.")

# --- Vérification Finale Java ---
print("\n--- Vérification Finale de la Version Java Active ---")
java_check_cmd = "java -version"
try:
    result = subprocess.run(java_check_cmd, shell=True, check=True, capture_output=True, text=True, timeout=10)
    print("Sortie STDERR (contient souvent la version OpenJDK):\n", result.stderr)
    if "21." not in result.stderr and "21." not in result.stdout:
         print("\nATTENTION: Java 21 ne semble PAS être la version active !")
    else:
         print("\nConfirmation: Java 21 semble être la version active.")
except Exception as e:
    print(f"\nERREUR lors de la vérification Java: {e}")

# --- Vérification Finale Pyserini ---
print("\n--- Vérification Finale de la Version Pyserini Installée ---")
try:
    result_pyserini = subprocess.run(f"{sys.executable} -m pip show pyserini", shell=True, check=True, capture_output=True, text=True, timeout=30)
    print(result_pyserini.stdout)
except Exception as e_pyserini_check:
    print(f"ERREUR lors de la vérification de Pyserini: {e_pyserini_check}")

print("\n--- Configuration Complète Terminée ---")
# Ajouter un délai pour s'assurer que tout est stable avant la prochaine cellule
print("\nPause de 5 secondes...")
time.sleep(5)
print("Prêt pour la suite.")

# === Cellule de Configuration Complète (Correction NLTK punkt_tab) ===
# Réunit toutes les étapes de setup nécessaires

import os
import sys
import subprocess
import time
import glob
import re
import json
import nltk # Importer nltk ici pour la partie NLTK
from tqdm.notebook import tqdm # Assurer l'import pour les fonctions
import traceback # Pour afficher les erreurs

print("--- Début de la Configuration Complète ---")
print("Cela peut prendre plusieurs minutes...")

# --- Partie 1: Installation Java 21 et Configuration ---
print("\n[1/9] Installation de OpenJDK 21...")
install_java_cmd = "apt-get update -qq > /dev/null && apt-get install -y openjdk-21-jdk-headless -qq > /dev/null"
try:
    subprocess.run(install_java_cmd, shell=True, check=True, timeout=180)
    print("OpenJDK 21 installé.")
except Exception as e:
    print(f"ERREUR lors de l'installation de Java 21: {e}")
    raise # Arrêter si Java ne s'installe pas

print("\n[2/9] Configuration de Java 21 comme défaut via update-alternatives...")
java_path_21 = "/usr/lib/jvm/java-21-openjdk-amd64/bin/java"
if os.path.exists(java_path_21):
    try:
        subprocess.run(f"update-alternatives --install /usr/bin/java java {java_path_21} 1", shell=True, check=True)
        subprocess.run(f"update-alternatives --set java {java_path_21}", shell=True, check=True)
        print("update-alternatives configuré pour java.")
    except Exception as e:
        print(f"ERREUR lors de la configuration de update-alternatives: {e}")
        # Continuer mais avertir
else:
    print(f"ATTENTION: Chemin Java 21 non trouvé à {java_path_21}. update-alternatives non configuré.")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
if not os.path.exists(os.environ["JAVA_HOME"]):
     print(f"ATTENTION: Le chemin JAVA_HOME '{os.environ['JAVA_HOME']}' n'existe pas.")

# --- Partie 2: Installation des outils de build C++ ---
print("\n[3/9] Installation des outils de build (build-essential, cmake)...")
install_build_cmd = "apt-get update -qq > /dev/null && apt-get install -y build-essential cmake -qq > /dev/null"
try:
    subprocess.run(install_build_cmd, shell=True, check=True, timeout=180)
    print("Outils de build installés.")
except Exception as e_build:
    print(f"ERREUR lors de l'installation des outils de build: {e_build}")
    # Continuer mais avertir

# --- Partie 3: Installation de pybind11 ---
print("\n[4/9] Installation de pybind11...")
install_pybind_cmd = f"{sys.executable} -m pip install pybind11 -q" # -q peut être enlevé si ça échoue
try:
    subprocess.run(install_pybind_cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
    print("pybind11 installé avec succès.")
except Exception as e_pybind:
    print(f"ERREUR lors de l'installation de pybind11: {e_pybind}")
    # Continuer mais avertir

# --- Partie 4: Installation des Paquets Python Principaux ---
print("\n[5/9] Installation de la DERNIÈRE Pyserini, NLTK, Pytrec_eval...")
# Installer sans -q pour voir les erreurs si ça se reproduit
install_pip_cmd = f"{sys.executable} -m pip install pyserini nltk pytrec_eval"
try:
    result_pip = subprocess.run(install_pip_cmd, shell=True, check=True, capture_output=True, text=True, timeout=600)
    print("Paquets Python principaux installés.")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: La commande pip principale a échoué avec le code {e.returncode}")
    print("Sortie STDOUT de Pip:\n", e.stdout)
    print("Sortie STDERR de Pip:\n", e.stderr)
    raise e # Arrêter si l'installation de pyserini échoue
except subprocess.TimeoutExpired as e:
    print("\nERREUR: La commande pip principale a dépassé le délai d'attente.")
    print("Sortie STDOUT de Pip (partielle):\n", e.stdout)
    print("Sortie STDERR de Pip (partielle):\n", e.stderr)
    raise e
except Exception as e_pip:
    print(f"\nERREUR inattendue lors de l'installation pip: {e_pip}")
    raise e_pip

# --- Partie 5: Téléchargement Ressources NLTK ---
print("\n[6/9] Téléchargement/Vérification des ressources NLTK...")
# S'assurer que nltk est importé
import nltk
# --- CORRECTION ICI: Ajout de 'punkt_tab' ---
nltk_resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4', 'punkt_tab']
# ---------------------------------------------
for resource in nltk_resources:
    try:
        # Déterminer le chemin de recherche correct pour nltk.data.find
        if resource == 'punkt' or resource == 'punkt_tab': # punkt_tab est aussi dans tokenizers
            resource_path = f'tokenizers/{resource}.zip'
        elif resource == 'omw-1.4':
             resource_path = f'corpora/{resource}.zip' # Open Multilingual Wordnet
        elif resource == 'wordnet':
             resource_path = f'corpora/{resource}.zip'
        else: # stopwords, etc.
            resource_path = f'corpora/{resource}.zip'

        # Essayer de trouver la ressource
        nltk.data.find(resource_path)
        # print(f"  Ressource NLTK '{resource}' déjà présente.")

    except LookupError:
        print(f"  Ressource NLTK '{resource}' non trouvée. Téléchargement...")
        try:
            nltk.download(resource, quiet=True)
            print(f"  Ressource '{resource}' téléchargée.")
        except Exception as e_download:
            # Capturer les erreurs potentielles de téléchargement (réseau, etc.)
            print(f"  ERREUR lors du téléchargement de '{resource}': {e_download}")
            # Optionnel: arrêter si une ressource critique manque
            # if resource in ['punkt', 'stopwords', 'wordnet']: raise

print("Ressources NLTK prêtes.")

# --- Partie 6: Définition des Chemins ---
print("\n[7/9] Définition des chemins...")

# !!! MODIFIEZ CETTE LIGNE AVEC LE CHEMIN CORRECT VERS VOTRE DOSSIER TREC !!!
DRIVE_PROJECT_PATH = "/content/drive/My Drive/Projet_RI/TREC" # <--- VÉRIFIEZ ET CORRIGEZ CE CHEMIN !

# --- Vérification et définition des autres chemins ---
if 'google.colab' in sys.modules:
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/My Drive'):
             print("  Montage de Google Drive...")
             drive.mount('/content/drive', force_remount=True)
        else:
             print("  Google Drive déjà monté.")
    except ModuleNotFoundError:
         print("ATTENTION: Google Colab non détecté ou erreur d'import.")
    except Exception as e_mount:
         print(f"ATTENTION: Erreur lors du montage de Drive: {e_mount}")

if not os.path.exists(DRIVE_PROJECT_PATH):
     raise FileNotFoundError(f"Le chemin Drive '{DRIVE_PROJECT_PATH}' n'existe pas. Vérifiez le chemin exact et le nom des dossiers.")

print(f"  Chemin du projet Drive utilisé: {DRIVE_PROJECT_PATH}")

AP_TAR_FILENAME = "AP.tar" # Nom du fichier archive
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, AP_TAR_FILENAME)
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "Topics-requetes")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "jugements de pertinence")
OUTPUT_DIR = "/content/ap_output"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
# Créer les répertoires de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
print(f"  Chemin Fichier AP CIBLE: {AP_TAR_PATH}")
print(f"  Chemin Qrels: {QRELS_DIR}")
print(f"  Chemin Runs: {RUN_DIR}")

# --- Partie 7: Définition Fonction Prétraitement ---
print("\n[8/9] Définition de la fonction preprocess_text...")
# S'assurer que nltk est importé avant d'utiliser ses modules
# (Déjà fait plus haut, mais redondance sans danger)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
# Utiliser des noms de variables différents pour éviter conflits potentiels
stop_words_set_global = set(stopwords.words('english'))
lemmatizer_obj_global = WordNetLemmatizer()
def preprocess_text(text):
    if not isinstance(text, str): return ""
    # Utiliser les objets globaux définis ici
    # Mettre la tokenisation dans un try-except spécifique pour voir si c'est elle qui échoue
    try:
        tokens = word_tokenize(text.lower())
    except LookupError as e_tok:
         # Essayer de télécharger la ressource manquante si c'est une LookupError NLTK
         if 'Resource' in str(e_tok) and 'not found' in str(e_tok):
              resource_name = str(e_tok).split('Resource ')[1].split(' ')[0]
              print(f"--- Tokenizer a besoin de '{resource_name}', tentative de téléchargement ---")
              try:
                  nltk.download(resource_name, quiet=True)
                  print(f"--- Ressource '{resource_name}' téléchargée, nouvelle tentative de tokenisation ---")
                  tokens = word_tokenize(text.lower()) # Retenter après téléchargement
              except Exception as e_dl_tok:
                  print(f"--- Échec du téléchargement de '{resource_name}': {e_dl_tok} ---")
                  raise e_tok # Relancer l'erreur originale si le téléchargement échoue
         else:
              raise e_tok # Relancer si ce n'est pas une ressource manquante connue
    except Exception as e_tok_other:
         print(f"Erreur inattendue dans word_tokenize: {e_tok_other}")
         raise e_tok_other

    filtered_tokens = [lemmatizer_obj_global.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words_set_global]
    return ' '.join(filtered_tokens)
print("  Fonction preprocess_text définie.")

# --- Partie 8: Parsing des Topics ---
print("\n[9/9] Parsing des topics...")
# S'assurer que re et glob sont importés
import re
import glob
def parse_topics(file_path):
    topics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
                topic_content = top_match.group(1)
                num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE)
                if not num_match: continue
                topic_id = num_match.group(1).strip()
                title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
                desc = desc_match.group(1).strip() if desc_match else ""
                if topic_id and title:
                     topics[topic_id] = {'title': title, 'desc': desc}
    except FileNotFoundError:
        print(f"  ATTENTION: Fichier topic non trouvé: {file_path}")
    except Exception as e_topic:
        print(f"  ATTENTION: Erreur lors du parsing de {file_path}: {e_topic}")
    return topics

if not os.path.exists(TOPICS_DIR):
     print(f"ATTENTION: Le dossier des topics '{TOPICS_DIR}' n'existe pas.")
     topic_files = []
else:
    topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))

all_topics = {}
if not topic_files:
     print(f"  ATTENTION: Aucun fichier topic trouvé dans {TOPICS_DIR}")
else:
    for tf in topic_files:
        all_topics.update(parse_topics(tf))

# Définir les dictionnaires même s'ils sont vides pour éviter NameError plus tard
# Mettre la création des dictionnaires prétraités dans un try-except au cas où preprocess_text échouerait encore
try:
    queries_short = {qid: data['title'] for qid, data in all_topics.items()}
    queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()}
    print(f"  {len(all_topics)} topics parsés.")
    print(f"  {len(queries_short)} requêtes courtes brutes créées.")
    print(f"  Prétraitement des requêtes...")
    queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
    queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}
    print(f"  Prétraitement des requêtes terminé.")
except Exception as e_preproc_queries:
     print(f"\nERREUR lors du prétraitement des requêtes: {e_preproc_queries}")
     print("Les dictionnaires prétraités pourraient être incomplets ou vides.")
     # Créer des dictionnaires vides pour éviter NameError plus tard
     queries_short_preprocessed = {}
     queries_long_preprocessed = {}


# --- Vérification Finale Java ---
print("\n--- Vérification Finale de la Version Java Active ---")
java_check_cmd = "java -version"
try:
    result = subprocess.run(java_check_cmd, shell=True, check=True, capture_output=True, text=True, timeout=10)
    print("Sortie STDERR (contient souvent la version OpenJDK):\n", result.stderr)
    if "21." not in result.stderr and "21." not in result.stdout:
         print("\nATTENTION: Java 21 ne semble PAS être la version active !")
    else:
         print("\nConfirmation: Java 21 semble être la version active.")
except Exception as e:
    print(f"\nERREUR lors de la vérification Java: {e}")

# --- Vérification Finale Pyserini ---
print("\n--- Vérification Finale de la Version Pyserini Installée ---")
try:
    result_pyserini = subprocess.run(f"{sys.executable} -m pip show pyserini", shell=True, check=True, capture_output=True, text=True, timeout=30)
    print(result_pyserini.stdout)
except Exception as e_pyserini_check:
    print(f"ERREUR lors de la vérification de Pyserini: {e_pyserini_check}")

print("\n--- Configuration Complète Terminée ---")
# Ajouter un délai pour s'assurer que tout est stable avant la prochaine cellule
print("\nPause de 5 secondes...")
time.sleep(5)
print("Prêt pour la suite.")

# === Cellule 0.4 (Modifiée): Extraire, Décompresser et Formater les Documents ===
import tarfile
import re
import json
import gzip # Importer le module gzip
from tqdm.notebook import tqdm
import os
import traceback

# Chemins définis dans la cellule précédente (full_setup_code_punkt_tab_fixed)
# AP_TAR_PATH devrait pointer vers ".../AP.tar"
# CORPUS_DIR devrait être défini

JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction, Décompression et Formatage depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# Vérifier si le fichier AP.tar existe
if not os.path.exists(AP_TAR_PATH):
    raise FileNotFoundError(f"Le fichier d'archive {AP_TAR_PATH} n'a pas été trouvé.")
else:
    tar_size = os.path.getsize(AP_TAR_PATH)
    print(f"  Taille du fichier {AP_TAR_PATH}: {tar_size} octets.")

# Regex (inchangées)
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

doc_count = 0
file_read_count = 0
skipped_members = 0
decompression_errors = 0

# Ouvrir/créer le fichier JSONL de sortie
try:
    # Utiliser encoding='utf-8' pour l'écriture
    with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:
        members = tar.getmembers()
        print(f"\n{len(members)} membres trouvés dans l'archive TAR.")

        for member in tqdm(members, desc="Traitement des fichiers TAR"):
            # Traiter seulement les fichiers se terminant par .gz ou .Z (typique pour TREC)
            # Ignorer les dossiers ou les fichiers non réguliers
            if not member.isfile() or not member.name.lower().endswith(('.gz', '.z')):
                skipped_members += 1
                continue

            file_read_count += 1
            content = "" # Initialiser content

            try:
                f = tar.extractfile(member)
                if f:
                    compressed_content = f.read()
                    f.close()

                    # --- AJOUT : Décompression Gzip ---
                    try:
                        # Décompresser le contenu lu
                        content_bytes = gzip.decompress(compressed_content)
                        # Décoder en texte APRES décompression
                        content = content_bytes.decode('utf-8', errors='ignore')
                    except gzip.BadGzipFile:
                        # print(f"Avertissement: Fichier {member.name} n'est pas un fichier gzip valide, tentative de lecture directe.")
                        # Essayer de décoder directement si ce n'était pas du gzip
                        content = compressed_content.decode('utf-8', errors='ignore')
                        decompression_errors += 1
                    except Exception as e_gzip:
                         print(f"\nErreur de décompression pour {member.name}: {e_gzip}")
                         decompression_errors += 1
                         continue # Passer au fichier suivant si la décompression échoue
                    # --- FIN AJOUT ---

                    # Chercher les documents dans le contenu décompressé et décodé
                    doc_matches = doc_pattern.findall(content)
                    if not doc_matches:
                         # Si aucun <DOC> trouvé, passer au membre suivant
                         continue

                    for doc_content in doc_matches:
                        docno_match = docno_pattern.search(doc_content)
                        if not docno_match:
                            continue
                        doc_id = docno_match.group(1).strip()

                        text_match = text_pattern.search(doc_content)
                        if text_match:
                           doc_text = text_match.group(1).strip()
                           doc_text = ' '.join(doc_text.split())
                        else:
                            doc_text = ""

                        try:
                            json_line = json.dumps({"id": str(doc_id), "contents": str(doc_text)})
                            outfile.write(json_line + '\n')
                            doc_count += 1
                        except Exception as e_write:
                            print(f"Erreur lors de l'écriture JSON pour doc_id {doc_id}: {e_write}")

            except KeyError as e_key:
                print(f"\nAvertissement: Membre '{member.name}' inaccessible (KeyError): {e_key}")
                skipped_members += 1
            except EOFError:
                 print(f"\nAvertissement: Fin de fichier inattendue lors de la lecture de {member.name}.")
                 skipped_members += 1
            except Exception as e_extract:
                print(f"\nErreur inattendue lors de l'extraction/lecture du membre {member.name}: {e_extract}")
                skipped_members += 1

except tarfile.ReadError as e_tar:
    print(f"\nERREUR: Impossible de lire le fichier TAR {AP_TAR_PATH}. Erreur: {e_tar}")
    raise e_tar
except FileNotFoundError:
     print(f"\nERREUR: Le fichier TAR {AP_TAR_PATH} n'a pas été trouvé.")
     raise FileNotFoundError
except Exception as e_general:
     print(f"\nERREUR générale lors du traitement: {e_general}")
     traceback.print_exc()
     raise e_general

print(f"\n--- Fin de l'Extraction et Décompression ---")
print(f"  {file_read_count} fichiers (.gz/.Z) lus depuis l'archive.")
print(f"  {skipped_members} membres ignorés.")
if decompression_errors > 0:
    print(f"  {decompression_errors} erreurs ou avertissements de décompression rencontrés.")
print(f"  {doc_count} documents au total ont été formatés et écrits dans {JSONL_OUTPUT_PATH}")

if doc_count == 0 and file_read_count > 0:
     print("\n*** PROBLEME MAJEUR: Aucun document n'a été extrait ! Vérifiez les regex ou la structure interne des fichiers décompressés. ***")
elif doc_count < 100000 and file_read_count > 0:
     print("\n  ATTENTION: Le nombre de documents extraits semble faible.")

# Vérifier la taille du fichier de sortie
if os.path.exists(JSONL_OUTPUT_PATH):
    output_size = os.path.getsize(JSONL_OUTPUT_PATH)
    print(f"  Taille finale de {JSONL_OUTPUT_PATH}: {output_size} octets.")
    if output_size == 0 and doc_count == 0:
        print("  CONFIRMATION: Le fichier de sortie est vide.")
    elif output_size > 0 and doc_count > 0:
         print("  SUCCÈS: Le fichier de sortie contient des données.")

# === Cellule 1.2: Indexation Baseline ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment dans la cellule de configuration complète
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl") # Fichier source (maintenant non vide)
# INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline") # Dossier cible
# CORPUS_DIR contient le fichier JSONL

# S'assurer que les variables sont définies (au cas où)
try:
    CORPUS_DIR
    INDEX_DIR_BASELINE
except NameError:
    print("ERREUR: Les variables CORPUS_DIR ou INDEX_DIR_BASELINE ne sont pas définies. Ré-exécutez la cellule de configuration.")
    # Optionnel: redéfinir ici, mais moins propre
    # OUTPUT_DIR = "/content/ap_output"
    # CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
    # INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "indexes/baseline")
    raise

print(f"Début de l'indexation Baseline (sans prétraitement explicite)...")
# Pyserini utilise le dossier CORPUS_DIR comme entrée pour JsonCollection
print(f"Dossier source contenant ap_docs.jsonl: {CORPUS_DIR}")
print(f"Répertoire de l'index cible: {INDEX_DIR_BASELINE}")

# Vérifier si le fichier source existe et n'est pas vide
jsonl_source_path = os.path.join(CORPUS_DIR, "ap_docs.jsonl")
if not os.path.exists(jsonl_source_path) or os.path.getsize(jsonl_source_path) == 0:
     raise FileNotFoundError(f"Le fichier source {jsonl_source_path} est manquant ou vide. L'étape d'extraction ('extract_code_tar_gzip_fixed') a peut-être échoué ou n'a pas été exécutée.")

# Commande Pyserini pour l'indexation
# Utilise la dernière version de Pyserini installée
index_cmd_baseline = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR,
    "--index", INDEX_DIR_BASELINE,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté (ex: 2 ou 8 selon les ressources Colab)
    "--storePositions", "--storeDocvectors", "--storeRaw" # Options utiles pour certaines techniques avancées
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_baseline)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion des erreurs/sorties
    # Augmentation possible du timeout si l'indexation est très longue
    result = subprocess.run(index_cmd_baseline, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    # Vérifier si la sortie indique un nombre non nul de documents indexés
    if "Total 0 documents indexed" in result.stdout:
         print("\nATTENTION: Pyserini indique que 0 document a été indexé malgré un fichier source non vide. Problème potentiel.")
    else:
         print(f"\nIndexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Baseline a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    raise e # Arrêter si l'indexation échoue
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Baseline a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Baseline: {e}")
    traceback.print_exc()
    raise e

# Vérification finale de l'index (taille)
print(f"\nVérification de la taille de l'index créé dans {INDEX_DIR_BASELINE}...")
if os.path.exists(INDEX_DIR_BASELINE):
    # Commande pour obtenir la taille totale du dossier
    du_cmd = f"du -sh '{INDEX_DIR_BASELINE}'"
    try:
        result_du = subprocess.run(du_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  Taille de l'index: {result_du.stdout.split()[0]}")
    except Exception as e_du:
        print(f"  Impossible de vérifier la taille de l'index: {e_du}")
else:
    print("  ATTENTION: Le dossier de l'index n'a pas été créé.")

# === Cellule 1.3: Préparer les Données Prétraitées ===
import json
from tqdm.notebook import tqdm
import os
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment
# JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl") # Fichier source (non vide)
# CORPUS_DIR

# S'assurer que les variables sont définies
try:
    CORPUS_DIR
    JSONL_OUTPUT_PATH
except NameError:
    print("ERREUR: Les variables CORPUS_DIR ou JSONL_OUTPUT_PATH ne sont pas définies. Ré-exécutez la cellule de configuration.")
    raise

JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl")

print(f"Préparation des données prétraitées depuis {JSONL_OUTPUT_PATH} vers {JSONL_PREPROC_PATH}...")

# S'assurer que la fonction preprocess_text est définie (normalement fait dans la cellule de setup)
if 'preprocess_text' not in globals():
    print("Erreur: La fonction 'preprocess_text' n'est pas définie. Ré-exécutez la cellule de configuration.")
    raise NameError("preprocess_text non définie")
else:
    doc_count_preproc = 0
    error_count = 0
    # Lire le fichier JSONL original et écrire le fichier prétraité
    try:
        # Utiliser utf-8 pour lire et écrire
        with open(JSONL_OUTPUT_PATH, 'r', encoding='utf-8') as infile, \
             open(JSONL_PREPROC_PATH, 'w', encoding='utf-8') as outfile:

            # Itérer sur le fichier d'entrée
            # Utiliser tqdm pour la barre de progression
            for line in tqdm(infile, desc="Prétraitement des documents"):
                try:
                    data = json.loads(line)
                    # Utiliser .get pour la robustesse si 'id' ou 'contents' manque
                    doc_id = data.get('id', None)
                    original_contents = data.get('contents', '')

                    if doc_id is None:
                        error_count += 1
                        continue

                    # Appliquer le prétraitement
                    preprocessed_contents = preprocess_text(original_contents)

                    # Écrire la nouvelle ligne JSONL
                    json_line = json.dumps({"id": str(doc_id), "contents": str(preprocessed_contents)})
                    outfile.write(json_line + '\n')
                    doc_count_preproc += 1

                except json.JSONDecodeError:
                    # print(f"Avertissement: Erreur de décodage JSON sur une ligne, ignorée.")
                    error_count += 1
                except Exception as e_line:
                    print(f"\nErreur inattendue lors du prétraitement d'une ligne (id={data.get('id', 'inconnu')}): {e_line}")
                    error_count += 1

        print(f"\nTerminé.")
        print(f"  {doc_count_preproc} documents prétraités et écrits dans {JSONL_PREPROC_PATH}")
        if error_count > 0:
             print(f"  {error_count} lignes ignorées à cause d'erreurs.")

        # Vérifier la taille du fichier de sortie
        if os.path.exists(JSONL_PREPROC_PATH):
            output_size = os.path.getsize(JSONL_PREPROC_PATH)
            print(f"  Taille finale de {JSONL_PREPROC_PATH}: {output_size} octets.")
            if output_size == 0 and doc_count_preproc > 0:
                 print("  ATTENTION: 0 octet écrit malgré le traitement de documents. Problème ?")
        else:
            print(f"  ATTENTION: Le fichier de sortie {JSONL_PREPROC_PATH} n'a pas été créé.")


    except FileNotFoundError:
        print(f"ERREUR: Le fichier d'entrée {JSONL_OUTPUT_PATH} n'a pas été trouvé.")
        raise
    except Exception as e_main:
        print(f"ERREUR générale lors de la préparation des données prétraitées: {e_main}")
        traceback.print_exc()
        raise

# === Cellule 1.4: Indexation Avec Prétraitement ===
import os # Assurer que os est importé
import subprocess # Pour exécuter la commande pyserini
import traceback # Pour afficher les erreurs détaillées

# Chemins définis précédemment
# JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl") # Fichier source
# INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed") # Dossier cible pour l'index
# CORPUS_DIR contient le fichier JSONL prétraité

# S'assurer que les variables sont définies
try:
    CORPUS_DIR
    INDEX_DIR_PREPROC
except NameError:
    print("ERREUR: Les variables CORPUS_DIR ou INDEX_DIR_PREPROC ne sont pas définies. Ré-exécutez la cellule de configuration.")
    # Optionnel: redéfinir ici, mais moins propre
    # OUTPUT_DIR = "/content/ap_output"
    # CORPUS_DIR = os.path.join(OUTPUT_DIR, "corpus")
    # INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "indexes/preprocessed")
    raise

JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl") # Chemin complet pour vérification

print(f"Début de l'indexation avec Prétraitement...")
# Note: Pyserini s'attend à un dossier en entrée pour JsonCollection,
# il trouvera ap_docs_preprocessed.jsonl dans CORPUS_DIR.
print(f"Collection source (dossier): {CORPUS_DIR}")
print(f"Fichier JSONL prétraité attendu: {JSONL_PREPROC_PATH}")
print(f"Répertoire de l'index cible: {INDEX_DIR_PREPROC}")

# Vérifier si le fichier prétraité existe et n'est pas vide
if not os.path.exists(JSONL_PREPROC_PATH) or os.path.getsize(JSONL_PREPROC_PATH) == 0:
    raise FileNotFoundError(f"Le fichier de données prétraitées {JSONL_PREPROC_PATH} est manquant ou vide. Assurez-vous que l'étape précédente (1.3) s'est bien terminée.")

# Commande Pyserini pour l'indexation prétraitée
index_cmd_preproc = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", CORPUS_DIR, # Pointeur vers le dossier contenant les jsonl
    "--index", INDEX_DIR_PREPROC,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "4", # Peut être ajusté
    "--storePositions", "--storeDocvectors", "--storeRaw",
    "--pretokenized" # Important: Indique que le texte est déjà tokenisé/traité
]

# Exécuter la commande
print(f"Exécution de la commande: {' '.join(index_cmd_preproc)}")
try:
    # Utiliser subprocess.run pour une meilleure gestion
    result = subprocess.run(index_cmd_preproc, check=True, capture_output=True, text=True, timeout=1800) # Timeout 30 minutes
    print("Sortie STDOUT:\n", result.stdout)
    print("Sortie STDERR:\n", result.stderr)
    # Vérifier si la sortie indique un nombre non nul de documents indexés
    if "Total 0 documents indexed" in result.stdout:
         print("\nATTENTION: Pyserini indique que 0 document a été indexé. Problème potentiel avec l'indexation prétraitée.")
    else:
        print(f"\nIndexation avec Prétraitement terminée. Index créé dans {INDEX_DIR_PREPROC}")
except subprocess.CalledProcessError as e:
    print(f"\nERREUR: L'indexation Prétraitée a échoué avec le code {e.returncode}")
    print("Sortie STDOUT:\n", e.stdout)
    print("Sortie STDERR:\n", e.stderr)
    raise e
except subprocess.TimeoutExpired as e:
    print(f"\nERREUR: L'indexation Prétraitée a dépassé le délai d'attente.")
    print("Sortie STDOUT (partielle):\n", e.stdout)
    print("Sortie STDERR (partielle):\n", e.stderr)
    raise e
except Exception as e:
    print(f"\nERREUR inattendue pendant l'indexation Prétraitée: {e}")
    traceback.print_exc()
    raise e

# Vérification finale de l'index (taille)
print(f"\nVérification de la taille de l'index créé dans {INDEX_DIR_PREPROC}...")
if os.path.exists(INDEX_DIR_PREPROC):
    # Commande pour obtenir la taille totale du dossier
    du_cmd = f"du -sh '{INDEX_DIR_PREPROC}'"
    try:
        result_du = subprocess.run(du_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  Taille de l'index: {result_du.stdout.split()[0]}")
    except Exception as e_du:
        print(f"  Impossible de vérifier la taille de l'index: {e_du}")
else:
    print("  ATTENTION: Le dossier de l'index n'a pas été créé.")

# === Cellule 3.1: Exécuter les Recherches (Séquentielles - BM25 & TF-IDF) ===
# Utilise la dernière Pyserini et Java 21 (devraient être actifs)
# S'assurer que les variables d'index et de requêtes sont définies

from pyserini.search.lucene import LuceneSearcher # Import principal
import time
from tqdm.notebook import tqdm # Toujours utile pour la progression
import traceback # Pour afficher les erreurs détaillées
import os # Assurer que os est importé
from jnius import autoclass, JavaException # Importer pour TF-IDF

# Essayer de définir K_RESULTS si ce n'est pas déjà fait
try:
    K_RESULTS
except NameError:
    print("Définition de K_RESULTS (nombre de résultats) à 1000...")
    K_RESULTS = 1000

# --- Configuration des modèles de similarité ---
# Charger la classe Java pour TF-IDF (ClassicSimilarity)
# Mettre dans un try-except au cas où l'import échouerait (peu probable maintenant)
try:
    ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')
    print("Classe ClassicSimilarity (pour TF-IDF) chargée avec succès.")
except JavaException as e_load_class:
    print(f"ERREUR Java lors du chargement de ClassicSimilarity: {e_load_class}")
    print("Les recherches TF-IDF échoueront probablement.")
    ClassicSimilarity = None # Mettre à None pour pouvoir vérifier plus tard
except Exception as e_load_gen:
     print(f"ERREUR inattendue lors du chargement de ClassicSimilarity: {e_load_gen}")
     ClassicSimilarity = None

# Vérifier que les variables nécessaires existent
try:
    INDEX_DIR_BASELINE
    INDEX_DIR_PREPROC
    RUN_DIR
    queries_short
    queries_long
    queries_short_preprocessed
    queries_long_preprocessed
    preprocess_text # Vérifier aussi la fonction
except NameError as e_missing_var:
    print(f"ERREUR: Variable essentielle manquante ({e_missing_var}). L'environnement a peut-être été perdu. Ré-exécutez la cellule de configuration complète.")
    raise e_missing_var


def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}" # Ex: baseline_short_bm25 or baseline_short_tfidf
    print(f"\nDébut recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', PréprocReq={use_preprocessed_query}, k={k}")

    all_results_list = []
    searcher = None # Initialiser à None

    try:
        # Initialiser le searcher UNE SEULE FOIS pour toutes les requêtes de ce run
        print(f"  Initialisation de LuceneSearcher pour {run_tag}...")
        # Assurer que LuceneSearcher est importé
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(index_path)
        print(f"  LuceneSearcher initialisé.")

        # Configurer le modèle de similarité
        if model == 'bm25':
            print("  Configuration de BM25...")
            searcher.set_bm25(k1=0.9, b=0.4)
            print("  BM25 configuré.")
        elif model == 'tfidf':
            if ClassicSimilarity is None:
                 print("ERREUR: Classe ClassicSimilarity non chargée. Impossible de configurer TF-IDF.")
                 print(f"--- ABANDON du run {run_tag} ---")
                 return # Ne pas continuer si la classe n'a pas pu être chargée

            print("  Configuration de ClassicSimilarity (TF-IDF)...")
            try:
                 searcher.set_similarity(ClassicSimilarity())
                 print("  ClassicSimilarity configurée.")
            except JavaException as e:
                 print(f"ERREUR Java lors de la configuration de ClassicSimilarity: {e}")
                 print(traceback.format_exc())
                 print(f"--- ABANDON du run {run_tag} à cause de l'erreur de configuration TF-IDF ---")
                 return
            except Exception as e_other:
                 print(f"ERREUR Inattendue lors de la configuration de ClassicSimilarity: {e_other}")
                 print(traceback.format_exc())
                 print(f"--- ABANDON du run {run_tag} à cause d'une erreur TF-IDF ---")
                 return
        else:
            print(f"Modèle '{model}' non reconnu, utilisation de BM25 par défaut...")
            searcher.set_bm25()
            print("  BM25 par défaut configuré.")

        # Itérer sur les requêtes séquentiellement
        query_errors = 0
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                # S'assurer que preprocess_text est défini
                if 'preprocess_text' not in globals():
                     raise NameError("La fonction preprocess_text n'est pas définie.")

                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                # Vérifier si la requête traitée est vide
                if not search_text.strip():
                     # print(f"  Avertissement: Requête QID {query_id} est vide après traitement, ignorée.")
                     continue # Ignorer les requêtes vides

                hits = searcher.search(search_text, k=k)

                # Formater les résultats pour cette requête
                for i in range(len(hits)):
                    rank = i + 1
                    doc_id = hits[i].docid
                    score = hits[i].score
                    # S'assurer que doc_id n'est pas None (peut arriver dans de rares cas)
                    if doc_id is None:
                        # print(f"  Avertissement: Doc ID est None pour QID {query_id} au rang {rank}, ignoré.")
                        continue
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            except Exception as e_query:
                # Compter les erreurs par requête mais continuer
                query_errors += 1
                if query_errors < 10: # Limiter l'affichage des erreurs par requête
                     print(f"\nErreur lors de la recherche pour QID {query_id} avec {run_tag}: {e_query}")
                elif query_errors == 10:
                     print("\nPlusieurs erreurs de recherche pour ce run, messages suivants masqués...")


        # Écrire les résultats dans le fichier de run TREC
        if all_results_list:
             # Utiliser encoding='utf-8' pour l'écriture
             with open(output_run_file, 'w', encoding='utf-8') as f_out:
                f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes de résultats écrites.")
        else:
            print("\n  Avertissement: Aucun résultat généré pour ce run.")

        if query_errors > 0:
            print(f"  Avertissement: {query_errors} erreurs rencontrées lors de la recherche sur les requêtes individuelles.")

        end_time = time.time()
        print(f"Recherche SÉQUENTIELLE terminée pour {run_tag}. Résultats sauvegardés dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")

    except Exception as e_main:
        # Erreur pendant l'initialisation du searcher ou configuration BM25
        print(f"\nERREUR MAJEURE pendant l'exécution de {run_tag}: {e_main}")
        print(traceback.format_exc())
    finally:
        # En théorie, Pyserini/jnius gère la fermeture de la JVM, pas besoin de fermer le searcher explicitement
        if searcher:
             print(f"  Nettoyage implicite des ressources pour {run_tag}.")
             pass


# --- Exécution des 8 configurations de recherche (Séquentiel) ---

print("\n--- DÉBUT DES RECHERCHES BASELINE ---")
# 1. Index Baseline + Requêtes Courtes + BM25
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")

# 2. Index Baseline + Requêtes Courtes + TF-IDF
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")

# 3. Index Baseline + Requêtes Longues + BM25
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")

# 4. Index Baseline + Requêtes Longues + TF-IDF
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")

print("\n--- DÉBUT DES RECHERCHES PRÉTRAITÉES ---")
# 5. Index Preprocessed + Requêtes Courtes (Prétraitées) + BM25
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)

# 6. Index Preprocessed + Requêtes Courtes (Prétraitées) + TF-IDF
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)

# 7. Index Preprocessed + Requêtes Longues (Prétraitées) + BM25
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)

# 8. Index Preprocessed + Requêtes Longues (Prétraitées) + TF-IDF
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)

print("\n--- Toutes les recherches de base (mode séquentiel) sont terminées. ---")

# === Cellule 7: Exécuter la Recherche Améliorée (RM3) ===
# Applique RM3 sur la meilleure configuration de base identifiée à l'étape 6.
# !! N'OUBLIEZ PAS DE CONFIGURER LES VARIABLES BEST_... CI-DESSOUS !!

from pyserini.search.lucene import LuceneSearcher
from jnius import autoclass, JavaException
from tqdm.notebook import tqdm
import time
import traceback
import os

# Recharger ClassicSimilarity au cas où
try: ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')
except Exception: ClassicSimilarity = None

# Vérifier variables nécessaires
try:
    INDEX_DIR_BASELINE; INDEX_DIR_PREPROC; RUN_DIR; K_RESULTS; EVAL_DIR;
    queries_short; queries_long; queries_short_preprocessed; queries_long_preprocessed;
    preprocess_text;
except NameError as e: print(f"ERREUR: Variable {e} manquante. Exécutez config complète."); raise

# --- À CONFIGURER selon vos meilleurs résultats de l'Étape 6 ---
# !! MODIFIEZ CECI EN FONCTION DE VOS RÉSULTATS D'ÉVALUATION !!
print("--- Configuration RM3 ---")
print("Veuillez éditer les variables BEST_... ci-dessous en fonction de vos meilleurs résultats MAP de l'étape précédente.")
# Exemple: si preproc + long + bm25 était le meilleur
BEST_INDEX_PATH = INDEX_DIR_PREPROC           # Ex: INDEX_DIR_BASELINE ou INDEX_DIR_PREPROC
BEST_QUERIES = queries_long_preprocessed      # Ex: queries_short, queries_long, ..._preprocessed
BEST_MODEL_BASE = 'bm25'                      # Ex: 'bm25' ou 'tfidf'
BEST_RUN_TAG_PREFIX = "preproc_long"          # Ex: 'baseline_short', 'preproc_long'
USE_PREPROC_QUERY_FOR_RM3 = False             # Généralement False si BEST_QUERIES est déjà prétraité
# ----------------------------------------------------------------
print(f"Configuration choisie pour RM3:")
print(f"  Index: {os.path.basename(BEST_INDEX_PATH)}")
# print(f"  Requêtes: (variable BEST_QUERIES)") # Difficile d'afficher le nom de la variable
print(f"  Modèle Base: {BEST_MODEL_BASE}")
print(f"  Préfixe Tag: {BEST_RUN_TAG_PREFIX}")
print(f"  Utiliser Preproc Requête?: {USE_PREPROC_QUERY_FOR_RM3}")

# Nom du fichier et tag pour le run RM3
PRF_RUN_FILE = os.path.join(RUN_DIR, f"{BEST_RUN_TAG_PREFIX}_{BEST_MODEL_BASE}_rm3.txt")
RM3_RUN_TAG = f"{BEST_RUN_TAG_PREFIX}_{BEST_MODEL_BASE}_rm3"

# Paramètres RM3
rm3_config = {'fb_terms': 10, 'fb_docs': 10, 'original_query_weight': 0.5}
print(f"  Paramètres RM3: {rm3_config}")

# --- Fonction de recherche RM3 (séquentielle) ---
# (Définition identique à celle de search_code_final, on peut la réutiliser si elle est dans la portée)
# Par sécurité, on la redéfinit ici au cas où l'utilisateur n'exécute que cette cellule après setup.
def perform_search_sequential_rm3(queries, index_path, model_base, k, output_run_file, run_tag, use_preprocessed_query=False, rm3_params=None):
    """Exécute la recherche RM3 séquentiellement."""
    start_time = time.time()
    print(f"\nDébut recherche SÉQUENTIELLE RM3: Modèle='{model_base}+RM3', Tag='{run_tag}', k={k}")
    all_results_list = []
    searcher = None
    try:
        print(f"  Initialisation LuceneSearcher..."); searcher = LuceneSearcher(index_path); print(f"  LuceneSearcher initialisé.")
        if model_base == 'bm25': print("  Config BM25 (base)..."); searcher.set_bm25(k1=0.9, b=0.4)
        elif model_base == 'tfidf':
            if ClassicSimilarity is None: raise ValueError("ClassicSimilarity non chargée.")
            print("  Config ClassicSimilarity (base)...")
            try: searcher.set_similarity(ClassicSimilarity())
            except Exception as e_sim: print(f"ERREUR config ClassicSimilarity: {e_sim}"); return
        else: print(f"Modèle base '{model_base}' non reconnu, utilise BM25."); searcher.set_bm25()
        print("  Activation RM3..."); searcher.set_rm3(**rm3_params); print("  RM3 activé.")
        query_errors = 0
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                if 'preprocess_text' not in globals(): raise NameError("preprocess_text non définie.")
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                if not search_text.strip(): continue
                hits = searcher.search(search_text, k=k)
                for i in range(len(hits)):
                    rank, doc_id, score = i + 1, hits[i].docid, hits[i].score
                    if doc_id is None: continue
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
            except Exception as e_query:
                query_errors += 1
                if query_errors < 5: print(f"\nErreur recherche RM3 QID {query_id}: {e_query}")
                elif query_errors == 5: print("\nPlusieurs erreurs recherche RM3...")
        if all_results_list:
             with open(output_run_file, 'w', encoding='utf-8') as f_out: f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes résultats écrites.")
        else: print("\n  Avertissement: Aucun résultat RM3 généré.")
        if query_errors > 0: print(f"  Avertissement: {query_errors} erreurs sur requêtes.")
        end_time = time.time()
        print(f"Recherche RM3 terminée pour {run_tag}. Sauvegardé dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")
    except Exception as e_main: print(f"\nERREUR MAJEURE run RM3 {run_tag}: {e_main}"); traceback.print_exc()
    finally:
        if searcher: print(f"  Nettoyage implicite ressources {run_tag}.")

# Lancer la recherche RM3 (après configuration des variables BEST_...)
print("\nLancement de la recherche RM3...")
perform_search_sequential_rm3(
    BEST_QUERIES, BEST_INDEX_PATH, BEST_MODEL_BASE, K_RESULTS,
    PRF_RUN_FILE, RM3_RUN_TAG,
    use_preprocessed_query=USE_PREPROC_QUERY_FOR_RM3, rm3_params=rm3_config
)

print("\n--- Exécution de la recherche RM3 terminée. ---")

# === Cellule 5: Exécuter les Recherches (Séquentielles - BM25 & TF-IDF) ===
# Lance les 8 combinaisons de recherche et sauvegarde les résultats.
# Assurez-vous que l'environnement Java 21 est toujours actif.
# Assurez-vous que les index existent et que les variables sont définies.

from pyserini.search.lucene import LuceneSearcher # Import principal
import time
from tqdm.notebook import tqdm
import traceback
import os
from jnius import autoclass, JavaException # Pour TF-IDF

# Définir K_RESULTS
try: K_RESULTS
except NameError: print("Définition K_RESULTS=1000"); K_RESULTS = 1000

# Charger ClassicSimilarity
try: ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity'); print("ClassicSimilarity chargée.")
except Exception as e: print(f"ERREUR chargement ClassicSimilarity: {e}"); ClassicSimilarity = None

# Vérifier variables nécessaires
try:
    INDEX_DIR_BASELINE; INDEX_DIR_PREPROC; RUN_DIR; K_RESULTS;
    queries_short; queries_long; queries_short_preprocessed; queries_long_preprocessed;
    preprocess_text;
    # Vérifier aussi l'existence des index
    if not os.path.exists(INDEX_DIR_BASELINE): raise FileNotFoundError(f"Index Baseline manquant: {INDEX_DIR_BASELINE}")
    if not os.path.exists(INDEX_DIR_PREPROC): raise FileNotFoundError(f"Index Preprocessed manquant: {INDEX_DIR_PREPROC}")
except NameError as e: print(f"ERREUR: Variable {e} manquante. Exécutez config complète."); raise
except FileNotFoundError as e: print(f"ERREUR: {e}"); raise

def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}"
    print(f"\nDébut recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', k={k}")

    all_results_list = []
    searcher = None

    try:
        print(f"  Initialisation LuceneSearcher..."); searcher = LuceneSearcher(index_path); print(f"  LuceneSearcher initialisé.")
        if model == 'bm25': print("  Config BM25..."); searcher.set_bm25(k1=0.9, b=0.4); print("  BM25 configuré.")
        elif model == 'tfidf':
            if ClassicSimilarity is None: print("ERREUR: ClassicSimilarity non chargée. ABANDON."); return
            print("  Config ClassicSimilarity (TF-IDF)...")
            try: searcher.set_similarity(ClassicSimilarity()); print("  ClassicSimilarity configurée.")
            except Exception as e_sim: print(f"ERREUR config ClassicSimilarity: {e_sim}"); return
        else: print(f"Modèle '{model}' non reconnu, utilise BM25."); searcher.set_bm25()

        query_errors = 0
        # S'assurer que preprocess_text est défini avant la boucle
        if 'preprocess_text' not in globals(): raise NameError("preprocess_text non définie.")

        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                if not search_text.strip(): continue # Ignorer requêtes vides

                hits = searcher.search(search_text, k=k)

                for i in range(len(hits)):
                    rank, doc_id, score = i + 1, hits[i].docid, hits[i].score
                    if doc_id is None: continue
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
            except Exception as e_query:
                query_errors += 1
                if query_errors < 5: print(f"\nErreur recherche QID {query_id}: {e_query}")
                elif query_errors == 5: print("\nPlusieurs erreurs recherche...")

        # Écrire résultats
        if all_results_list:
             # Créer le dossier RUN_DIR si besoin (normalement fait par setup)
             os.makedirs(os.path.dirname(output_run_file), exist_ok=True)
             with open(output_run_file, 'w', encoding='utf-8') as f_out: f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes résultats écrites dans {os.path.basename(output_run_file)}.")
        else: print("\n  Avertissement: Aucun résultat généré pour ce run.")
        if query_errors > 0: print(f"  Avertissement: {query_errors} erreurs sur requêtes.")

        end_time = time.time()
        print(f"Recherche terminée pour {run_tag}. Sauvegardé dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")
    except Exception as e_main: print(f"\nERREUR MAJEURE run {run_tag}: {e_main}"); traceback.print_exc()
    finally:
        if searcher: print(f"  Nettoyage implicite ressources {run_tag}.")

# --- Exécution des 8 configurations ---
print("\n--- DÉBUT DES RECHERCHES BASELINE ---")
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt"); perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt"); perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt"); perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt"); perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")
print("\n--- DÉBUT DES RECHERCHES PRÉTRAITÉES ---")
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt"); perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt"); perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt"); perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt"); perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)
print("\n--- Toutes les recherches de base (mode séquentiel) sont terminées. ---")

# Vérifier si des fichiers ont été créés
print(f"\nVérification du contenu de {RUN_DIR} après les recherches...")
!ls -l {RUN_DIR}

# === Cellule 5: Exécuter les Recherches (Séquentielles - BM25 & TF-IDF) ===
# Lance les 8 combinaisons de recherche et sauvegarde les résultats.
# Assurez-vous que l'environnement Java 21 est toujours actif.
# Assurez-vous que les index existent et que les variables sont définies.

from pyserini.search.lucene import LuceneSearcher # Import principal
import time
from tqdm.notebook import tqdm
import traceback
import os
from jnius import autoclass, JavaException # Pour TF-IDF

# Définir K_RESULTS
try: K_RESULTS
except NameError: print("Définition K_RESULTS=1000"); K_RESULTS = 1000

# Charger ClassicSimilarity
try: ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity'); print("ClassicSimilarity chargée.")
except Exception as e: print(f"ERREUR chargement ClassicSimilarity: {e}"); ClassicSimilarity = None

# Vérifier variables nécessaires
try:
    INDEX_DIR_BASELINE; INDEX_DIR_PREPROC; RUN_DIR; K_RESULTS;
    queries_short; queries_long; queries_short_preprocessed; queries_long_preprocessed;
    preprocess_text;
    # Vérifier aussi l'existence des index
    if not os.path.exists(INDEX_DIR_BASELINE): raise FileNotFoundError(f"Index Baseline manquant: {INDEX_DIR_BASELINE}")
    if not os.path.exists(INDEX_DIR_PREPROC): raise FileNotFoundError(f"Index Preprocessed manquant: {INDEX_DIR_PREPROC}")
except NameError as e: print(f"ERREUR: Variable {e} manquante. Exécutez config complète."); raise
except FileNotFoundError as e: print(f"ERREUR: {e}"); raise

def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}"
    print(f"\nDébut recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', k={k}")

    all_results_list = []
    searcher = None

    try:
        print(f"  Initialisation LuceneSearcher..."); searcher = LuceneSearcher(index_path); print(f"  LuceneSearcher initialisé.")
        if model == 'bm25': print("  Config BM25..."); searcher.set_bm25(k1=0.9, b=0.4); print("  BM25 configuré.")
        elif model == 'tfidf':
            if ClassicSimilarity is None: print("ERREUR: ClassicSimilarity non chargée. ABANDON."); return
            print("  Config ClassicSimilarity (TF-IDF)...")
            try: searcher.set_similarity(ClassicSimilarity()); print("  ClassicSimilarity configurée.")
            except Exception as e_sim: print(f"ERREUR config ClassicSimilarity: {e_sim}"); return
        else: print(f"Modèle '{model}' non reconnu, utilise BM25."); searcher.set_bm25()

        query_errors = 0
        # S'assurer que preprocess_text est défini avant la boucle
        if 'preprocess_text' not in globals(): raise NameError("preprocess_text non définie.")

        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                if not search_text.strip(): continue # Ignorer requêtes vides

                hits = searcher.search(search_text, k=k)

                for i in range(len(hits)):
                    rank, doc_id, score = i + 1, hits[i].docid, hits[i].score
                    if doc_id is None: continue
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
            except Exception as e_query:
                query_errors += 1
                if query_errors < 5: print(f"\nErreur recherche QID {query_id}: {e_query}")
                elif query_errors == 5: print("\nPlusieurs erreurs recherche...")

        # Écrire résultats
        if all_results_list:
             # Créer le dossier RUN_DIR si besoin (normalement fait par setup)
             os.makedirs(os.path.dirname(output_run_file), exist_ok=True)
             with open(output_run_file, 'w', encoding='utf-8') as f_out: f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes résultats écrites dans {os.path.basename(output_run_file)}.")
        else: print("\n  Avertissement: Aucun résultat généré pour ce run.")
        if query_errors > 0: print(f"  Avertissement: {query_errors} erreurs sur requêtes.")

        end_time = time.time()
        print(f"Recherche terminée pour {run_tag}. Sauvegardé dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")
    except Exception as e_main: print(f"\nERREUR MAJEURE run {run_tag}: {e_main}"); traceback.print_exc()
    finally:
        if searcher: print(f"  Nettoyage implicite ressources {run_tag}.")

# --- Exécution des 8 configurations ---
print("\n--- DÉBUT DES RECHERCHES BASELINE ---")
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt"); perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt"); perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt"); perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt"); perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")
print("\n--- DÉBUT DES RECHERCHES PRÉTRAITÉES ---")
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt"); perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt"); perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt"); perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt"); perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)
print("\n--- Toutes les recherches de base (mode séquentiel) sont terminées. ---")

# Vérifier si des fichiers ont été créés
print(f"\nVérification du contenu de {RUN_DIR} après les recherches...")
!ls -l {RUN_DIR}

OUTPUT_DIR = "/content/ap_output"
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")

# Chemin exact où les résultats de recherche sont attendus
RUN_DIR_PATH="/content/ap_output/runs/"

echo "Vérification du contenu de : ${RUN_DIR_PATH}"
ls -l ${RUN_DIR_PATH}

# === Cellule de Vérification du Contenu du Dossier Runs ===
# Utilise les commandes shell de Colab préfixées par '!'

# Chemin exact où les résultats de recherche sont attendus
# (Défini dans la cellule de configuration complète)
RUN_DIR_PATH="/content/ap_output/runs/"

# Utiliser '!' pour exécuter la commande shell 'echo'
print(f"Vérification du contenu de : {RUN_DIR_PATH}")

# Utiliser '!' pour exécuter la commande shell 'ls -l'
# Mettre le chemin entre guillemets pour gérer les espaces potentiels (même s'il n'y en a pas ici)
!ls -lh "{RUN_DIR_PATH}"

# === Cellule 2: Restauration des fichiers depuis Google Drive (Tout Inclus) ===
import os
import subprocess
import time

# Chemin où les fichiers ont été sauvegardés sur Drive
try: DRIVE_PROJECT_PATH # Défini dans la cellule précédente
except NameError: print("ERREUR: DRIVE_PROJECT_PATH non défini. Exécutez config complète."); raise

DRIVE_BACKUP_DIR = os.path.join(DRIVE_PROJECT_PATH, "colab_output_backup")

# Chemin cible dans Colab
TARGET_RESTORE_DIR = "/content/ap_output" # = OUTPUT_DIR défini précédemment

print(f"Source sur Drive : {DRIVE_BACKUP_DIR}")
print(f"Cible dans Colab : {TARGET_RESTORE_DIR}")

# Vérifier si le dossier de sauvegarde existe
if os.path.exists(DRIVE_BACKUP_DIR):
    os.makedirs(TARGET_RESTORE_DIR, exist_ok=True) # Créer dossier cible si besoin

    print("\nRestauration des fichiers (corpus et index) en cours... (Peut prendre plusieurs minutes)")
    # Commande de copie récursive
    copy_cmd = f"cp -r -v '{DRIVE_BACKUP_DIR}/.' '{TARGET_RESTORE_DIR}/'"
    try:
        process = subprocess.run(copy_cmd, shell=True, check=True, capture_output=True, text=True, timeout=900) # Timeout 15 minutes pour les index
        print("\nRestauration terminée avec succès !")
        print(f"Les fichiers de {DRIVE_BACKUP_DIR} ont été copiés dans {TARGET_RESTORE_DIR}")
        # Vérifier le contenu restauré (y compris les index)
        print("\nContenu du dossier restauré (partiel):")
        !ls -l {TARGET_RESTORE_DIR}
        print("\nContenu du dossier indexes (restauré):")
        !ls -l {TARGET_RESTORE_DIR}/indexes
    except subprocess.CalledProcessError as e:
         print(f"\nERREUR restauration (code {e.returncode}). Vérifiez si backup existe et contient corpus/, indexes/baseline/, indexes/preprocessed/.")
         print("STDERR:", e.stderr); raise e
    except Exception as e: print(f"\nERREUR restauration: {e}"); raise e
else:
    print(f"ERREUR: Dossier sauvegarde {DRIVE_BACKUP_DIR} inexistant.")
    print("Impossible de restaurer. Il faut relancer extraction et indexations.")
    raise FileNotFoundError(f"Dossier sauvegarde non trouvé: {DRIVE_BACKUP_DIR}")

# === Cellule 4: Exécuter les Recherches (Séquentielles - BM25 & QLD) ===
# Lance les 8 combinaisons de recherche en utilisant BM25 et QLD.
# S'assure que l'environnement Java 21 est actif et que les index/variables sont définis/restaurés.

from pyserini.search.lucene import LuceneSearcher # Import principal
import time
from tqdm.notebook import tqdm
import traceback
import os
from jnius import JavaException # Importer seulement JavaException, ClassicSimilarity n'est pas utilisé

# Définir K_RESULTS
try: K_RESULTS
except NameError: print("Définition K_RESULTS=1000"); K_RESULTS = 1000

# Vérifier variables nécessaires et existence des index restaurés
try:
    INDEX_DIR_BASELINE; INDEX_DIR_PREPROC; RUN_DIR; K_RESULTS;
    queries_short; queries_long; queries_short_preprocessed; queries_long_preprocessed;
    preprocess_text;
    if not os.path.exists(INDEX_DIR_BASELINE): raise FileNotFoundError(f"Index Baseline restauré manquant: {INDEX_DIR_BASELINE}")
    if not os.path.exists(INDEX_DIR_PREPROC): raise FileNotFoundError(f"Index Preprocessed restauré manquant: {INDEX_DIR_PREPROC}")
    # Vérifier aussi que les fichiers de corpus sont là (restaurés)
    if not os.path.exists(os.path.join(CORPUS_DIR, "ap_docs.jsonl")): raise FileNotFoundError("ap_docs.jsonl manquant après restauration.")
    if not os.path.exists(os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl")): raise FileNotFoundError("ap_docs_preprocessed.jsonl manquant après restauration.")

except NameError as e: print(f"ERREUR: Variable {e} manquante. Exécutez config complète."); raise
except FileNotFoundError as e: print(f"ERREUR: {e}"); raise

def perform_search_sequential_qld(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes (BM25 ou QLD)."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}"
    print(f"\nDébut recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', k={k}")

    all_results_list = []
    searcher = None

    try:
        print(f"  Initialisation LuceneSearcher..."); searcher = LuceneSearcher(index_path); print(f"  LuceneSearcher initialisé.")

        # Configurer similarité
        if model == 'bm25':
            print("  Configuration BM25..."); searcher.set_bm25(k1=0.9, b=0.4); print("  BM25 configuré.")
        elif model == 'qld': # Utiliser Query Likelihood Dirichlet
            print("  Configuration QLD..."); searcher.set_qld(); print("  QLD configuré.")
        else:
            print(f"Modèle '{model}' non reconnu, utilise BM25 par défaut."); searcher.set_bm25()

        # Itérer sur les requêtes
        query_errors = 0
        if 'preprocess_text' not in globals(): raise NameError("preprocess_text non définie.")

        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                if not search_text.strip(): continue # Ignorer requêtes vides

                hits = searcher.search(search_text, k=k)

                for i in range(len(hits)):
                    rank, doc_id, score = i + 1, hits[i].docid, hits[i].score
                    if doc_id is None: continue
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
            except Exception as e_query:
                query_errors += 1
                if query_errors < 5: print(f"\nErreur recherche QID {query_id}: {e_query}")
                elif query_errors == 5: print("\nPlusieurs erreurs recherche...")

        # Écrire résultats
        if all_results_list:
             os.makedirs(os.path.dirname(output_run_file), exist_ok=True)
             with open(output_run_file, 'w', encoding='utf-8') as f_out: f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes résultats écrites dans {os.path.basename(output_run_file)}.")
        else: print("\n  Avertissement: Aucun résultat généré pour ce run.")
        if query_errors > 0: print(f"  Avertissement: {query_errors} erreurs sur requêtes.")

        end_time = time.time()
        print(f"Recherche terminée pour {run_tag}. Sauvegardé dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")
    except Exception as e_main: print(f"\nERREUR MAJEURE run {run_tag}: {e_main}"); traceback.print_exc()
    finally:
        if searcher: print(f"  Nettoyage implicite ressources {run_tag}.")

# --- Exécution des 8 configurations (BM25 et QLD) ---
print("\n--- DÉBUT DES RECHERCHES BASELINE (BM25/QLD) ---")
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt"); perform_search_sequential_qld(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")
run_file_2 = os.path.join(RUN_DIR, "baseline_short_qld.txt"); perform_search_sequential_qld(queries_short, INDEX_DIR_BASELINE, 'qld', K_RESULTS, run_file_2, "baseline_short") # Utilise qld
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt"); perform_search_sequential_qld(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")
run_file_4 = os.path.join(RUN_DIR, "baseline_long_qld.txt"); perform_search_sequential_qld(queries_long, INDEX_DIR_BASELINE, 'qld', K_RESULTS, run_file_4, "baseline_long") # Utilise qld
print("\n--- DÉBUT DES RECHERCHES PRÉTRAITÉES (BM25/QLD) ---")
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt"); perform_search_sequential_qld(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)
run_file_6 = os.path.join(RUN_DIR, "preproc_short_qld.txt"); perform_search_sequential_qld(queries_short_preprocessed, INDEX_DIR_PREPROC, 'qld', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False) # Utilise qld
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt"); perform_search_sequential_qld(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)
run_file_8 = os.path.join(RUN_DIR, "preproc_long_qld.txt"); perform_search_sequential_qld(queries_long_preprocessed, INDEX_DIR_PREPROC, 'qld', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False) # Utilise qld
print("\n--- Toutes les recherches de base (BM25/QLD) sont terminées. ---")

# Vérifier si des fichiers ont été créés
print(f"\nVérification du contenu de {RUN_DIR} après les recherches...")
!ls -l {RUN_DIR}

# === Cellule 6: Évaluation des Runs (BM25/QLD) ===
# Lit les fichiers Qrels, lit les fichiers de résultats (.txt) du dossier RUN_DIR,
# calcule MAP et P@10, et affiche/sauvegarde les tableaux récapitulatifs.
# Devrait maintenant évaluer les runs BM25 et QLD.

import pandas as pd
import glob
import pytrec_eval
import os
import traceback

# Vérifier que les chemins sont définis
try:
    QRELS_DIR
    RUN_DIR
    EVAL_DIR
except NameError:
    print("ERREUR: Variables de chemin non définies. Exécutez la cellule de configuration complète.")
    raise

print(f"Préparation des Qrels depuis: {QRELS_DIR}")
qrels_files = sorted(glob.glob(os.path.join(QRELS_DIR, "qrels.*.txt")))
if not qrels_files: print(f"ATTENTION: Aucun fichier Qrels trouvé dans {QRELS_DIR}."); qrels_dict = {}
else:
    print(f"Fichiers Qrels trouvés: {qrels_files}")
    all_qrels_data = []
    for qf in qrels_files:
        try:
            # Lire le fichier qrels en spécifiant les types pour éviter les erreurs
            qrels_df = pd.read_csv(qf, sep='\s+', names=['query_id', 'unused', 'doc_id', 'relevance'],
                                   dtype={'query_id': str, 'unused': str, 'doc_id': str, 'relevance': int})
            all_qrels_data.append(qrels_df[['query_id', 'doc_id', 'relevance']])
        except Exception as e: print(f"Erreur lecture Qrels {qf}: {e}")
    if not all_qrels_data: print("ERREUR: Impossible lire données Qrels."); qrels_dict = {}
    else:
        combined_qrels_df = pd.concat(all_qrels_data, ignore_index=True)
        qrels_dict = {}
        # Convertir le DataFrame en dictionnaire attendu par pytrec_eval
        for _, row in combined_qrels_df.iterrows():
            qid, did, rel = str(row['query_id']), str(row['doc_id']), int(row['relevance'])
            if rel < 0: continue # Ignorer jugements négatifs
            if qid not in qrels_dict: qrels_dict[qid] = {}
            qrels_dict[qid][did] = rel
        print(f"Total {len(qrels_dict)} requêtes avec jugements chargées.")

# --- Évaluation des Runs ---
if not qrels_dict: print("\nAucun jugement de pertinence chargé, impossible d'évaluer.")
else:
    measures = {'map', 'P_10'} # Métriques à calculer
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, measures) # Initialiser l'évaluateur
    # Trouver tous les fichiers .txt dans le dossier des runs
    run_files = sorted(glob.glob(os.path.join(RUN_DIR, "*.txt")))
    print(f"\n{len(run_files)} fichiers de run à évaluer trouvés dans {RUN_DIR}.")
    print(f"  Fichiers: {[os.path.basename(f) for f in run_files]}") # Afficher les noms

    results_summary = [] # Liste pour stocker les résultats agrégés
    if not run_files: print(f"ATTENTION: Aucun fichier de run (.txt) trouvé dans {RUN_DIR}.")
    else:
        # Boucler sur chaque fichier de run trouvé
        for run_file in run_files:
            run_name = os.path.basename(run_file)
            print(f"\n--- Évaluation: {run_name} ---")
            run_dict = {} # Dictionnaire pour stocker les résultats de ce run
            error_count = 0
            line_count = 0
            try:
                # Lire le fichier run ligne par ligne
                with open(run_file, 'r', encoding='utf-8') as f_run:
                    for line in f_run:
                        line_count += 1
                        parts = line.strip().split()
                        # Vérifier le format TREC (6 colonnes)
                        if len(parts) != 6: error_count += 1; continue
                        qid, _, did, _, score, _ = parts # Extraire les infos utiles
                        try: score = float(score) # Convertir le score en float
                        except ValueError: error_count += 1; continue
                        qid = str(qid) # Assurer que qid est une chaîne
                        # Stocker le score pour ce document et cette requête
                        if qid not in run_dict: run_dict[qid] = {}
                        run_dict[qid][did] = score
                if error_count > 0: print(f"  Avertissement: {error_count} lignes mal formatées ignorées sur {line_count} lignes.")

                # Filtrer le run pour ne garder que les requêtes présentes dans les Qrels
                filtered_run_dict = {qid: docs for qid, docs in run_dict.items() if qid in qrels_dict}
                ignored_q = len(run_dict) - len(filtered_run_dict)
                if ignored_q > 0: print(f"  Avertissement: {ignored_q} requêtes run ignorées (absentes Qrels).")
                if not filtered_run_dict: print("  Erreur: Aucune requête ne correspond aux Qrels."); continue

                # Évaluer le run filtré avec pytrec_eval
                eval_results = evaluator.evaluate(filtered_run_dict)
                # Calculer les moyennes des métriques sur toutes les requêtes évaluées
                all_maps = [q_res.get("map", 0) for q_res in eval_results.values()]
                all_p10s = [q_res.get("P_10", 0) for q_res in eval_results.values()]
                avg_map = sum(all_maps) / len(all_maps) if all_maps else 0
                avg_p10 = sum(all_p10s) / len(all_p10s) if all_p10s else 0

                # Afficher les résultats moyens pour ce run
                print(f"  MAP: {avg_map:.4f}")
                print(f"  P@10: {avg_p10:.4f}")

                # Extraire les informations du nom de fichier pour le résumé
                parts = run_name.replace('.txt','').split('_')
                if len(parts) >= 3:
                    index_type, query_type, model_type = parts[0], parts[1], parts[2]
                    # Gérer le tag RM3 s'il est présent (pour l'évaluation finale)
                    if len(parts) > 3 and parts[-1] == 'rm3':
                         model_type = "_".join(parts[2:]) # Ex: BM25_RM3 ou QLD_RM3
                    else:
                         model_type = "_".join(parts[2:]) # Ex: BM25 ou QLD

                    # Ajouter les résultats au résumé
                    results_summary.append({
                        "Run Name": run_name, "Index": index_type,
                        "Query Type": query_type.capitalize(),
                        "Weighting Scheme": model_type.upper().replace('_', '+'), # Formatage pour affichage
                        "MAP": avg_map, "P@10": avg_p10
                    })
                else: print(f"  Avertissement: Impossible parser nom run '{run_name}'.")

            except FileNotFoundError: print(f"  Erreur: Fichier run non trouvé: {run_file}")
            except Exception as e: print(f"  Erreur évaluation {run_name}: {e}"); traceback.print_exc()

        # Afficher et sauvegarder le résumé final
        if results_summary:
            print("\n\n=== Tableau Récapitulatif des Résultats (BM25/QLD) ===")
            results_df = pd.DataFrame(results_summary)
            # Trier pour une meilleure lisibilité
            results_df = results_df.sort_values(by=["Index", "Query Type", "Weighting Scheme"])

            # Afficher le DataFrame complet
            print("\n--- Résultats Complets ---")
            print(results_df.to_markdown(index=False, floatfmt=".4f"))

            # Essayer d'afficher les tableaux pivots
            try:
                pivot_map = results_df.pivot_table(index=['Query Type', 'Weighting Scheme'], columns='Index', values='MAP')
                print("\n--- MAP (Tableau Pivot) ---")
                print(pivot_map.to_markdown(floatfmt=".4f"))
            except Exception as e_pivot: print(f"\n(Erreur création tableau pivot MAP: {e_pivot})")

            try:
                pivot_p10 = results_df.pivot_table(index=['Query Type', 'Weighting Scheme'], columns='Index', values='P@10')
                print("\n--- P@10 (Tableau Pivot) ---")
                print(pivot_p10.to_markdown(floatfmt=".4f"))
            except Exception as e_pivot: print(f"\n(Erreur création tableau pivot P@10: {e_pivot})")

            # Sauvegarder le DataFrame complet final
            summary_file_path = os.path.join(EVAL_DIR, "evaluation_summary_final.csv")
            try:
                 results_df.to_csv(summary_file_path, index=False)
                 print(f"\nTableau récapitulatif complet sauvegardé: {summary_file_path}")
            except Exception as e_save: print(f"\nErreur sauvegarde résumé: {e_save}")
        else: print("\nAucun résultat d'évaluation à afficher.")

# === Cellule 7: Exécuter la Recherche Améliorée (RM3) ===
# Applique RM3 sur la meilleure configuration de base identifiée à l'étape 6.
# !! N'OUBLIEZ PAS DE CONFIGURER LES VARIABLES BEST_... CI-DESSOUS !!

from pyserini.search.lucene import LuceneSearcher
from jnius import autoclass, JavaException
from tqdm.notebook import tqdm
import time
import traceback
import os

# Recharger ClassicSimilarity n'est plus nécessaire car on utilise BM25/QLD
# try: ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')
# except Exception: ClassicSimilarity = None

# Vérifier variables nécessaires
try:
    INDEX_DIR_BASELINE; INDEX_DIR_PREPROC; RUN_DIR; K_RESULTS; EVAL_DIR;
    queries_short; queries_long; queries_short_preprocessed; queries_long_preprocessed;
    preprocess_text;
except NameError as e: print(f"ERREUR: Variable {e} manquante. Exécutez config complète."); raise

# --- À CONFIGURER selon vos meilleurs résultats de l'Étape 6 (BM25/QLD) ---
# !! MODIFIEZ CECI EN FONCTION DE VOS RÉSULTATS D'ÉVALUATION !!
print("--- Configuration RM3 ---")
print("Veuillez éditer les variables BEST_... ci-dessous en fonction de vos meilleurs résultats MAP de l'étape précédente.")
# Exemple: si preproc + long + bm25 était le meilleur
BEST_INDEX_PATH = INDEX_DIR_PREPROC           # Ex: INDEX_DIR_BASELINE ou INDEX_DIR_PREPROC
BEST_QUERIES = queries_long_preprocessed      # Ex: queries_short, queries_long, ..._preprocessed
BEST_MODEL_BASE = 'bm25'                      # Ex: 'bm25' ou 'qld' (celui qui a donné le meilleur MAP)
BEST_RUN_TAG_PREFIX = "preproc_long"          # Ex: 'baseline_short', 'preproc_long'
USE_PREPROC_QUERY_FOR_RM3 = False             # Généralement False si BEST_QUERIES est déjà prétraité
# ----------------------------------------------------------------
print(f"Configuration choisie pour RM3:")
print(f"  Index: {os.path.basename(BEST_INDEX_PATH)}")
# print(f"  Requêtes: (variable BEST_QUERIES)") # Difficile d'afficher le nom de la variable
print(f"  Modèle Base: {BEST_MODEL_BASE}")
print(f"  Préfixe Tag: {BEST_RUN_TAG_PREFIX}")
print(f"  Utiliser Preproc Requête?: {USE_PREPROC_QUERY_FOR_RM3}")

# Nom du fichier et tag pour le run RM3
PRF_RUN_FILE = os.path.join(RUN_DIR, f"{BEST_RUN_TAG_PREFIX}_{BEST_MODEL_BASE}_rm3.txt")
RM3_RUN_TAG = f"{BEST_RUN_TAG_PREFIX}_{BEST_MODEL_BASE}_rm3"

# Paramètres RM3
rm3_config = {'fb_terms': 10, 'fb_docs': 10, 'original_query_weight': 0.5}
print(f"  Paramètres RM3: {rm3_config}")

# --- Fonction de recherche RM3 (séquentielle) ---
def perform_search_sequential_rm3(queries, index_path, model_base, k, output_run_file, run_tag, use_preprocessed_query=False, rm3_params=None):
    """Exécute la recherche RM3 séquentiellement."""
    start_time = time.time()
    print(f"\nDébut recherche SÉQUENTIELLE RM3: Modèle='{model_base}+RM3', Tag='{run_tag}', k={k}")
    all_results_list = []
    searcher = None
    try:
        print(f"  Initialisation LuceneSearcher..."); searcher = LuceneSearcher(index_path); print(f"  LuceneSearcher initialisé.")
        # Configurer similarité base
        if model_base == 'bm25': print("  Config BM25 (base)..."); searcher.set_bm25(k1=0.9, b=0.4)
        elif model_base == 'qld': print("  Config QLD (base)..."); searcher.set_qld()
        else: print(f"Modèle base '{model_base}' non reconnu, utilise BM25."); searcher.set_bm25()
        # Activer RM3
        print("  Activation RM3..."); searcher.set_rm3(**rm3_params); print("  RM3 activé.")
        # Itérer sur requêtes
        query_errors = 0
        if 'preprocess_text' not in globals(): raise NameError("preprocess_text non définie.")
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                if not search_text.strip(): continue
                hits = searcher.search(search_text, k=k)
                for i in range(len(hits)):
                    rank, doc_id, score = i + 1, hits[i].docid, hits[i].score
                    if doc_id is None: continue
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
            except Exception as e_query:
                query_errors += 1
                if query_errors < 5: print(f"\nErreur recherche RM3 QID {query_id}: {e_query}")
                elif query_errors == 5: print("\nPlusieurs erreurs recherche RM3...")
        # Écrire résultats
        if all_results_list:
             os.makedirs(os.path.dirname(output_run_file), exist_ok=True) # Assurer que le dossier existe
             with open(output_run_file, 'w', encoding='utf-8') as f_out: f_out.writelines(all_results_list)
             print(f"\n  {len(all_results_list)} lignes résultats écrites dans {os.path.basename(output_run_file)}.")
        else: print("\n  Avertissement: Aucun résultat RM3 généré.")
        if query_errors > 0: print(f"  Avertissement: {query_errors} erreurs sur requêtes.")
        end_time = time.time()
        print(f"Recherche RM3 terminée pour {run_tag}. Sauvegardé dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.")
    except Exception as e_main: print(f"\nERREUR MAJEURE run RM3 {run_tag}: {e_main}"); traceback.print_exc()
    finally:
        if searcher: print(f"  Nettoyage implicite ressources {run_tag}.")

# Lancer la recherche RM3 (après configuration des variables BEST_...)
print("\nLancement de la recherche RM3...")
perform_search_sequential_rm3(
    BEST_QUERIES, BEST_INDEX_PATH, BEST_MODEL_BASE, K_RESULTS,
    PRF_RUN_FILE, RM3_RUN_TAG,
    use_preprocessed_query=USE_PREPROC_QUERY_FOR_RM3, rm3_params=rm3_config
)

print("\n--- Exécution de la recherche RM3 terminée. ---")
# Vérifier si le fichier a été créé
print(f"\nVérification de la création du fichier {PRF_RUN_FILE}...")
!ls -l "{PRF_RUN_FILE}"

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered)

!pip install pysolr

# === Cellule 0.2: Installation des bibliothèques ===
# Pyserini nécessite Java 11, installons-le
!apt-get update -qq > /dev/null && apt-get install -y openjdk-11-jdk-headless -qq > /dev/null

# Installer Pyserini, NLTK et Pytrec_eval
!pip install pyserini==0.24.0 -q # Installe une version spécifique pour la stabilité
!pip install nltk -q
!pip install pytrec_eval -q

# Définir la variable d'environnement JAVA_HOME
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Télécharger les ressources NLTK nécessaires
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) # Ajouté pour WordNet

print("Installation terminée et ressources NLTK téléchargées.")

# === Cellule 0.3: Définir les chemins ===
# !!! ADAPTEZ CE CHEMIN VERS VOTRE DOSSIER SUR GOOGLE DRIVE !!!
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Projet_RI"

# Vérification que le chemin existe
if not os.path.exists(DRIVE_PROJECT_PATH):
    raise FileNotFoundError(f"Le chemin spécifié n'existe pas : {DRIVE_PROJECT_PATH}. Vérifiez le chemin dans la Cellule 0.1 et 0.3.")

AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, "/content/drive/MyDrive/Projet_RI/AP.tar") # Assumant que c'est un .tar.gz, sinon ajustez
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "/content/drive/MyDrive/Projet_RI/topics/")
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "/content/drive/MyDrive/Projet_RI/ql/")

# Chemins pour les sorties (index, résultats, etc.) dans l'environnement Colab
OUTPUT_DIR = "/content/drive/MyDrive/Projet_RI/output/"
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "/content/drive/MyDrive/Projet_RI/baseline")
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "/content/drive/MyDrive/Projet_RI/pre")
CORPUS_DIR = os.path.join(OUTPUT_DIR, "/content/drive/MyDrive/Projet_RI/Corpus") # Pour les documents extraits/formatés
RUN_DIR = os.path.join(OUTPUT_DIR, "/content/drive/MyDrive/Projet_RI/runs") # Pour les fichiers de résultats TREC
EVAL_DIR = os.path.join(OUTPUT_DIR, "/content/drive/MyDrive/Projet_RI/eval") # Pour les fichiers d'évaluation

# Créer les répertoires de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"Chemin du projet Drive: {DRIVE_PROJECT_PATH}")
print(f"Répertoire de sortie Colab: {OUTPUT_DIR}")

# === Cellule 0.4: Extraire et Formater les Documents ===
import tarfile
import re
import json
from tqdm.notebook import tqdm # Barre de progression

# Chemin vers le fichier JSONL qui sera généré
JSONL_OUTPUT_PATH = os.path.join(CORPUS_DIR, "ap_docs.jsonl")

print(f"Extraction et formatage des documents depuis {AP_TAR_PATH} vers {JSONL_OUTPUT_PATH}...")

# Regex pour extraire DOCNO et TEXT
doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

# Compteur pour vérifier
doc_count = 0

# Ouvrir/créer le fichier JSONL de sortie
with open(JSONL_OUTPUT_PATH, 'w') as outfile, tarfile.open(AP_TAR_PATH, "r") as tar:  # Changed mode to "r" # Assurez-vous que c'est bien .gz
    # Itérer sur chaque membre (fichier/dossier) dans l'archive tar
    for member in tqdm(tar.getmembers(), desc="Traitement des fichiers TAR"):
        # Vérifier si c'est un fichier régulier
        if member.isfile():
            # Extraire le contenu du fichier
            f = tar.extractfile(member)
            if f: # S'assurer que l'extraction a réussi
                content = f.read().decode('utf-8', errors='ignore') # Lire et décoder

                # Trouver tous les documents dans le fichier actuel
                for doc_match in doc_pattern.finditer(content):
                    doc_content = doc_match.group(1)

                    # Extraire DOCNO
                    docno_match = docno_pattern.search(doc_content)
                    if not docno_match:
                        continue # Passer si pas de DOCNO
                    doc_id = docno_match.group(1).strip()

                    # Extraire TEXT (et le nettoyer un peu)
                    text_match = text_pattern.search(doc_content)
                    if text_match:
                       doc_text = text_match.group(1).strip()
                       # Nettoyage simple: remplacer les nouvelles lignes par des espaces
                       doc_text = ' '.join(doc_text.split())
                    else:
                        doc_text = "" # Mettre une chaîne vide si pas de champ TEXT

                    # Écrire l'entrée JSONL
                    json_line = json.dumps({"id": doc_id, "contents": doc_text})
                    outfile.write(json_line + '\n')
                    doc_count += 1

print(f"Terminé. {doc_count} documents formatés dans {JSONL_OUTPUT_PATH}")
# Note: La collection AP88-90 contient environ 164 597 documents. Vérifiez si ce nombre est proche.
# Si AP.tar.gz contient des sous-dossiers (ap88, ap89, etc.), ce code devrait fonctionner.
# Si AP.tar.gz contient directement les fichiers ap88xxxx, cela fonctionnera aussi.
# Si c'est juste AP.tar (non compressé), changez "r:gz" en "r:"

# === Cellule 1.1: Fonction de Prétraitement ===
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Applique la tokenisation, la mise en minuscule, la suppression
    de la ponctuation, la suppression des stop words et la lemmatisation.
    """
    # Tokenisation et minuscules
    tokens = word_tokenize(text.lower())

    # Suppression ponctuation et mots non alphabétiques + stop words
    filtered_tokens = [
        lemmatizer.lemmatize(w) for w in tokens
        if w.isalpha() and w not in stop_words # Garde seulement les mots alphabétiques non-stop words
    ]

    # Rejoint les tokens en une chaîne de caractères
    return ' '.join(filtered_tokens)

# Exemple d'utilisation
sample_text = "This is an example showing Information Retrieval with lemmatization and stop words removal."
preprocessed_sample = preprocess_text(sample_text)
print(f"Original: {sample_text}")
print(f"Preprocessed: {preprocessed_sample}")

# === Cellule 1.2: Indexation Baseline ===
print(f"Début de l'indexation Baseline (sans prétraitement explicite)...")
print(f"Collection source: {JSONL_OUTPUT_PATH}")
print(f"Répertoire de l'index: {INDEX_DIR_BASELINE}")

# Commande Pyserini pour l'indexation
# -input: dossier contenant les fichiers JSONL
# -collection: type de collection (JsonCollection pour nos fichiers .jsonl)
# -generator: comment traiter les fichiers (LuceneDocumentGenerator crée un document par ligne JSON)
# -index: chemin où sauvegarder l'index
# -threads: nombre de threads à utiliser (ajustez selon les ressources Colab, 4 est raisonnable)
# -storePositions -storeDocvectors -storeRaw: stocke informations supplémentaires utiles pour certaines recherches avancées (comme le re-ranking ou PRF)
!python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input {CORPUS_DIR} \
  --index {INDEX_DIR_BASELINE} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw

print(f"Indexation Baseline terminée. Index créé dans {INDEX_DIR_BASELINE}")

# === Cellule 1.3: Préparer les Données Prétraitées ===
JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl")

print(f"Préparation des données prétraitées vers {JSONL_PREPROC_PATH}...")

doc_count_preproc = 0
# Lire le fichier JSONL original et écrire le fichier prétraité
with open(JSONL_OUTPUT_PATH, 'r') as infile, open(JSONL_PREPROC_PATH, 'w') as outfile:
    for line in tqdm(infile, desc="Prétraitement des documents"):
        try:
            data = json.loads(line)
            doc_id = data['id']
            original_contents = data['contents']

            # Appliquer le prétraitement
            preprocessed_contents = preprocess_text(original_contents)

            # Écrire la nouvelle ligne JSONL
            json_line = json.dumps({"id": doc_id, "contents": preprocessed_contents})
            outfile.write(json_line + '\n')
            doc_count_preproc += 1
        except json.JSONDecodeError:
            print(f"Erreur de décodage JSON sur une ligne, ignorée.") # Au cas où une ligne serait malformée
        except Exception as e:
            print(f"Erreur inattendue lors du prétraitement: {e}") # Autres erreurs possibles

print(f"Terminé. {doc_count_preproc} documents prétraités dans {JSONL_PREPROC_PATH}")

# === Cellule 1.3: Préparer les Données Prétraitées ===
JSONL_PREPROC_PATH = os.path.join(CORPUS_DIR, "ap_docs_preprocessed.jsonl")

print(f"Préparation des données prétraitées vers {JSONL_PREPROC_PATH}...")

doc_count_preproc = 0
# Lire le fichier JSONL original et écrire le fichier prétraité
with open(JSONL_OUTPUT_PATH, 'r') as infile, open(JSONL_PREPROC_PATH, 'w') as outfile:
    for line in tqdm(infile, desc="Prétraitement des documents"):
        try:
            data = json.loads(line)
            doc_id = data['id']
            original_contents = data['contents']

            # Appliquer le prétraitement
            preprocessed_contents = preprocess_text(original_contents)

            # Écrire la nouvelle ligne JSONL
            json_line = json.dumps({"id": doc_id, "contents": preprocessed_contents})
            outfile.write(json_line + '\n')
            doc_count_preproc += 1
        except json.JSONDecodeError:
            print(f"Erreur de décodage JSON sur une ligne, ignorée.") # Au cas où une ligne serait malformée
        except Exception as e:
            print(f"Erreur inattendue lors du prétraitement: {e}") # Autres erreurs possibles

print(f"Terminé. {doc_count_preproc} documents prétraités dans {JSONL_PREPROC_PATH}")

# === Cellule 1.4: Indexation Avec Prétraitement ===
print(f"Début de l'indexation avec Prétraitement...")
print(f"Collection source: {JSONL_PREPROC_PATH}") # Utilise le fichier .jsonl prétraité
print(f"Répertoire de l'index: {INDEX_DIR_PREPROC}")

# La commande est identique, mais pointe vers le fichier JSONL prétraité
!python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input {CORPUS_DIR} \
  --index {INDEX_DIR_PREPROC} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw \
  --pretokenized # Important: Indique que le texte est déjà tokenisé (évite une re-tokenisation par Lucene)

print(f"Indexation avec Prétraitement terminée. Index créé dans {INDEX_DIR_PREPROC}")

# === Cellule 2.1: Parser les Fichiers Topics ===
import glob # Pour trouver les fichiers correspondant à un pattern

def parse_topics(file_path):
    """Parse un fichier topic TREC standard."""
    topics = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Utilise regex pour trouver chaque bloc <top>
        for top_match in re.finditer(r"<top>(.*?)</top>", content, re.DOTALL):
            topic_content = top_match.group(1)
            # Extrait le numéro (num)
            num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic_content, re.IGNORECASE)
            if not num_match: continue
            topic_id = num_match.group(1).strip()

            # Extrait le titre (title) - prend tout après <title> jusqu'au prochain tag
            title_match = re.search(r"<title>\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""

            # Extrait la description (desc)
            desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
            desc = desc_match.group(1).strip() if desc_match else ""

            # Extrait la narrative (narr) - pas utilisée ici mais pourrait l'être
            # narr_match = re.search(r"<narr>\s*Narrative:\s*(.*?)\s*(?=<|$)", topic_content, re.IGNORECASE | re.DOTALL)
            # narr = narr_match.group(1).strip() if narr_match else ""

            if topic_id and title: # Au moins un ID et un titre
                 topics[topic_id] = {'title': title, 'desc': desc}
    return topics

# Trouver tous les fichiers topics
topic_files = sorted(glob.glob(os.path.join(TOPICS_DIR, "topics.*.txt")))
print(f"Fichiers topics trouvés: {topic_files}")

all_topics = {}
for tf in topic_files:
    print(f"Parsing {tf}...")
    all_topics.update(parse_topics(tf))

print(f"Total de {len(all_topics)} topics parsés.")

# Créer les dictionnaires de requêtes courtes et longues
queries_short = {qid: data['title'] for qid, data in all_topics.items()}
queries_long = {qid: data['title'] + " " + data['desc'] for qid, data in all_topics.items()} # Concatène titre et description

# Optionnel: Créer des versions prétraitées des requêtes
queries_short_preprocessed = {qid: preprocess_text(q) for qid, q in queries_short.items()}
queries_long_preprocessed = {qid: preprocess_text(q) for qid, q in queries_long.items()}

print(f"Exemple Requête Courte (ID 51): {queries_short.get('51', 'Non trouvé')}")
print(f"Exemple Requête Longue (ID 51): {queries_long.get('51', 'Non trouvé')}")
print(f"Exemple Requête Courte Prétraitée (ID 51): {queries_short_preprocessed.get('51', 'Non trouvé')}")
print(f"Exemple Requête Longue Prétraitée (ID 51): {queries_long_preprocessed.get('51', 'Non trouvé')}")

# === Cellule 3.1: Fonction de Recherche et Sauvegarde ===
from pyserini.search.lucene import LuceneSearcher
import time
from multiprocessing import Pool, cpu_count

# --- Configuration des modèles de similarité ---
# Pyserini/Lucene utilise BM25 par défaut (avec k1=0.9, b=0.4)
# Pour TF-IDF, nous utilisons ClassicSimilarity de Lucene.
# Cela nécessite d'importer la classe Java via Pyjnius (le pont Python-Java de Pyserini)
from jnius import autoclass
ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')

def perform_search_single_query(args):
    """Fonction exécutée par chaque processus pour une seule requête."""
    query_id, query_text, index_path, model, k, run_tag, use_preprocessed_query = args

    try:
        # Initialiser le searcher DANS le processus fils
        searcher = LuceneSearcher(index_path)

        # Configurer le modèle de similarité
        if model == 'bm25':
            # Utiliser les valeurs par défaut de Pyserini ou spécifier les vôtres
            searcher.set_bm25(k1=0.9, b=0.4) # Valeurs standard BM25 TREC
        elif model == 'tfidf':
            searcher.set_similarity(ClassicSimilarity()) # Appliquer TF-IDF (ClassicSimilarity)
        else:
            # Par défaut ou erreur
            searcher.set_bm25() # Rétablir BM25 par sécurité

        # Prétraiter la requête si nécessaire (pour l'index prétraité)
        search_text = preprocess_text(query_text) if use_preprocessed_query else query_text

        # Exécuter la recherche
        hits = searcher.search(search_text, k=k)

        # Formater les résultats pour cette requête
        query_results = []
        for i in range(len(hits)):
            rank = i + 1
            doc_id = hits[i].docid
            score = hits[i].score
            # Format TREC: qid Q0 docid rank score run_tag
            query_results.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

        return query_results

    except Exception as e:
        print(f"Erreur lors de la recherche pour QID {query_id} avec {run_tag}: {e}")
        return [] # Retourne une liste vide en cas d'erreur


def run_search_parallel(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche en parallèle pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}" # Ex: baseline_short_bm25
    print(f"Début recherche: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', PréprocReq={use_preprocessed_query}, k={k}")

    # Préparer les arguments pour chaque tâche de processus
    tasks = []
    for query_id, query_text in queries.items():
        tasks.append((query_id, query_text, index_path, model, k, run_tag, use_preprocessed_query))

    # Utiliser un Pool de processus pour la parallélisation
    # Utiliser N-1 coeurs pour laisser un peu de marge, ou cpu_count()
    num_workers = max(1, cpu_count() - 1)
    print(f"Utilisation de {num_workers} processus parallèles...")

    all_results_list = []
    # Utiliser tqdm pour la barre de progression avec le Pool
    with Pool(num_workers) as pool:
       # pool.imap_unordered exécute les tâches et retourne les résultats dès qu'ils sont prêts
       # Cela peut être plus rapide si certaines requêtes prennent plus de temps
       results_iterator = pool.imap_unordered(perform_search_single_query, tasks)
       # Envelopper avec tqdm pour la barre de progression
       for result in tqdm(results_iterator, total=len(tasks), desc=f"Recherche {run_tag}"):
           all_results_list.extend(result) # Ajouter les lignes de résultats retournées par chaque processus


    # Écrire les résultats dans le fichier de run TREC
    with open(output_run_file, 'w') as f_out:
       f_out.writelines(all_results_list)

    end_time = time.time()
    print(f"Recherche terminée pour {run_tag}. Résultats sauvegardés dans {output_run_file}")
    print(f"Temps écoulé: {end_time - start_time:.2f} secondes.\n")


# --- Exécution des différentes configurations ---
K_RESULTS = 1000 # Nombre de documents à retourner par requête (standard TREC)

# 1. Index Baseline + Requêtes Courtes + BM25
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt")
run_search_parallel(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")

# 2. Index Baseline + Requêtes Courtes + TF-IDF
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt")
run_search_parallel(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")

# 3. Index Baseline + Requêtes Longues + BM25
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt")
run_search_parallel(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")

# 4. Index Baseline + Requêtes Longues + TF-IDF
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt")
run_search_parallel(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")

# --- Recherches sur l'index prétraité ---
# Important: Utiliser les requêtes prétraitées correspondantes

# 5. Index Preprocessed + Requêtes Courtes (Prétraitées) + BM25
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt")
run_search_parallel(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)
 # Note: Les requêtes sont déjà prétraitées, donc use_preprocessed_query=False dans la fonction
 #       (car elle applique preprocess_text si True) - c'est un peu contre-intuitif
 #       Alternative: passer `queries_short` et mettre `use_preprocessed_query=True`. Choisissons la première option pour la clarté.

# 6. Index Preprocessed + Requêtes Courtes (Prétraitées) + TF-IDF
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt")
run_search_parallel(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)

# 7. Index Preprocessed + Requêtes Longues (Prétraitées) + BM25
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt")
run_search_parallel(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)

# 8. Index Preprocessed + Requêtes Longues (Prétraitées) + TF-IDF
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt")
run_search_parallel(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)

print("Toutes les recherches de base sont terminées.")

# === Cellule 3.1 (Modifiée): Fonction de Recherche et Sauvegarde (Séquentielle d'abord) ===
from pyserini.search.lucene import LuceneSearcher
import time
from tqdm.notebook import tqdm # Toujours utile pour la progression
import traceback # Pour afficher les erreurs détaillées

# --- Configuration des modèles de similarité ---
from jnius import autoclass, JavaException
ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')

def perform_search_sequential(queries, index_path, model, k, output_run_file, run_tag_prefix, use_preprocessed_query=False):
    """Exécute la recherche séquentiellement pour un ensemble de requêtes."""
    start_time = time.time()
    run_tag = f"{run_tag_prefix}_{model}" # Ex: baseline_short_bm25
    print(f"Début recherche SÉQUENTIELLE: Index='{os.path.basename(index_path)}', Modèle='{model}', Tag='{run_tag}', PréprocReq={use_preprocessed_query}, k={k}")

    all_results_list = []
    searcher = None # Initialiser à None

    try:
        # Initialiser le searcher UNE SEULE FOIS pour toutes les requêtes de ce run
        print(f"  Initialisation de LuceneSearcher pour {run_tag}...")
        searcher = LuceneSearcher(index_path)
        print(f"  LuceneSearcher initialisé.")

        # Configurer le modèle de similarité
        if model == 'bm25':
            print("  Configuration de BM25...")
            searcher.set_bm25(k1=0.9, b=0.4)
            print("  BM25 configuré.")
        elif model == 'tfidf':
            print("  Configuration de ClassicSimilarity (TF-IDF)...")
            try:
                 searcher.set_similarity(ClassicSimilarity())
                 print("  ClassicSimilarity configurée.")
            except JavaException as e:
                 print(f"ERREUR Java lors de la configuration de ClassicSimilarity: {e}")
                 print(traceback.format_exc()) # Affiche la trace complète de l'erreur Java
                 raise # Arrête l'exécution pour ce run si la similarité ne peut être définie
        else:
            print("  Configuration BM25 par défaut...")
            searcher.set_bm25()
            print("  BM25 par défaut configuré.")

        # Itérer sur les requêtes séquentiellement
        for query_id, query_text in tqdm(queries.items(), desc=f"Recherche {run_tag}"):
            try:
                search_text = preprocess_text(query_text) if use_preprocessed_query else query_text
                hits = searcher.search(search_text, k=k)

                # Formater les résultats pour cette requête
                for i in range(len(hits)):
                    rank = i + 1
                    doc_id = hits[i].docid
                    score = hits[i].score
                    all_results_list.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            except Exception as e_query:
                print(f"\nErreur lors de la recherche pour QID {query_id} avec {run_tag}: {e_query}")
                # Continue avec la requête suivante

        # Écrire les résultats dans le fichier de run TREC
        with open(output_run_file, 'w') as f_out:
           f_out.writelines(all_results_list)

        end_time = time.time()
        print(f"Recherche SÉQUENTIELLE terminée pour {run_tag}. Résultats sauvegardés dans {output_run_file}")
        print(f"Temps écoulé: {end_time - start_time:.2f} secondes.\n")

    except Exception as e_main:
        print(f"\nERREUR MAJEURE pendant l'exécution de {run_tag}: {e_main}")
        print(traceback.format_exc()) # Affiche la trace complète de l'erreur
    finally:
        # Important: Fermer le searcher pour libérer les ressources Java, même en cas d'erreur
        if searcher:
             try:
                 # Note: Pyserini ne semble pas avoir de méthode close() explicite sur LuceneSearcher
                 # La JVM devrait se nettoyer, mais c'est une bonne pratique si disponible
                 # searcher.close() # Décommentez si une telle méthode existe dans votre version
                 print(f"  Nettoyage implicite des ressources pour {run_tag}.")
                 pass
             except Exception as e_close:
                 print(f"  Erreur lors de la tentative de fermeture du searcher pour {run_tag}: {e_close}")


# --- Exécution des différentes configurations (en mode séquentiel) ---
K_RESULTS = 1000 # Nombre de documents à retourner par requête

# 1. Index Baseline + Requêtes Courtes + BM25
run_file_1 = os.path.join(RUN_DIR, "baseline_short_bm25.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_1, "baseline_short")

# 2. Index Baseline + Requêtes Courtes + TF-IDF
run_file_2 = os.path.join(RUN_DIR, "baseline_short_tfidf.txt")
perform_search_sequential(queries_short, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_2, "baseline_short")

# 3. Index Baseline + Requêtes Longues + BM25
run_file_3 = os.path.join(RUN_DIR, "baseline_long_bm25.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'bm25', K_RESULTS, run_file_3, "baseline_long")

# 4. Index Baseline + Requêtes Longues + TF-IDF
run_file_4 = os.path.join(RUN_DIR, "baseline_long_tfidf.txt")
perform_search_sequential(queries_long, INDEX_DIR_BASELINE, 'tfidf', K_RESULTS, run_file_4, "baseline_long")

# --- Recherches sur l'index prétraité ---
# 5. Index Preprocessed + Requêtes Courtes (Prétraitées) + BM25
run_file_5 = os.path.join(RUN_DIR, "preproc_short_bm25.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_5, "preproc_short", use_preprocessed_query=False)

# 6. Index Preprocessed + Requêtes Courtes (Prétraitées) + TF-IDF
run_file_6 = os.path.join(RUN_DIR, "preproc_short_tfidf.txt")
perform_search_sequential(queries_short_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_6, "preproc_short", use_preprocessed_query=False)

# 7. Index Preprocessed + Requêtes Longues (Prétraitées) + BM25
run_file_7 = os.path.join(RUN_DIR, "preproc_long_bm25.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'bm25', K_RESULTS, run_file_7, "preproc_long", use_preprocessed_query=False)

# 8. Index Preprocessed + Requêtes Longues (Prétraitées) + TF-IDF
run_file_8 = os.path.join(RUN_DIR, "preproc_long_tfidf.txt")
perform_search_sequential(queries_long_preprocessed, INDEX_DIR_PREPROC, 'tfidf', K_RESULTS, run_file_8, "preproc_long", use_preprocessed_query=False)

print("Toutes les recherches de base (mode séquentiel) sont terminées.")

# --- Note importante ---
# Si cette cellule s'exécute sans planter (même si c'est lent),
# le problème est probablement lié à la parallélisation (mémoire/conflits JVM).
# Si elle plante encore, surtout lors des runs 'tfidf',
# le problème pourrait être lié à ClassicSimilarity ou à l'environnement Java lui-même.

!pip install pyserini

# === Cellule 1.1: Fonction de Prétraitement ===
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Applique la tokenisation, la mise en minuscule, la suppression
    de la ponctuation, la suppression des stop words et la lemmatisation.
    """
    # Tokenisation et minuscules
    tokens = word_tokenize(text.lower())

    # Suppression ponctuation et mots non alphabétiques + stop words
    filtered_tokens = [
        lemmatizer.lemmatize(w) for w in tokens
        if w.isalpha() and w not in stop_words # Garde seulement les mots alphabétiques non-stop words
    ]

    # Rejoint les tokens en une chaîne de caractères
    return ' '.join(filtered_tokens)

# Exemple d'utilisation
sample_text = "This is an example showing Information Retrieval with lemmatization and stop words removal."
preprocessed_sample = preprocess_text(sample_text)
print(f"Original: {sample_text}")
print(f"Preprocessed: {preprocessed_sample}")

import nltk
nltk.download('punkt_tab')

# === Cellule 0.3: Définir les chemins ===
# !!! ADAPTEZ CE CHEMIN VERS VOTRE DOSSIER SUR GOOGLE DRIVE !!!
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Projet_RI"

# Vérification que le chemin existe
if not os.path.exists(DRIVE_PROJECT_PATH):
    raise FileNotFoundError(f"Le chemin spécifié n'existe pas : {DRIVE_PROJECT_PATH}. Vérifiez le chemin dans la Cellule 0.1 et 0.3.")

# Corrected the path for AP_TAR_PATH by removing the extra DRIVE_PROJECT_PATH
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, "AP.tar") # Assumant que c'est un .tar.gz, sinon ajustez
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "topics/") #Corrected the path
QRELS_DIR = os.path.join

# === Cellule 0.3: Définir les chemins ===
# !!! ADAPTEZ CE CHEMIN VERS VOTRE DOSSIER SUR GOOGLE DRIVE !!!
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Projet_RI"

# Vérification que le chemin existe
if not os.path.exists(DRIVE_PROJECT_PATH):
    raise FileNotFoundError(f"Le chemin spécifié n'existe pas : {DRIVE_PROJECT_PATH}. Vérifiez le chemin dans la Cellule 0.1 et 0.3.")

# Corrected the path for AP_TAR_PATH by removing the extra DRIVE_PROJECT_PATH
AP_TAR_PATH = os.path.join(DRIVE_PROJECT_PATH, "AP.tar") # Assumant que c'est un .tar.gz, sinon ajustez
TOPICS_DIR = os.path.join(DRIVE_PROJECT_PATH, "topics/") #Corrected the path
QRELS_DIR = os.path.join(DRIVE_PROJECT_PATH, "ql/") #Corrected the path


# Chemins pour les sorties (index, résultats, etc.) dans l'environnement Colab
OUTPUT_DIR = os.path.join(DRIVE_PROJECT_PATH, "output/") #Corrected the path
INDEX_DIR_BASELINE = os.path.join(OUTPUT_DIR, "baseline") #Corrected the path
INDEX_DIR_PREPROC = os.path.join(OUTPUT_DIR, "pre") #Corrected the path
CORPUS_DIR = os.path.join(OUTPUT_DIR, "Corpus") # Pour les documents extraits/formatés
RUN_DIR = os.path.join(OUTPUT_DIR, "runs") # Pour les fichiers de résultats TREC
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval") # Pour les fichiers d'évaluation

# Créer les répertoires de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_BASELINE, exist_ok=True)
os.makedirs(INDEX_DIR_PREPROC, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"Chemin du projet Drive: {DRIVE_PROJECT_PATH}")
print(f"Répertoire de sortie Colab: {OUTPUT_DIR}")

with open(JSONL_OUTPUT_PATH, 'w') as outfile, tarfile.open(AP_TAR_PATH, "r:") as tar:  # Changed mode to "r"
