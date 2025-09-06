# 📡 AT&T — SMS Spam Detector

Détection automatique des SMS **spam** vs **ham** pour protéger les clients d’AT&T.  
Ce projet s’appuie sur un **fine-tuning de BERT/DistilBERT** (HuggingFace Transformers) et met en avant les **bonnes métriques métier** (Recall spam, F1 spam, Precision ham). Le livrable principal est un **notebook exécutable et prêt à présenter** : `Projet.ipynb`.

---

## 📖 Contexte

AT&T est un géant des télécommunications ; ses utilisateurs sont exposés à des **SMS indésirables (spams)** qui nuisent à la satisfaction et à la sécurité (phishing, liens frauduleux).  
Historiquement, le marquage des spams a été **manuel** — ce n’est plus scalable.

**Objectif :** construire un **classifieur automatique** qui identifie un SMS comme **ham (légitime)** ou **spam**, **uniquement à partir du texte**.

## 🎯 Objectifs & critères de succès

- Construire un pipeline **reproductible** : préprocessing → entraînement → évaluation.
- Utiliser le **transfert learning** (BERT/DistilBERT) pour compenser la faible quantité de données.
- Rapporter des **métriques adaptées au risque métier** :
  - **Recall (spam)** : ne pas laisser passer de spams (priorité).
  - **F1 (spam)** : équilibre précision/rappel côté spam.
  - **Precision (ham)** : limiter les faux positifs (ne pas bloquer des messages légitimes).

## 🗂️ Données (schéma & conseils)

- Fichier CSV avec 2 colonnes :
  - `label` : valeur textuelle **"ham"** ou **"spam"**
  - `message` : contenu textuel du SMS  
- Si vous partez d’un CSV type “SMS Spam Collection” (souvent `v1` / `v2`) :
  - renommez **`v1 → label`** et **`v2 → message`** avant l’encodage.
- **Encodage des labels** : `ham → 0`, `spam → 1`.

💡 **Encodage & lecture robuste** :
- Le notebook utilise `chardet` pour détecter l’encodage si nécessaire.
- Supprimez les doublons et lignes vides pour éviter les biais ou crashs.


## 🧭 Méthodologie (ce que fait le notebook)

1. **Chargement & préparation**
   - Lecture du CSV (`pandas`), renommage (`label`, `message`), nettoyage de base.
   - Encodage binaire des labels (`ham`=0, `spam`=1).
2. **Split train/test stratifié**
   - `train_test_split(..., stratify=labels, test_size=0.2, random_state=42)`.
   - **Pas de fuite** de données ; reproductibilité assurée.
3. **Tokenization (HuggingFace)**
   - `BertTokenizerFast` / `AutoTokenizer`, `max_length=128`, `truncation`, `padding`.
4. **Modélisation**
   - `BertForSequenceClassification` / `AutoModelForSequenceClassification` (2 classes).
   - Tête de classification binaire.
5. **Entraînement (Trainer)**
   - `TrainingArguments`: `epochs=3`, `batch_size=8-16`, `lr≈2e-5`, `evaluation_strategy="epoch"`, `load_best_model_at_end=True`.
6. **Évaluation**
   - `classification_report` (precision/recall/F1 par classe).
   - `confusion_matrix` (lecture FN/FP).
   - Visualisations (seaborn/matplotlib).
7. **(Optionnel) Sauvegarde**
   - `save_pretrained` pour le modèle et le tokenizer (déploiement possible).


## 📊 Interprétation des métriques

- **Recall (spam)** ↑ : **critique** pour ne pas laisser passer de spams.
- **Precision (ham)** : éviter de bloquer des messages légitimes.
- **F1 (spam)** : bonne synthèse de performance sur la classe à risque.

> Rappel lecture matrice de confusion :
> - **FN (spam→ham)** : *spams manqués* (⚠️ risque).
> - **FP (ham→spam)** : *hams bloqués* (gêne utilisateur).


## 🔁 Reproductibilité

- **Seeds** fixés (`random_state=42`, seeds torch si GPU dispo).
- **Split stratifié** conservant la proportion spam/ham.
- **Évaluation sur jeu de test** strictement tenu à l’écart pendant l’entraînement.


## ⚙️ Installation

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt


