# Complete code with everything functional , showing 3D Structure also
# Gemini




# ===============================
# Cell 1: Imports
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

import joblib


import requests
import nglview as nv
from io import StringIO
import py3Dmol
from IPython.display import display , HTML


# ===============================
# Cell 2: Load datasets
# ===============================
df_train = pd.read_csv("ecoli_train.csv")
df_test  = pd.read_csv("ecoli_test.csv")

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

print("Columns:", df_train.columns.tolist())
display(df_train.head())


# ===============================
# Cell 3: Sanity check
# ===============================
print("Missing values in train:\n", df_train.isna().sum())
print("Missing values in test:\n", df_test.isna().sum())

print("\nLabel distribution in train:\n", df_train['label'].value_counts())
print("\nLabel distribution in test:\n", df_test['label'].value_counts())


# ===============================
# Cell 4: Feature extraction (AA composition)
# ===============================
VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

def aa_composition(seq: str):
    """Return amino acid fractions + sequence length as features"""
    if not isinstance(seq, str):
        seq = "" if pd.isna(seq) else str(seq)
    s = seq.upper().strip()
    length = len(s) if len(s) > 0 else 1
    counts = {aa: 0 for aa in VALID_AA}
    for ch in s:
        if ch in counts:
            counts[ch] += 1
    feats = {f"frac_{aa}": counts[aa]/length for aa in VALID_AA}
    feats["seq_len"] = length
    return feats

def featurize_dataframe(df, seq_col="seq"):
    features = df[seq_col].apply(aa_composition).apply(pd.Series)
    return features


# ===============================
# Cell 5: Build feature tables
# ===============================
X_train = featurize_dataframe(df_train, seq_col="seq")
y_train = df_train["label"].astype(int)

X_test = featurize_dataframe(df_test, seq_col="seq")
y_test = df_test["label"].astype(int)

print("Train features shape:", X_train.shape)
print("Test features shape:", X_test.shape)


# ===============================
# Cell 6: Train models (LogReg, RF, SVM)
# ===============================
results = {}

# Logistic Regression
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)
proba_lr = logreg.predict_proba(X_test)[:, 1]
results["LogReg"] = {
    "accuracy": accuracy_score(y_test, pred_lr),
    "f1": precision_recall_fscore_support(y_test, pred_lr, average="binary")[2],
    "roc_auc": roc_auc_score(y_test, proba_lr)
}

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]
results["RandomForest"] = {
    "accuracy": accuracy_score(y_test, pred_rf),
    "f1": precision_recall_fscore_support(y_test, pred_rf, average="binary")[2],
    "roc_auc": roc_auc_score(y_test, proba_rf)
}

# SVM (RBF kernel)
svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
])
svm_rbf.fit(X_train, y_train)
pred_svm = svm_rbf.predict(X_test)
proba_svm = svm_rbf.predict_proba(X_test)[:, 1]
results["SVM_RBF"] = {
    "accuracy": accuracy_score(y_test, pred_svm),
    "f1": precision_recall_fscore_support(y_test, pred_svm, average="binary")[2],
    "roc_auc": roc_auc_score(y_test, proba_svm)
}

results


# ===============================
# Cell 7: Evaluation — Confusion matrix & report
# ===============================
best_name = max(results, key=lambda k: results[k]["f1"])
print(f"Best model: {best_name}")
print("Metrics:", results[best_name])

best_pred = {"LogReg": pred_lr, "RandomForest": pred_rf, "SVM_RBF": pred_svm}[best_name]
print("\nClassification Report:\n")
print(classification_report(y_test, best_pred, digits=4))

cm = confusion_matrix(y_test, best_pred)
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1]); plt.yticks([0,1])
for (i,j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center", color="red")
plt.show()


# ===============================
# Cell 8: Save best model
# ===============================
best_model = {"LogReg": logreg, "RandomForest": rf, "SVM_RBF": svm_rbf}[best_name]
joblib.dump(best_model, f"{best_name}_model.joblib")
print("Model saved as", f"{best_name}_model.joblib")


# ===============================
# Cell 9: Predict on new sequence + 3D structure (RandomForest) using datasets
# ===============================

# ===============================
# Final Merged Script
# Protein Solubility Predictor with Advanced 3D Structure Visualization
# ===============================

# -------------------------------
# 1. Install and Import Packages
# -------------------------------
# !pip install joblib pandas matplotlib requests py3Dmol -q



print("✅ Packages and libraries loaded successfully!")

# -------------------------------
# 2. Load Model and Datasets
# -------------------------------
# Load the pre-trained RandomForest model
try:
    best_model = joblib.load("RandomForest_model.joblib")
    print("✅ Model 'RandomForest_model.joblib' loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'RandomForest_model.joblib' not found. Make sure the model file is in the correct directory.")
    # Exit or handle the error appropriately
    exit()


# Load datasets from CSV files
try:
    df_A = pd.read_csv("Ecoli.csv")
    df_B = pd.read_csv("Human.csv")

    # Strip any spaces from column names
    df_A.columns = df_A.columns.str.strip()
    df_B.columns = df_B.columns.str.strip()

    # Combine datasets for a comprehensive mapping
    df_all = pd.concat([df_A, df_B], ignore_index=True)

    # Create mappings for quick lookups
    uniprot_to_seq = dict(zip(df_all['Entry'], df_all['Sequence']))
    seq_to_uniprot = dict(zip(df_all['Sequence'], df_all['Entry']))
    print("✅ Datasets 'Ecoli.csv' and 'Human.csv' loaded and processed.")

except FileNotFoundError as e:
    print(f"❌ Error: Dataset file not found -> {e}. Please ensure the CSV files are present.")
    exit()


# ----------------------------------------------------
# 3. ADVANCED 3D STRUCTURE FETCHING (from Script 2)
# ----------------------------------------------------
def fetch_rcsb_structure_fixed(uniprot_id):
    """Fetches protein structure from RCSB PDB using their robust API."""
    try:
        print(f"🔍 Searching RCSB PDB for UniProt ID: {uniprot_id}")
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": uniprot_id
                }
            },
            "return_type": "entry"
        }
        response = requests.post(search_url, json=query, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if 'result_set' in data and data['result_set']:
                pdb_id = data['result_set'][0]['identifier']
                print(f"✅ Found PDB ID in RCSB: {pdb_id}")
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                print(f"📥 Downloading structure from: {pdb_url}")
                pdb_response = requests.get(pdb_url, timeout=30)
                if pdb_response.status_code == 200:
                    print("✅ PDB structure downloaded successfully from RCSB!")
                    return pdb_response.text, f"RCSB PDB ({pdb_id})"
        print("INFO: No structure found in RCSB PDB for this ID.")
        return None, None
    except Exception as e:
        print(f"❌ Error during RCSB PDB search: {e}")
        return None, None

def fetch_alphafold_direct(uniprot_id):
    """Tries a direct download from AlphaFold as a fallback."""
    try:
        print(f"🔍 Trying direct AlphaFold download for: {uniprot_id}")
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            print("✅ Direct AlphaFold download successful!")
            return response.text, "AlphaFold DB"
        return None, None
    except Exception as e:
        print(f"❌ Direct AlphaFold download failed: {e}")
        return None, None

def get_protein_structure_smart(uniprot_id):
    """Smart function to fetch 3D structure. Tries RCSB first, then AlphaFold."""
    print(f"\n🎯 Attempting to fetch 3D structure for: {uniprot_id}")
    
    # Method 1: Try the primary, more reliable source (RCSB PDB)
    structure, source = fetch_rcsb_structure_fixed(uniprot_id)
    if structure:
        return structure, source
    
    # Method 2: Fallback to direct AlphaFold download
    structure, source = fetch_alphafold_direct(uniprot_id)
    if structure:
        return structure, source
    
    print(f"❌ No 3D structure found in any database for {uniprot_id}")
    return None, None

def display_3d_structure_clean(pdb_content, width=800, height=500):
    """Displays the 3D structure using py3Dmol with clean styling."""
    try:
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(pdb_content, 'pdb')
        viewer.setStyle({'cartoon': {'color': 'spectrum'}})
        viewer.setStyle({'hetflag': True}, {'stick': {'colorscheme': 'greenCarbon'}})
        viewer.zoomTo()
        print("✅ 3D viewer created successfully!")
        return viewer
    except Exception as e:
        print(f"❌ An error occurred during 3D display: {e}")
        return None

# ----------------------------------------------------
# 4. PREDICTION & ANALYSIS FUNCTIONS (from Script 1)
# ----------------------------------------------------
def aa_composition(seq: str):
    """Calculates the amino acid composition and length of a sequence."""
    VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")
    s = seq.upper().strip()
    length = len(s)
    counts = {aa: 0 for aa in VALID_AA}
    for ch in s:
        if ch in counts:
            counts[ch] += 1
    feats = {f"frac_{aa}": counts[aa] / length for aa in VALID_AA}
    feats["seq_len"] = length
    return feats

def validate_sequence(seq: str, min_len=10):
    """Checks if a sequence is a valid amino acid sequence."""
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    seq = seq.upper().strip()
    if len(seq) < min_len:
        return False
    return all(ch in VALID_AA for ch in seq)

def predict_sequence(seq: str, model):
    """Predicts solubility for a given sequence using the loaded model."""
    x = pd.DataFrame([aa_composition(seq)])
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else None
    return pred, proba

def simulate_ph_effect(seq, base_prob):
    """Simulates how pH might affect solubility based on charged residues."""
    comp = aa_composition(seq)
    acidic = comp["frac_D"] + comp["frac_E"]
    basic = comp["frac_K"] + comp["frac_R"] + comp["frac_H"]
    ph_values = range(2, 13)
    probs = []
    for ph in ph_values:
        adj = (acidic * (ph / 14)) + (basic * ((14 - ph) / 14))
        adjusted_prob = min(max(base_prob + (adj - 0.1), 0), 1)
        probs.append(adjusted_prob)
    return list(ph_values), probs

def simulate_temp_effect(seq, base_prob):
    """Simulates how temperature might affect solubility based on hydrophobicity."""
    comp = aa_composition(seq)
    hydrophobic = sum(comp[f"frac_{aa}"] for aa in "AILMV")
    temps = range(20, 85, 5)
    probs = []
    for temp in temps:
        temp_effect = -hydrophobic * ((temp - 37) / 100)
        adjusted_prob = min(max(base_prob + temp_effect, 0), 1)
        probs.append(adjusted_prob)
    return list(temps), probs

# ===============================
# 5. MAIN WORKFLOW
# ===============================
print("\n" + "="*60)
print("🧬 PROTEIN SOLUBILITY PREDICTOR + 3D STRUCTURE VISUALIZER 🧬")
print("="*60)
user_input = input("Enter a UniProt ID or an amino acid sequence: ").strip().upper()

sequence_to_use = None
uniprot_id_to_show = None

# Case 1: Input is a known UniProt ID from our dataset
if user_input in uniprot_to_seq:
    sequence_to_use = uniprot_to_seq[user_input]
    uniprot_id_to_show = user_input
    print(f"✅ Found UniProt ID '{user_input}' in the dataset.")

# Case 2: Input is a known sequence from our dataset
elif user_input in seq_to_uniprot:
    sequence_to_use = user_input
    uniprot_id_to_show = seq_to_uniprot[user_input]
    print(f"✅ Found sequence in the dataset. Corresponding UniProt ID is '{uniprot_id_to_show}'.")

# Case 3: Input is a new, valid amino acid sequence
elif validate_sequence(user_input):
    sequence_to_use = user_input
    print("✅ Valid new amino acid sequence provided.")

# Case 4: Invalid input
else:
    print("\n❌ Invalid Input! Please enter a valid UniProt ID (e.g., P00533) or an amino acid sequence (at least 10 residues, using standard one-letter codes).")

# --- If we have a valid sequence, proceed with analysis ---
if sequence_to_use:
    pred, proba = predict_sequence(sequence_to_use, best_model)
    
    print("\n" + "-"*20 + " PREDICTION RESULTS " + "-"*20)
    print(f"Prediction: {'SOLUBLE (1)' if pred == 1 else 'INSOLUBLE (0)'}")
    if proba is not None:
        print(f"Probability of being Soluble: {proba[0]:.2%}")
    print("-" * 58)

    if proba is not None:
        # --- Generate and show plots ---
        print("\n📊 Generating analysis plots...")

        # Pie Chart for Solubility Probability
        plt.figure(figsize=(5, 5))
        plt.pie([proba[0], 1 - proba[0]], labels=["Soluble", "Insoluble"],
                autopct="%1.1f%%", colors=["#66bb6a", "#ef5350"],
                shadow=True, explode=(0.05, 0.05), startangle=140)
        plt.title("Predicted Solubility Probability")
        plt.show()
        
        # Bar Chart for Amino Acid Composition
        composition = aa_composition(sequence_to_use)
        aa_order = list("ACDEFGHIKLMNPQRSTVWY")
        aa_frac = [composition[f"frac_{aa}"] for aa in aa_order]
        plt.figure(figsize=(12, 5))
        plt.bar(aa_order, aa_frac, color='skyblue')
        plt.xlabel("Amino Acid")
        plt.ylabel("Fraction in Sequence")
        plt.title("Amino Acid Composition of Input Sequence")
        plt.show()
        
        # Line Plot for Simulated pH Effect
        ph_vals, ph_probs = simulate_ph_effect(sequence_to_use, proba[0])
        plt.figure(figsize=(7, 5))
        plt.plot(ph_vals, ph_probs, marker="o", color="purple")
        plt.xlabel("pH")
        plt.ylabel("Simulated Solubility Probability")
        plt.title("Simulated Effect of pH on Solubility")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()
        
        # Line Plot for Simulated Temperature Effect
        temp_vals, temp_probs = simulate_temp_effect(sequence_to_use, proba[0])
        plt.figure(figsize=(7, 5))
        plt.plot(temp_vals, temp_probs, marker="o", color="red")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Simulated Solubility Probability")
        plt.title("Simulated Effect of Temperature on Solubility")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()
    
    # --- Fetch and display 3D structure if a UniProt ID is available ---
    if uniprot_id_to_show:
        structure_content, source = get_protein_structure_smart(uniprot_id_to_show)
        if structure_content and source:
            print(f"\n✨ Displaying 3D structure for {uniprot_id_to_show} from {source}")
            print("💡 Controls: Rotate=Click+drag • Zoom=Scroll • Pan=Right-click+drag")
            viewer = display_3d_structure_clean(structure_content)
            if viewer:
                display(viewer)
        else:
            print(f"\nINFO: Could not retrieve a 3D structure for UniProt ID '{uniprot_id_to_show}'.")
    else:
        print("\nINFO: 3D structure display is only available for proteins with a known UniProt ID from the dataset.")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE")
print("="*60)