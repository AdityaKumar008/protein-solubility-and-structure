from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import requests

import os
import gdown

MODEL_PATH = "RandomForest_model.joblib"

# Download model from Google Drive if not exists
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1yz1zswRwMUdGl895DLBposgWsEOI1JDE"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")


# --- 1. SETUP THE FLASK APP ---
app = Flask(__name__)
CORS(app)

# --- 2. HELPER FUNCTION (Corrected to match training) ---
def aa_composition(seq: str):
    VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")
    s = seq.upper().strip()
    length = len(s) if len(s) > 0 else 1
    counts = {aa: 0 for aa in VALID_AA}
    for ch in s:
        if ch in counts:
            counts[ch] += 1
    feats = {f"frac_{aa}": counts[aa]/length for aa in VALID_AA}
    feats["seq_len"] = float(length)
    return feats

# --- GLOBAL VARS ---
model = None
uniprot_to_seq = {}
seq_to_uniprot = {}
X_train_columns = []

# --- NEW: LAZY LOAD FUNCTIONS ---
def load_model():
    global model
    if model is None:
        print("⚙️ Loading model from file...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model ready")
    return model


def load_datasets():
    global uniprot_to_seq, seq_to_uniprot, X_train_columns
    if uniprot_to_seq:
        return  # Already loaded

    print("📥 Downloading datasets from Drive...")
    HUMAN_URL = "https://drive.google.com/uc?id=1hS-f7Ti7lg0deZvd-6b3opByrkbuY4og"
    ECOLI_URL = "https://drive.google.com/uc?id=1xLTUZGSq7TIc0bPHHnBmDO-62022SdY2"

    def fetch_csv(url, filename):
        r = requests.get(url)
        open(filename, "wb").write(r.content)
        return pd.read_csv(filename)

    df_A = fetch_csv(ECOLI_URL, "Ecoli.csv")
    df_B = fetch_csv(HUMAN_URL, "Human.csv")

    df_A.columns = df_A.columns.str.strip()
    df_B.columns = df_B.columns.str.strip()
    df_all = pd.concat([df_A, df_B], ignore_index=True)

    uniprot_to_seq = dict(zip(df_all['Entry'], df_all['Sequence']))
    seq_to_uniprot = dict(zip(df_all['Sequence'], df_all['Entry']))

    dummy = "ACDEFGHIKLMNPQRSTVWY"
    X_train_columns = list(pd.DataFrame([aa_composition(dummy)]).columns)

    print("🔥 Dataset READY")


# --- 4. OTHER HELPER FUNCTIONS ---
def validate_sequence(seq: str, min_len=10):
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    if len(seq) < min_len: return False
    return all(ch in VALID_AA for ch in seq)

def simulate_ph_effect(seq, base_prob):
    s = seq.upper().strip()
    length = len(s) if len(s) > 0 else 1
    acidic = (s.count('D') + s.count('E')) / length
    basic = (s.count('K') + s.count('R') + s.count('H')) / length
    ph_values = list(range(2, 13))
    probs = [min(max(base_prob + (((acidic * (ph/14)) + (basic * ((14-ph)/14))) - 0.1), 0), 1) for ph in ph_values]
    return ph_values, probs

def simulate_temp_effect(seq, base_prob):
    s = seq.upper().strip()
    length = len(s) if len(s) > 0 else 1
    hydrophobic = sum(s.count(aa) for aa in "AILMV") / length
    temps = list(range(20, 85, 5))
    probs = [min(max(base_prob + (-hydrophobic * ((temp - 37) / 100)), 0), 1) for temp in temps]
    return temps, probs

# --- START: PROTEIN STRUCTURE FETCH ---
def fetch_rcsb_structure(uniprot_id):
    try:
        print(f"🔍 Searching RCSB PDB for {uniprot_id}")
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
            }, "return_type": "entry"
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.post(search_url, json=query, timeout=30, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get('result_set'):
                pdb_id = data['result_set'][0]['identifier']
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                pdb_response = requests.get(pdb_url, timeout=30, headers=headers)
                if pdb_response.status_code == 200:
                    return pdb_response.text
        return None
    except:
        return None

def fetch_alphafold_direct(uniprot_id):
    try:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            return response.text
        return None
    except:
        return None

def get_protein_structure_smart(uniprot_id):
    if not uniprot_id or uniprot_id == "User Sequence": return None
    return fetch_rcsb_structure(uniprot_id) or fetch_alphafold_direct(uniprot_id)

# --- 5. PREDICTION FUNCTION ---
def make_prediction(user_input):

    load_model()
    load_datasets()

    sequence_to_use = None
    uniprot_id = "User Sequence"

    if user_input in uniprot_to_seq:
        sequence_to_use = uniprot_to_seq[user_input]
        uniprot_id = user_input
    elif user_input in seq_to_uniprot:
        sequence_to_use = user_input
        uniprot_id = seq_to_uniprot[user_input]
    elif validate_sequence(user_input):
        sequence_to_use = user_input
    
    if not sequence_to_use:
        raise ValueError("Invalid Input. Enter UniProt ID or valid AA sequence.")

    features_dict = aa_composition(sequence_to_use)
    features_df = pd.DataFrame([features_dict])[X_train_columns]

    prediction_label = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    comp_for_chart = {k.replace('frac_', ''): v * 100 for k, v in features_dict.items() if k.startswith('frac_')}

    ph_vals, ph_probs = simulate_ph_effect(sequence_to_use, probabilities[1])
    temp_vals, temp_probs = simulate_temp_effect(sequence_to_use, probabilities[1])
    
    pdb_content = get_protein_structure_smart(uniprot_id)

    return {
        "uniprot_id": uniprot_id,
        "prediction": int(prediction_label),
        "probability": {"insoluble": float(probabilities[0]), "soluble": float(probabilities[1])},
        "composition": comp_for_chart,
        "ph_graph": {"ph_values": ph_vals, "solubility_values": ph_probs},
        "temp_graph": {"temp_values": temp_vals, "solubility_values": temp_probs},
        "pdb_data": pdb_content
    }

# --- 6. API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        user_input = data.get('sequence', '').upper().strip()
        if not user_input:
            return jsonify({"error": "No sequence provided"}), 400
        return jsonify(make_prediction(user_input))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 7. RUN ---
if __name__ == '__main__':
    print("Starting Flask server for prediction...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
