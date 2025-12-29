from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import requests
import os
import gdown

# --- 1. SETUP THE FLASK APP ---
app = Flask(__name__)
CORS(app)

# --- 2. HELPER FUNCTION (Corrected to match training) ---
def aa_composition(seq: str):
    """
    Return amino acid fractions + sequence length as features.
    """
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

# --- 3. LOAD MODEL AND DATASETS ---
# --- 3. LOAD MODEL AND DATASETS ---
model = None
uniprot_to_seq = {}
seq_to_uniprot = {}
X_train_columns = []

try:
    MODEL_PATH = "RandomForest_model.joblib"

    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1yz1zswRwMUdGl895DLBposgWsEOI1JDE"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")

    df_A = pd.read_csv("Ecoli.csv")
    df_B = pd.read_csv("Human.csv")
    df_A.columns = df_A.columns.str.strip()
    df_B.columns = df_B.columns.str.strip()
    df_all = pd.concat([df_A, df_B], ignore_index=True)

    uniprot_to_seq = dict(zip(df_all['Entry'], df_all['Sequence']))
    seq_to_uniprot = dict(zip(df_all['Sequence'], df_all['Entry']))
    print("✅ Datasets loaded.")

    dummy_seq = "ACDEFGHIKLMNPQRSTVWY"
    X_train_columns = list(pd.DataFrame([aa_composition(dummy_seq)]).columns)
    print(f"✅ Feature order locked ({len(X_train_columns)} features).")

except Exception as e:
    print(f"❌ Startup error: {e}")


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

# --- START: REPLICATING YOUR JUPYTER NOTEBOOK LOGIC FOR 3D STRUCTURE ---
def fetch_rcsb_structure(uniprot_id):
    """Fetches protein structure from RCSB PDB using their robust API."""
    try:
        print(f"🔍 Method 1: Searching RCSB PDB for UniProt ID: {uniprot_id}")
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
                print(f"✅ Found PDB ID in RCSB: {pdb_id}")
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                pdb_response = requests.get(pdb_url, timeout=30, headers=headers)
                if pdb_response.status_code == 200:
                    print("✅ PDB structure downloaded successfully from RCSB!")
                    return pdb_response.text
        print("INFO: No structure found in RCSB PDB for this ID.")
        return None
    except Exception as e:
        print(f"❌ Error during RCSB PDB search: {e}")
        return None

def fetch_alphafold_direct(uniprot_id):
    """Tries a direct download from AlphaFold as a fallback."""
    try:
        print(f"🔍 Method 2: Trying direct AlphaFold download for: {uniprot_id}")
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=15, headers=headers)
        if response.status_code == 200:
            print("✅ Direct AlphaFold download successful!")
            return response.text
        return None
    except Exception as e:
        print(f"❌ Direct AlphaFold download failed: {e}")
        return None

def get_protein_structure_smart(uniprot_id):
    """Smart function to fetch 3D structure. Tries RCSB first, then AlphaFold."""
    if not uniprot_id or uniprot_id == "User Sequence": return None
    
    # Method 1: Try RCSB PDB
    structure = fetch_rcsb_structure(uniprot_id)
    if structure: return structure
    
    # Method 2: Fallback to AlphaFold
    structure = fetch_alphafold_direct(uniprot_id)
    if structure: return structure
    
    print(f"❌ No 3D structure found in any database for {uniprot_id}")
    return None
# --- END: REPLICATING YOUR JUPYTER NOTEBOOK LOGIC FOR 3D STRUCTURE ---

# --- 5. PREDICTION FUNCTION ---
def make_prediction(user_input):
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
        raise ValueError("Invalid Input. Please provide a valid UniProt ID or an amino acid sequence (min 10 characters).")

    features_dict = aa_composition(sequence_to_use)
    features_df = pd.DataFrame([features_dict])[X_train_columns]

    prediction_label = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    comp_for_chart = {k.replace('frac_', ''): v * 100 for k, v in features_dict.items() if k.startswith('frac_')}

    ph_vals, ph_probs = simulate_ph_effect(sequence_to_use, probabilities[1])
    temp_vals, temp_probs = simulate_temp_effect(sequence_to_use, probabilities[1])
    
    # Call the new, robust function
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
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
    
    try:
        data = request.get_json()
        user_input = data.get('sequence', '').upper().strip()

        if not user_input:
            return jsonify({"error": "No sequence provided"}), 400

        results = make_prediction(user_input)
        return jsonify(results)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# --- 7. RUN THE APP ---
if __name__ == '__main__':
    print("Starting Flask server for prediction...")
    app.run(host="0.0.0.0" ,debug=False, port=5000)

