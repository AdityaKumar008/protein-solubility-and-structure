# Protein Solubility and Structure

## Description

This project predicts **protein solubility in an *E. coli* expression environment** using a machine learning approach based on amino acid composition.
It also demonstrates **automatic 3D protein structure retrieval and visualization** from established biological databases.

> ⚠️ This repository focuses on **demonstration, architecture, and results**, as full backend deployment is not feasible under free hosting tiers.

---

## 🎥 Demo Video

A complete working demonstration of the project (backend + frontend) is provided as a recorded video.

**Demo Video Link:**  
👉 https://drive.google.com/file/d/111LEgGQMFVOf3f2EKWsZ2ExM6jRelnkl/view?usp=drivesdk

The video shows:
- Backend Flask server running live in the terminal
- Model inference using UniProt ID / protein sequence
- Frontend interaction and result visualization
- 3D protein structure retrieval and display

---

## 🌐 Live Frontend Demo

**Frontend (static):**  
https://adityakumar008.github.io/protein-solubility-and-structure/

> Note: This frontend is deployed using GitHub Pages and works independently of the backend.

---

## ✨ Features

- Protein solubility prediction (*E. coli* context)
- Accepts **UniProt ID** or **amino acid sequence**
- Uses a **pre-trained Random Forest model**
- Amino-acid composition–based feature engineering
- Automatic **3D protein structure fetching**
- Interactive, web-based UI

---

## 🧠 About the Machine Learning Model

- The solubility prediction model is a **Random Forest classifier**
- Trained locally using curated protein datasets
- Saved as a `.joblib` file for reuse (no retraining required)

### ❗ Why the trained model is not in this repository

- The trained model file is **~500 MB**
- GitHub does not allow files larger than **100 MB**
- Free hosting platforms also impose strict storage limits

Therefore:
- The **model file is excluded from GitHub**
- The **full working inference is demonstrated via video**
- The **code remains reproducible** for local execution

---

## 🚫 Why Backend Is Not Deployed Online

The backend (Flask + ML model) is **not deployed** due to:
- Large model size (~500 MB)
- Free-tier hosting memory and storage limits
- Cold-start and timeout issues for ML inference

Instead, this project demonstrates:
- Correct backend logic
- Real-time predictions (shown in demo video)
- Complete end-to-end workflow

This approach is commonly accepted for **academic and portfolio projects**.

---

## 🧬 3D Protein Structure Source

Protein structures are **not predicted from scratch**.

They are fetched dynamically using UniProt IDs from:
- **RCSB Protein Data Bank (PDB)** – primary source
- **AlphaFold Protein Structure Database** – fallback

Visualization is handled on the frontend.

---

## 🛠 Tech Stack

- Python
- Flask
- Scikit-learn (Random Forest)
- Pandas, NumPy
- HTML, CSS, JavaScript
- py3Dmol (3D visualization)

---

## ▶️ How to Run Locally

> Backend files are excluded from GitHub due to size constraints.

1. Clone the repository
2. Add the trained model (`.joblib`) and datasets locally
3. Install required Python dependencies
4. Run the Flask backend (`app.py`)
5. Open the frontend in a browser
6. Enter a **UniProt ID** or **protein sequence**

---

## 📁 Project Structure (Conceptual)

- `app.py` – Flask backend
- `Final_Code.py` – ML training & analysis
- `*.csv` – Training datasets
- `*.joblib` – Trained ML model (local only)
- Frontend files – UI and visualization
- Demo video – Full working proof

---

## 👤 Author

**Aditya Kumar**  
LinkedIn: https://www.linkedin.com/in/aditya-kumar-b7920b328  
GitHub: https://github.com/AdityaKumar008
