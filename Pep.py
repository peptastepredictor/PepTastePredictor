
import os
import io
import textwrap
from typing import Tuple, Dict, List, Union

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# BioPython helpers
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.SeqUtils import seq3
    HAS_BIOPY = True
except Exception:
    HAS_BIOPY = False

# Optional: PeptideBuilder for nicer PDB building
try:
    import PeptideBuilder
    from PeptideBuilder import Geometry
    from Bio.PDB import PDBIO
    HAS_PEPTIDEBUILDER = True
except Exception:
    HAS_PEPTIDEBUILDER = False

# Optional: 3D viewer
try:
    import py3Dmol
    HAS_PY3DMOL = True
except Exception:
    HAS_PY3DMOL = False

# -----------------------------
# Constants / Config
# -----------------------------
APP_TITLE = "PepTastePredictor"
MODEL_PATH = "taste_model.pkl"
DATA_PATH = "AIML.xlsx"
AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWY")
THREE_LETTER = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
    "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
    "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"
}

# -----------------------------
# Utility functions (feature computation, validation, docking)
# -----------------------------
def clean_seq(seq: str) -> str:
    return "".join(str(seq).upper().split())

def validate_sequence(seq: str) -> Tuple[bool, str]:
    """
    Validate sequence. Returns (True, cleaned_seq) if valid, otherwise (False, message).
    """
    if seq is None:
        return False, "No sequence provided."
    s = clean_seq(seq)
    if s == "":
        return False, "Sequence is empty after cleaning."
    bad = set(s) - AA_ALLOWED
    if bad:
        return False, f"Invalid characters in sequence: {''.join(sorted(bad))}"
    return True, s

def compute_features_single(seq: str) -> Dict[str, float]:
    """
    Compute numeric features for a single sequence using Bio.SeqUtils.ProtParam.
    Uses deterministic ordering (dict insertion order).
    """
    if not HAS_BIOPY:
        raise RuntimeError("Biopython is required for feature computation (install with `pip install biopython`).")
    analysed = ProteinAnalysis(seq)
    aa_composition = analysed.amino_acids_percent
    ai = (
        aa_composition.get("A", 0) * 100
        + aa_composition.get("V", 0) * 100 * 2.9
        + (aa_composition.get("I", 0) + aa_composition.get("L", 0)) * 100 * 3.9
    )
    ss = analysed.secondary_structure_fraction()
    feats = {
        "Molecular weight(g/mol)": float(f"{analysed.molecular_weight():.5f}"),
        "Isoelectric Point": float(f"{analysed.isoelectric_point():.4f}"),
        "Aromaticity": float(f"{analysed.aromaticity():.5f}"),
        "Instability Index": float(f"{analysed.instability_index():.4f}"),
        "Net charge -pH:7": float(f"{analysed.charge_at_pH(7.0):.4f}"),
        "GRAVY (Hydropathy)": float(f"{analysed.gravy():.5f}"),
        "Aliphatic Index": float(f"{ai:.4f}"),
        "Flexibility (avg)": float(f"{np.mean(analysed.flexibility()):.5f}"),
        "Helix Fraction": float(f"{ss[0]:.5f}"),
        "Turn Fraction": float(f"{ss[1]:.5f}"),
        "Sheet Fraction": float(f"{ss[2]:.5f}"),
    }
    return feats

def compute_features(sequences: Union[str, List[str], pd.Series]) -> Union[Dict[str, float], pd.DataFrame]:
    """
    If input is a single string, returns dict.
    If list-like, returns DataFrame with stable column order.
    """
    if isinstance(sequences, str):
        return compute_features_single(sequences)
    seqs = list(sequences)
    rows = []
    for s in seqs:
        ok, cleaned_or_msg = validate_sequence(s)
        if not ok:
            # produce row of NaNs (caller should filter invalid sequences)
            rows.append({k: np.nan for k in compute_features_single("ACD").keys()})
        else:
            rows.append(compute_features_single(cleaned_or_msg))
    df = pd.DataFrame(rows)
    return df

def compute_peptide_properties(seq: str) -> Dict[str, Union[str, float]]:
    """
    Friendly properties mapping for UI display.
    """
    feats = compute_features_single(seq)
    props = {
        "Sequence": seq,
        "Length": len(seq),
        "Molecular weight (g/mol)": feats["Molecular weight(g/mol)"],
        "Isoelectric point": feats["Isoelectric Point"],
        "GRAVY (Hydropathy)": feats["GRAVY (Hydropathy)"],
        "Aliphatic Index": feats["Aliphatic Index"],
        "Instability Index": feats["Instability Index"],
        "Net charge at pH 7": feats["Net charge -pH:7"],
        "Aromaticity": feats["Aromaticity"],
        "Helix fraction": feats["Helix Fraction"],
        "Turn fraction": feats["Turn Fraction"],
        "Sheet fraction": feats["Sheet Fraction"],
    }
    return props

def compute_docking_score(gravy: float, confidence: float) -> int:
    """
    Heuristic 0..100 docking "score" combining confidence and hydrophobicity.
    """
    g_norm = (gravy + 2.0) / 4.0  # assume typical GRAVY in [-2,2]
    g_norm = float(np.clip(g_norm, 0.0, 1.0))
    c_norm = float(np.clip(confidence, 0.0, 1.0))
    score = int(round((c_norm * 0.7 + g_norm * 0.3) * 100))
    return int(np.clip(score, 0, 100))

# -----------------------------
# PDB builders (PeptideBuilder fallback)
# -----------------------------
def build_peptide_pdb_peptidebuilder(seq: str) -> str:
    if not HAS_PEPTIDEBUILDER:
        raise RuntimeError("PeptideBuilder not available")
    if len(seq) == 0:
        raise ValueError("Empty sequence")
    structure = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        geom = Geometry.geometry(aa)
        PeptideBuilder.add_residue(structure, geom)
    io = PDBIO()
    tmp = "temp_pepbuild.pdb"
    io.set_structure(structure)
    io.save(tmp)
    with open(tmp, "r") as f:
        pdb_text = f.read()
    try:
        os.remove(tmp)
    except Exception:
        pass
    return pdb_text

def build_linear_ca_trace_pdb(seq: str, step: float = 3.8) -> str:
    lines = []
    serial = 1
    x = y = z = 0.0
    for i, aa in enumerate(seq, start=1):
        res3 = THREE_LETTER.get(aa, "GLY")
        line = (
            f"ATOM  {serial:5d}  CA  {res3:>3s} A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        lines.append(line)
        serial += 1
        x += step
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)

def save_pdb_bytes(seq: str) -> Tuple[str, bytes]:
    """
    Try to build PDB with PeptideBuilder; fallback to CA-trace.
    Returns (filename, bytes_content).
    """
    seq_clean = clean_seq(seq)
    if not seq_clean:
        raise ValueError("Empty sequence")
    try:
        pdb_str = build_peptide_pdb_peptidebuilder(seq_clean)
    except Exception:
        pdb_str = build_linear_ca_trace_pdb(seq_clean)
    safe_seq = "".join([c for c in seq_clean if c.isalnum()])
    if not safe_seq:
        safe_seq = "peptide"
    filename = f"{safe_seq}.pdb"
    return filename, pdb_str.encode("utf-8")

# -----------------------------
# Streamlit app UI + model training / loading
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧬")

# Minimal CSS for nicer look
st.markdown(
    """
    <style>
      .hero { background: linear-gradient(90deg,#071422,#0b1724); padding:18px; border-radius:10px; color: #e6eef8; }
      .card { padding:10px; border-radius:8px; background:#0f1724; color:#e6eef8; }
      .small { font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.title(APP_TITLE)
    st.write("Predict peptide taste classes, compute properties, visualize structures.")
    st.divider()
    st.markdown("**Try sample sequences**")
    sample_sequences = ["EDEGEQPRPF", "GGGSSH", "ACDEFGHIK", "FLGFR"]
    sample = st.selectbox("Sample", sample_sequences)
    if st.button("Use sample"):
        st.session_state["user_seq"] = sample
    st.divider()
    st.markdown("**About This Tool**")
    st.markdown(
        "PepTastePredictor is a **machine learning-based predictive model** trained on peptide "
        "taste classification data. It uses a Random Forest classifier to predict peptide taste classes "
        "based on computed biochemical properties."
    )
    st.markdown("**Key Features:**")
    st.markdown(
        "- **Machine Learning Model**: Trained on labeled peptide sequences using scikit-learn\n"
        "- **Property Computation**: Uses Biopython (Bio.SeqUtils.ProtParam) to compute 11 key biochemical features\n"
        "- **Structure Generation**: Generates 3D PDB files using PeptideBuilder for visualization\n"
        "- **Confidence Scores**: Provides probability distributions for predicted taste classes"
    )

# Load dataset (cached)
@st.cache_data
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    # normalize peptide column if present
    if "peptide" in df.columns:
        df["peptide"] = df["peptide"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    return df

data = load_dataset()

# Train or load model (cached resource)
@st.cache_resource
def train_or_load_model():
    """
    Train or load RandomForest classifier.
    Returns (model, X_df, y_series, acc, report)
    """
    if data.empty:
        return None, None, None, None, None

    # Try loading saved model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            # compute features for the dataset for quick metrics
            if "peptide" in data.columns and "Taste" in data.columns:
                X_full = compute_features(data["peptide"].tolist())
                y_full = data["Taste"].astype(str).str.strip()
                # drop invalid rows
                mask_valid = ~X_full.isna().any(axis=1)
                if mask_valid.any():
                    try:
                        y_pred_full = model.predict(X_full.loc[mask_valid])
                        acc_full = accuracy_score(y_full.loc[mask_valid], y_pred_full)
                        report_full = classification_report(y_full.loc[mask_valid], y_pred_full, output_dict=True)
                    except Exception:
                        acc_full = None
                        report_full = None
                else:
                    acc_full = None
                    report_full = None
                return model, X_full, y_full, acc_full, report_full
            else:
                return model, None, None, None, None
        except Exception:
            st.warning("Found model file but failed to load. Retraining...")

    # Train fresh model
    if "peptide" not in data.columns or "Taste" not in data.columns:
        return None, None, None, None, None

    X_all = compute_features(data["peptide"].tolist())
    y_all = data["Taste"].astype(str).str.strip()

    # drop rows with NaN features (invalid sequences)
    mask_valid = ~X_all.isna().any(axis=1)
    X_all = X_all.loc[mask_valid].reset_index(drop=True)
    y_all = y_all.loc[mask_valid].reset_index(drop=True)

    if X_all.empty:
        return None, None, None, None, None

    # remove classes with fewer than 2 samples (can't stratify)
    counts = y_all.value_counts()
    valid_classes = counts[counts >= 2].index
    mask_class = y_all.isin(valid_classes)
    X = X_all.loc[mask_class].reset_index(drop=True)
    y = y_all.loc[mask_class].reset_index(drop=True)

    if len(y.unique()) < 2:
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        st.warning("Could not persist trained model to disk.")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, X, y, acc, report

model, X_df, y_series, acc, report = train_or_load_model()

# Layout: tabs
tabs = st.tabs(["Predict ✨", "Batch 📤", "Model 🔍", "Dataset 📂", "Structure 🧬"])

# Dataset tab
with tabs[3]:
    st.subheader("📂 Dataset Preview")
    if data.empty:
        st.info(f"No dataset found at {DATA_PATH}. Place AIML.xlsx in the app root to enable training.")
    else:
        st.dataframe(data.head(50))

# Model metrics tab
with tabs[2]:
    st.subheader("🔎 Model Metrics")
    if model is None:
        st.info("Model not available. Ensure dataset is present and has enough labeled examples.")
    else:
        with st.expander("Classification report"):
            if acc is not None:
                st.success(f"Model hold-out accuracy: {acc:.3f}")
            if report is not None:
                st.json(report)
        with st.expander("Top feature importances"):
            try:
                feat_imp = pd.Series(model.feature_importances_, index=X_df.columns).sort_values(ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax)
                ax.set_title("Top 20 Feature Importances")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not compute feature importances: {e}")
        with st.expander("Confusion matrix"):
            try:
                y_pred_all = model.predict(X_df)
                cm = confusion_matrix(y_series, y_pred_all, labels=model.classes_)
                cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax2)
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Actual")
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Could not compute confusion matrix: {e}")

# Predict tab
with tabs[0]:
    st.subheader("🔬 Predict Taste and Properties of a Peptide")
    st.caption("Enter a peptide sequence (one-letter amino acid codes).")

    col_main, col_side = st.columns([2, 1])
    with col_main:
        user_seq = st.text_input("Peptide sequence:", key="user_seq", value=st.session_state.get("user_seq", ""))
        predict_btn = st.button("Predict")

        if predict_btn:
            if not user_seq:
                st.warning("Please enter a peptide sequence.")
            else:
                ok, msg = validate_sequence(user_seq)
                if not ok:
                    st.error(msg)
                else:
                    # compute single features
                    try:
                        feats_df = compute_features([msg])
                    except Exception as e:
                        st.error(f"Feature computation failed: {e}")
                        feats_df = pd.DataFrame()

                    if feats_df.empty or feats_df.isna().all(axis=None):
                        st.warning("Feature computation failed or returned invalid values.")
                    else:
                        if model is None:
                            st.warning("Model not available to generate prediction.")
                            predicted = "Unknown"
                            probs = None
                        else:
                            predicted = model.predict(feats_df)[0]
                            try:
                                probs = model.predict_proba(feats_df)[0]
                            except Exception:
                                probs = None

                        st.markdown(f"<div class='card'><h3 style='margin:0'>Predicted Taste: <span style='color:#66b8ff'>{predicted}</span></h3></div>", unsafe_allow_html=True)

                        if probs is not None:
                            st.markdown("**Class probabilities**")
                            top_idx = np.argsort(probs)[::-1][:6]
                            for i in top_idx:
                                cls = model.classes_[i]
                                p = probs[i]
                                pct = int(round(p * 100))
                                color = "#198754" if pct >= 60 else ("#0d6efd" if pct >= 30 else "#fd7e14")
                                st.markdown(
                                    f"<div class='card small' style='margin-bottom:6px'>"
                                    f"<strong>{cls}</strong> <span style='float:right'>{pct}%</span>"
                                    f"<div style='background:#e9ecef; border-radius:6px; height:10px; margin-top:6px;'>"
                                    f"<div style='width:{pct}%; background:{color}; height:100%; border-radius:6px;'></div>"
                                    "</div></div>",
                                    unsafe_allow_html=True
                                )

                        # Side column: properties, docking score, 3-letter seq, PDB
                        with col_side:
                            st.subheader("🧪 Properties")
                            try:
                                props = compute_peptide_properties(msg)
                                # docking heuristic
                                confidence = float(np.max(probs)) if probs is not None else 0.0
                                gravy = props.get("GRAVY (Hydropathy)", 0.0)
                                docking_score = compute_docking_score(gravy, confidence)
                                props["Docking score (pred)"] = f"{docking_score} / 100"
                                # Convert all values to strings to avoid Arrow serialization issues
                                props_str = {k: str(v) for k, v in props.items()}
                                st.table(pd.DataFrame.from_dict(props_str, orient="index", columns=["Value"]))
                                # visual progress
                                try:
                                    st.progress(int(docking_score))
                                except Exception:
                                    st.progress(int(docking_score) / 100.0)
                            except Exception as e:
                                st.error(f"Could not compute properties: {e}")

                            st.subheader("🧬 3-letter Sequence")
                            if HAS_BIOPY:
                                try:
                                    aa3 = [seq3(res) for res in msg]
                                    st.write(" - ".join(aa3))
                                except Exception:
                                    st.write("Could not compute 3-letter sequence.")
                            else:
                                st.write("Biopython not installed: cannot convert to 3-letter codes.")

                            st.divider()
                            st.subheader("🧬 PDB Structure")
                            st.warning(
                                "⚠️ **Note**: This is a simple predicted structure based on peptide builder. "
                                "For practical usage such as molecular docking and other computational studies, "
                                "use **ColabFold** (available in the Structure tab) to generate high-confidence "
                                "AI-predicted 3D structures."
                            )
                            # Auto-generate PDB file
                            try:
                                fname, pdb_bytes = save_pdb_bytes(msg)
                                st.success(f"✅ Generated: {fname}")
                                st.download_button(
                                    label="📥 Download PDB",
                                    data=pdb_bytes,
                                    file_name=fname,
                                    mime="chemical/x-pdb",
                                    key=f"pdb_download_{msg}"
                                )
                            except Exception as e:
                                st.error(f"PDB generation failed: {e}")

# Batch tab
with tabs[1]:
    st.subheader("📤 Batch Prediction from File")
    uploaded = st.file_uploader("Upload CSV or Excel with a 'peptide' column", type=["csv", "xls", "xlsx"])

    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            df_up = None

        if df_up is not None:
            df_up.columns = df_up.columns.str.strip()
            if "peptide" not in df_up.columns:
                st.error("File must contain 'peptide' column.")
            else:
                df_up["_peptide_norm"] = df_up["peptide"].astype(str).str.strip().str.upper()
                df_up["_valid"] = df_up["_peptide_norm"].apply(lambda s: validate_sequence(s)[0])
                if not df_up["_valid"].any():
                    st.error("No valid peptide sequences found.")
                else:
                    with st.spinner("Computing features and predicting..."):
                        feats_up = compute_features(df_up.loc[df_up["_valid"], "_peptide_norm"].tolist())
                    if feats_up.empty:
                        st.error("Feature computation produced no results.")
                    elif model is None:
                        st.error("Model unavailable for predictions.")
                    else:
                        try:
                            preds = model.predict(feats_up)
                            df_up.loc[df_up["_valid"], "Predicted_Taste"] = preds
                            
                            # Add properties for each valid peptide
                            valid_seqs = df_up.loc[df_up["_valid"], "_peptide_norm"].tolist()
                            properties_list = []
                            for seq in valid_seqs:
                                try:
                                    props = compute_peptide_properties(seq)
                                    properties_list.append(props)
                                except Exception:
                                    properties_list.append({})
                            
                            # Create properties dataframe
                            props_df = pd.DataFrame(properties_list)
                            
                            # Add properties columns to the main dataframe
                            df_up.loc[df_up["_valid"], props_df.columns] = props_df.values
                            
                            # Prepare download dataframe (remove internal columns)
                            download_df = df_up.drop(columns=["_peptide_norm", "_valid"])
                            
                            st.write("Batch predictions with properties:")
                            st.dataframe(download_df)
                            st.download_button(
                                "📥 Download CSV with Predictions & Properties", 
                                data=download_df.to_csv(index=False).encode(), 
                                file_name="predictions_with_properties.csv", 
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Batch prediction failed: {e}")

# Structure tab (py3Dmol)
with tabs[4]:
    st.subheader("🧬 Structure Visualization (upload PDB)")
    st.markdown(
        "You can generate a predicted 3D structure for your peptide using "
        "**[ColabFold]("
        "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
        ")**:\n"
        "1. Open the link above in Google Colab.\n"
        "2. Paste your peptide sequence in FASTA format.\n"
        "3. Run the notebook (needs a GPU runtime).\n"
        "4. Download the predicted PDB file.\n"
        "5. Upload the PDB file below to visualize it here."
    )
    uploaded_pdb = st.file_uploader("Upload PDB file", type=["pdb"])
    def _show_pdb_in_py3dmol(pdb_text: str, width: int = 700, height: int = 450):
        if not HAS_PY3DMOL:
            st.error("py3Dmol is not installed. Install with `pip install py3Dmol` to use the viewer.")
            return
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_text, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        st.components.v1.html(view._make_html(), height=height)
    if uploaded_pdb is not None:
        try:
            pdb_text = uploaded_pdb.read().decode("utf-8")
            _show_pdb_in_py3dmol(pdb_text)
        except Exception as e:
            st.error(f"Failed to render PDB: {e}")
    else:
        st.info("No PDB uploaded yet. Use ColabFold to predict structures and upload the PDB file.")

# Footer
st.markdown("---")
st.markdown(" Feature computation uses Biopython; PDB generation uses PeptideBuilder.")
