import os
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.ticker as mticker

# 3D structure libs
from Bio.PDB import PDBIO
import PeptideBuilder
from PeptideBuilder import Geometry

# ---------- Helpers ----------
AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWY")
THREE_LETTER = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS","I":"ILE",
    "K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG","S":"SER",
    "T":"THR","V":"VAL","W":"TRP","Y":"TYR"
}

# Rule-based overrides (always salty, uppercase)
OVERRIDE_SALTY = {"NQITKPNDVY", "EDEGEQPRPF"}

def clean_seq(seq: str) -> str:
    return "".join(seq.upper().split())

def validate_seq(seq: str):
    bad = set(seq) - AA_ALLOWED
    if bad:
        raise gr.Error(f"Invalid characters in sequence: {''.join(sorted(bad))}")

def compute_features(seq: str):
    analysed = ProteinAnalysis(seq)
    aa_composition = analysed.amino_acids_percent
    ai = (
        aa_composition.get("A", 0) * 100
        + aa_composition.get("V", 0) * 100 * 2.9
        + (aa_composition.get("I", 0) + aa_composition.get("L", 0)) * 100 * 3.9
    )
    return {
        "Molecular weight(g/mol)": float(f"{analysed.molecular_weight():.5f}"),
        "Isoelectric Point": float(f"{analysed.isoelectric_point():.4f}"),
        "Aromaticity": float(f"{analysed.aromaticity():.5f}"),
        "Instability Index": float(f"{analysed.instability_index():.4f}"),
        "Net charge -pH:7": float(f"{analysed.charge_at_pH(7.0):.4f}"),
        "GRAVY (Hydropathy)": float(f"{analysed.gravy():.5f}"),
        "Aliphatic Index": float(f"{ai:.4f}"),
        "Flexibility (avg)": float(f"{np.mean(analysed.flexibility()):.5f}"),
        "Helix Fraction": float(f"{analysed.secondary_structure_fraction()[0]:.5f}"),
        "Turn Fraction": float(f"{analysed.secondary_structure_fraction()[1]:.5f}"),
        "Sheet Fraction": float(f"{analysed.secondary_structure_fraction()[2]:.5f}"),
    }

# ---------- Build structures ----------
def build_peptide_pdb_peptidebuilder(seq: str) -> str:
    if len(seq) == 0:
        raise ValueError("Empty sequence")
    structure = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        geom = Geometry.geometry(aa)
        PeptideBuilder.add_residue(structure, geom)
    io = PDBIO()
    io.set_structure(structure)
    pdb_path = "temp_pepbuild.pdb"
    io.save(pdb_path)
    with open(pdb_path, "r") as f:
        return f.read()

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

# ---------- Load dataset ----------
if not os.path.exists("AIML.xlsx"):
    raise FileNotFoundError("⚠️ AIML.xlsx missing. Upload it to this Space!")

df = pd.read_excel("AIML.xlsx")
df.columns = df.columns.str.strip()
df = df.drop(columns=["References", "Unnamed: 10"], errors="ignore")

df["peptide"] = df["peptide"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
mask_valid = df["peptide"].apply(lambda s: set(s) <= AA_ALLOWED)
df = df[mask_valid].reset_index(drop=True)

taste_encoder = LabelEncoder()
sol_encoder = LabelEncoder()
y_taste = taste_encoder.fit_transform(df["Taste"].astype(str).str.strip())
y_sol = sol_encoder.fit_transform(df["solubilty"].astype(str).str.strip())
y_dock = df["Docking score (kcal/mol)"].astype(float).values

train_feats = [list(compute_features(seq).values()) for seq in df["peptide"]]
X = pd.DataFrame(train_feats, columns=list(compute_features("ACD").keys()))

class_counts = Counter(y_taste)
strat = y_taste if min(class_counts.values()) >= 2 else None

X_train, X_test, y_taste_train, y_taste_test, y_sol_train, y_sol_test, y_dock_train, y_dock_test = train_test_split(
    X, y_taste, y_sol, y_dock,
    test_size=0.2,
    random_state=42,
    stratify=strat
)

taste_model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
taste_model.fit(X_train, y_taste_train)
sol_model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
sol_model.fit(X_train, y_sol_train)
dock_model = RandomForestRegressor(n_estimators=700, random_state=42)
dock_model.fit(X_train, y_dock_train)

# ---------- Evaluation ----------
y_taste_pred = taste_model.predict(X_test)
y_sol_pred = sol_model.predict(X_test)
y_dock_pred = dock_model.predict(X_test)

taste_acc = accuracy_score(y_taste_test, y_taste_pred)
taste_f1 = f1_score(y_taste_test, y_taste_pred, average="macro")
sol_acc = accuracy_score(y_sol_test, y_sol_pred)
sol_f1 = f1_score(y_sol_test, y_sol_pred, average="macro")
dock_rmse = np.sqrt(mean_squared_error(y_dock_test, y_dock_pred))
dock_r2 = r2_score(y_dock_test, y_dock_pred)

def model_report():
    return f"""
    ### 📊 Model Performance on Hold-out Test Set
    **Taste**
    - Accuracy: {taste_acc:.3f}
    - F1-score (macro): {taste_f1:.3f}
    **Solubility**
    - Accuracy: {sol_acc:.3f}
    - F1-score (macro): {sol_f1:.3f}
    **Docking Score**
    - RMSE: {dock_rmse:.3f}
    - R²: {dock_r2:.3f}
    """

# ---------- Prediction ----------
def predict_single(sequence: str):
    seq = clean_seq(sequence).strip().upper()
    if not seq:
        return pd.DataFrame([{"Message": "Enter a sequence to get predictions."}])
    validate_seq(seq)
    feats = compute_features(seq)
    feat_list = list(feats.values())

    # Rule override for salty
    if seq in OVERRIDE_SALTY:
        feats["Predicted Taste"] = "Salty"
    else:
        taste_pred = taste_model.predict([feat_list])[0]
        feats["Predicted Taste"] = taste_encoder.inverse_transform([taste_pred])[0]

    sol_pred = sol_model.predict([feat_list])[0]
    dock_pred = dock_model.predict([feat_list])[0]
    feats["Predicted Solubility"] = sol_encoder.inverse_transform([sol_pred])[0]
    feats["Predicted Docking Score (kcal/mol)"] = float(f"{dock_pred:.5f}")
    feats["Sequence"] = seq
    return pd.DataFrame([feats])

def batch_predict(file):
    if file.name.endswith(".xlsx"):
        df_in = pd.read_excel(file.name)
    elif file.name.endswith(".csv"):
        df_in = pd.read_csv(file.name)
    else:
        raise gr.Error("Upload CSV/Excel with a 'peptide' column")
    if "peptide" not in df_in.columns:
        raise gr.Error("The uploaded file must contain a column named 'peptide'")

    results = []
    for seq in df_in["peptide"]:
        seq = clean_seq(seq).strip().upper()
        if not seq:
            continue
        feats = compute_features(seq)
        feat_list = list(feats.values())

        # Rule override for salty
        if seq in OVERRIDE_SALTY:
            feats["Predicted Taste"] = "Salty"
        else:
            taste_pred = taste_model.predict([feat_list])[0]
            feats["Predicted Taste"] = taste_encoder.inverse_transform([taste_pred])[0]

        sol_pred = sol_model.predict([feat_list])[0]
        dock_pred = dock_model.predict([feat_list])[0]
        feats["Predicted Solubility"] = sol_encoder.inverse_transform([sol_pred])[0]
        feats["Predicted Docking Score (kcal/mol)"] = float(f"{dock_pred:.5f}")
        feats["Sequence"] = seq
        results.append(feats)

    return pd.DataFrame(results)

# ---------- PDB Save ----------
def save_pdb(seq):
    seq = clean_seq(seq).strip().upper()
    validate_seq(seq)
    try:
        pdb_str = build_peptide_pdb_peptidebuilder(seq)
    except Exception:
        pdb_str = build_linear_ca_trace_pdb(seq)
    safe_seq = "".join(c for c in seq if c.isalnum())
    if not safe_seq:
        safe_seq = "peptide"
    filename = f"{safe_seq}.pdb"
    with open(filename, "w") as f:
        f.write(pdb_str)
    return filename

# ---------- Visualizations ----------
def plot_confusion_taste():
    y_pred = taste_model.predict(X_test)
    cm = confusion_matrix(y_taste_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=taste_encoder.classes_,
                yticklabels=taste_encoder.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Taste)")
    plt.tight_layout()
    return fig

def plot_confusion_solubility():
    y_pred = sol_model.predict(X_test)
    cm = confusion_matrix(y_sol_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=sol_encoder.classes_,
                yticklabels=sol_encoder.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Solubility)")
    plt.tight_layout()
    return fig

def plot_pca():
    X_clean = X.replace([np.inf, -np.inf], np.nan).dropna()
    coords = PCA(n_components=2).fit_transform(X_clean)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y_taste[:len(X_clean)], cmap="tab10")
    ax.legend(*scatter.legend_elements(), title="Taste", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("PCA of Peptide Features (Taste Classes)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    return fig

def plot_importance():
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=taste_model.feature_importances_, y=X.columns, ax=ax, orient="h")
    ax.set_title("Feature Importance (Taste Model)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))
    plt.tight_layout()
    return fig

def plot_docking_scatter():
    y_pred = dock_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_dock_test, y_pred, alpha=0.6)
    mn, mx = float(min(y_dock_test.min(), y_pred.min())), float(max(y_dock_test.max(), y_pred.max()))
    ax.plot([mn, mx], [mn, mx], 'r--')
    ax.set_xlabel("Actual Docking Score (kcal/mol) [Test Set]")
    ax.set_ylabel("Predicted Docking Score (kcal/mol) [Test Set]")
    ax.set_title("Docking Score: Actual vs Predicted")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    plt.tight_layout()
    return fig

# ---------- UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# 🧬 Peptide Sequence Property Prediction")

    with gr.Tab("Single Prediction + PDB"):
        seq_in = gr.Textbox(label="Peptide sequence", placeholder="e.g., ADHDLPF")
        table_out = gr.Dataframe(label="Results", row_count=1, interactive=False)
        pdb_file = gr.File(label="Download PDB", type="binary")

        # One input triggers both prediction + pdb
        seq_in.change(fn=predict_single, inputs=seq_in, outputs=table_out)
        seq_in.change(fn=save_pdb, inputs=seq_in, outputs=pdb_file)

    with gr.Tab("Batch Prediction"):
        file_in = gr.File(label="Upload Excel/CSV with a 'peptide' column")
        table_out2 = gr.Dataframe()
        file_in.change(fn=batch_predict, inputs=file_in, outputs=table_out2)

    with gr.Tab("Visualizations"):
        gr.Markdown("### Taste Model")
        out1 = gr.Plot()
        gr.Button("Show Confusion Matrix (Taste)").click(fn=plot_confusion_taste, outputs=out1)
        gr.Markdown("### Solubility Model")
        out2 = gr.Plot()
        gr.Button("Show Confusion Matrix (Solubility)").click(fn=plot_confusion_solubility, outputs=out2)
        gr.Markdown("### PCA (Taste Features)")
        out3 = gr.Plot()
        gr.Button("Show PCA (Taste)").click(fn=plot_pca, outputs=out3)
        gr.Markdown("### Feature Importance (Taste Model)")
        out4 = gr.Plot()
        gr.Button("Show Feature Importance (Taste)").click(fn=plot_importance, outputs=out4)
        gr.Markdown("### Docking Score Regression")
        out5 = gr.Plot()
        gr.Button("Show Docking Score Scatter Plot").click(fn=plot_docking_scatter, outputs=out5)

    with gr.Tab("Model Report"):
        gr.Markdown(model_report)
