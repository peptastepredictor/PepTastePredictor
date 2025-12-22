import pandas as pd
from modlamp.descriptors import GlobalDescriptor
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Validation constants
VALID_AA = set(list("ACDEFGHIKLMNPQRSTVWY"))


def compute_features(seq_list):
    """Compute biochemical descriptors using modlamp for a list of sequences.
    Returns a pandas DataFrame with descriptor values (or empty DataFrame on failure).
    """
    if not seq_list:
        return pd.DataFrame()
    try:
        gd = GlobalDescriptor(seq_list)
        gd.calculate_all()
        return pd.DataFrame(gd.descriptor)
    except Exception:
        return pd.DataFrame()


def compute_peptide_properties(seq: str) -> dict:
    """Compute basic peptide properties using Biopython."""
    analysis = ProteinAnalysis(seq)

    props = {
        "Molecular Weight": round(analysis.molecular_weight(), 2),
        "Isoelectric Point": round(analysis.isoelectric_point(), 2),
        "Aromaticity": round(analysis.aromaticity(), 3),
        "Instability Index": round(analysis.instability_index(), 2),
        "Gravy (Hydrophobicity)": round(analysis.gravy(), 3)
    }
    return props


def validate_sequence(seq: str):
    """Return (True, cleaned_seq) or (False, reason)"""
    if seq is None:
        return False, "Empty sequence"
    s = str(seq).strip().upper()
    if len(s) < 2:
        return False, "Sequence too short"
    if len(s) > 200:
        return False, "Sequence too long (max 200 aa)"
    invalid = sorted(set([c for c in s if c not in VALID_AA]))
    if invalid:
        return False, f"Invalid characters in sequence: {''.join(invalid)}"
    return True, s


def compute_docking_score(gravy: float, confidence: float, w_conf: float = 0.7) -> int:
    """Compute a simple docking-style score (0-100) from GRAVY and model confidence.

    - gravy: GRAVY hydrophobicity score from Biopython (approx range -2 to +2)
    - confidence: model confidence for the predicted class (0.0 - 1.0)
    - w_conf: weight for model confidence (default 0.7), remainder is gravy contribution

    The gravy contribution is normalized from [-2,2] -> [0,1]. The final score is
    a weighted combination, scaled to 0-100 and returned as an int.
    """
    # Normalize gravy (clamp to expected range)
    try:
        g = float(gravy)
    except Exception:
        g = 0.0
    norm_g = (g + 2.0) / 4.0
    norm_g = max(0.0, min(1.0, norm_g))

    try:
        c = float(confidence)
    except Exception:
        c = 0.0
    c = max(0.0, min(1.0, c))

    w_g = 1.0 - float(w_conf)
    score = (w_conf * c + w_g * norm_g) * 100.0
    return int(round(max(0.0, min(100.0, score))))
