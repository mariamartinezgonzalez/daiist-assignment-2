"""Cost model, in euros per case (one case = one image).
Every number is my own assumption and is justified in REPORT.md."""
C_FP = 5
C_FN = 500
C_TP = 5
C_TN = 0
C_REVIEW = 2
BASE_RATE = 0.3

def require_filled():
    missing = [k for k, v in globals().items() if k.isupper() and v is None]
    if missing:
        raise SystemExit(f"Fill in costs.py first, missing: {', '.join(missing)}")
