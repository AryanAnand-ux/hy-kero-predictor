"""
run_pipeline.py
───────────────
One-click script to run the entire ML pipeline:
  1. Preprocess raw data → merged_dataset.csv
  2. Engineer features  → train/test splits
  3. Train all models + fine-tune → best_model.pkl + metrics

Usage:
  python run_pipeline.py
"""

import sys
import os

# Ensure UTF-8 output
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import run as run_preprocess
from features   import run as run_features
from train      import run as run_train


def main():
    print("=" * 60)
    print("  HY Kero Flash Point — Full ML Pipeline")
    print("=" * 60)

    # Phase 1
    run_preprocess()

    # Phase 2
    run_features()

    # Phase 3 + 4
    results, best = run_train()

    print("\n" + "=" * 60)
    print(f"  [BEST MODEL] : {best}")
    print(f"  Test RMSE: {results[best]['test']['rmse']:.3f}")
    print(f"  Test R2:   {results[best]['test']['r2']:.4f}")
    print(f"  Test MAPE: {results[best]['test']['mape']:.2f}%")
    print("=" * 60)
    print("\n  [OK] Pipeline complete! You can now:")
    print("    1. Start the backend:  cd backend && uvicorn main:app --reload")
    print("    2. Start the frontend: cd frontend && npm run dev")
    print()


if __name__ == "__main__":
    main()
