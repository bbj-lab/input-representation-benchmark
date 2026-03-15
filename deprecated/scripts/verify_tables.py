import os
import glob
import pickle
import numpy as np
import sklearn.metrics
import scipy.stats

def verify_all_results():
    files = glob.glob("**/*test/*preds*.pkl", recursive=True)
    results = {}
    for f in sorted(files):
        try:
            with open(f, "rb") as pkl:
                data = pickle.load(pkl)
            
            outcomes_res = {}
            for outcome, y_pred in data.get("predictions", {}).items():
                y_true = data["labels"][outcome]
                
                # Try to apply qualifiers to y_true
                if "qualifiers" in data and outcome in data["qualifiers"]:
                    qual = data["qualifiers"][outcome]
                    y_true = np.array(y_true)[qual]
                
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)

                is_reg = "ridge" in f or "continuous" in outcome or outcome in ['peak_creatinine', 'peak_troponin', 'min_hemoglobin', 'peak_potassium', 'min_glucose', 'peak_bnp', 'time_to_icu_hours', 'los_hours']
                
                if is_reg:
                    try:
                        rho, _ = scipy.stats.spearmanr(y_true, y_pred)
                        outcomes_res[outcome] = {"rho": rho}
                    except Exception as e:
                        outcomes_res[outcome] = {"error": str(e)}
                else:
                    try:
                        auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                        auprc = sklearn.metrics.average_precision_score(y_true, y_pred)
                        outcomes_res[outcome] = {"auc": auc, "auprc": auprc}
                    except Exception as e:
                        outcomes_res[outcome] = {"error": str(e)}
            
            results[f] = outcomes_res
        except Exception as e:
            results[f] = {"error": str(e)}
            
    with open("verification_dump.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}\n")
            if isinstance(v, dict):
                for out, met in v.items():
                    f.write(f"  {out}: {met}\n")
            else:
                f.write(f"  {v}\n")

if __name__ == "__main__":
    verify_all_results()
