import os
import glob
import pickle

def extract_metrics(pattern, metrics):
    files = glob.glob(pattern, recursive=True)
    results = {}
    for f in sorted(files):
        # Extract meaningful subpath to identify the model
        parts = f.split("/")
        if "exp2" in parts:
            exp = "exp2"
            cond = parts[parts.index("exp2")+1]
        elif "exp3" in parts:
            exp = "exp3"
            cond = parts[parts.index("exp3")+2] if "arms" in f else parts[parts.index("exp3")+1]
        elif "exp1" in parts or "evalL4096" in f:
            exp = "exp1"
            if "exp1" in parts:
                cond = parts[parts.index("exp1")+1]
            else:
                # the folder before 'test' is the condition
                cond = os.path.basename(os.path.dirname(os.path.dirname(f)))
        else:
            continue
            
        stem = os.path.basename(f)
        try:
            with open(f, "rb") as pkl:
                data = pickle.load(pkl)
            
            # Recalculate or rely on logged metrics? The script currently doesn't save the exact metrics
            # inside the PKL, it just saves `predictions` and `labels`. I'll compute metrics here on the fly.
            
            # Since computing AUC requires sklearn, we'll do it if there are classification tasks.
            import sklearn.metrics
            import scipy.stats
            import numpy as np
            
            outcomes_res = {}
            for outcome, y_pred in data.get("predictions", {}).items():
                y_true = data["labels"][outcome]
                if "qualifiers" in data and outcome in data.get("qualifiers", {}):
                    y_true = y_true[data["qualifiers"][outcome]]
                if "reg" in stem:
                    # Regression
                    try:
                        r2 = sklearn.metrics.r2_score(y_true, y_pred)
                        rho, _ = scipy.stats.spearmanr(y_true, y_pred)
                        outcomes_res[outcome] = f"R2={r2:.3f} Rho={rho:.3f}"
                    except Exception as e:
                        outcomes_res[outcome] = f"Err: {e}"
                else:
                    # Classification
                    try:
                        auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                        outcomes_res[outcome] = f"AUC={auc:.3f}"
                    except Exception as e:
                        outcomes_res[outcome] = f"Err: {e}"
            results[f"{exp} | {cond} | {stem}"] = outcomes_res
        except Exception as e:
            results[f"Error reading {f}"] = repr(e)
            
    return results

if __name__ == "__main__":
    print("-" * 50)
    print("NEW REGRESSION RESULTS")
    print("-" * 50)
    res_reg = extract_metrics("**/*test/reg_ridge_regression-preds*.pkl", [])
    for k, v in res_reg.items():
        print(f"\n{k}")
        if isinstance(v, dict):
            for out, met in v.items():
                print(f"  {out}: {met}")
        else:
            print(v)

    print("\n" + "-" * 50)
    print("NEW MLP PROBE RESULTS")
    print("-" * 50)
    res_mlp = extract_metrics("**/*test/mlp-preds*.pkl", [])
    for k, v in res_mlp.items():
        print(f"\n{k}")
        if isinstance(v, dict):
            for out, met in v.items():
                print(f"  {out}: {met}")
        else:
            print(v)

    print("\n" + "-" * 50)
    print("NEW LR RESULTS (ECE/AUC Checks)")
    print("-" * 50)
    res_lr = extract_metrics("**/*test/logistic_regression-preds*.pkl", [])
    for k, v in list(res_lr.items())[:5]: # just print a few to verify
        print(f"\n{k}")
        if isinstance(v, dict):
            for out, met in v.items():
                print(f"  {out}: {met}")
        else:
            print(v)
