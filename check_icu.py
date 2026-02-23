import pickle
import numpy as np

path = "data/exp2/deciles_none_unfused_time_rope_first_24h-tokenized/test/mlp-preds-model-discrete-time_rope.pkl"
try:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "icu_admission" in d["predictions"]:
        probs = d["predictions"]["icu_admission"]
        labels = d["labels"]["icu_admission"]
        print("Probs min:", np.min(probs), "max:", np.max(probs), "var:", np.var(probs))
        print("First 10 probs:", probs[:10])
except Exception as e:
    print(e)
