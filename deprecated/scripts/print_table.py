import collections
import re

def aggregate_to_table():
    with open('verification_dump.txt', 'r') as f:
        lines = f.readlines()

    metrics = collections.defaultdict(dict)
    current_file = None
    
    for line in lines:
        if line.startswith('benchmarks/') or line.startswith('data/'):
            current_file = line.strip()
        elif current_file and ':' in line:
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                outcome = parts[0]
                val_str = parts[1]
                if 'auc' in val_str:
                    auc = float(re.search(r"'auc': ([\d\.]+)", val_str).group(1))
                    auprc = float(re.search(r"'auprc': ([\d\.]+)", val_str).group(1))
                    
                    if "logistic_regression" not in current_file: 
                        continue
                    
                    # extract simple config prefix
                    if "benchmarks" in current_file:
                        conf = current_file.split("/data/")[2].split("_first_24h")[0]
                        conf = conf.replace("_evalL4096", "")
                    else:
                        conf = current_file.split("/")[3].split("_first_24h")[0]
                        conf = conf.replace("_evalL4096", "")
                        
                    if outcome not in metrics[conf]:
                        metrics[conf][outcome] = {'auc': auc, 'auprc': auprc}
                    else:
                        metrics[conf][outcome]['auc'] = max(metrics[conf][outcome]['auc'], auc)
                        metrics[conf][outcome]['auprc'] = max(metrics[conf][outcome]['auprc'], auprc)

    print("Experiment 1 and 2 checks:")
    for conf in sorted(metrics.keys()):
        print(f"\n{conf}")
        line_auc = []
        line_auprc = []
        for out in ['same_admission_death', 'long_length_of_stay', 'icu_admission', 'imv_event']:
            if out in metrics[conf]:
                line_auc.append(f"{metrics[conf][out]['auc']:.3f}")
                line_auprc.append(f"{metrics[conf][out]['auprc']:.3f}")
            else:
                line_auc.append("---")
                line_auprc.append("---")
        print("AUC: " + " | ".join(line_auc))
        print("AUPC: " + " | ".join(line_auprc))

if __name__ == '__main__':
    aggregate_to_table()
