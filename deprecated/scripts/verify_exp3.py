import re
import numpy as np
from collections import defaultdict

def parse_metrics(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    results = defaultdict(list)
    current_config = None
    
    for line in content.split('\n'):
        if line.startswith('exp3 |'):
            current_config = line.split('|')[1].strip()
            continue
            
        if current_config and ':' in line and 'R2=' in line and 'Rho=' in line:
            var_name = line.split(':')[0].strip()
            rho_val = float(re.search(r'Rho=([\d\.\-]+)', line).group(1))
            
            results[current_config + '_' + var_name].append(rho_val)
            
        if line.startswith('NEW MLP') or line.startswith('NEW LR KEY') or line.startswith('exp1 |'):
             current_config = None

    print("Experiment 3 Configs:")
    configs = ['meds_freqmatched', 'meds_mapped', 'meds_randomized', 'meds_icu']
    vars = ['time_to_icu_hours', 'peak_creatinine', 'min_hemoglobin', 'peak_potassium', 'min_glucose', 'peak_troponin', 'peak_bnp']
    for conf in configs:
        print(f"--- {conf} ---")
        line = []
        for v in vars:
            vals = results[conf + '_' + v]
            if vals:
                line.append(f"{np.mean(vals):.3f}")
            else:
                line.append("MISSING")
        print(" & ".join(line))

if __name__ == '__main__':
    parse_metrics('metrics_dump_exp1_truelab.txt')
