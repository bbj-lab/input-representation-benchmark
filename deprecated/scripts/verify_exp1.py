import re
import numpy as np
from collections import defaultdict

def parse_metrics(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    results = defaultdict(list)
    current_config = None
    is_fused = None
    
    for line in content.split('\n'):
        if line.startswith('exp1 |'):
            current_config = line.split('|')[1].strip()
            is_fused = '_fused_time_tokens' in current_config
            continue
            
        if current_config and ':' in line and 'R2=' in line and 'Rho=' in line:
            var_name = line.split(':')[0].strip()
            rho_val = float(re.search(r'Rho=([\d\.\-]+)', line).group(1))
            
            fused_key = 'Fused' if is_fused else 'Unfused'
            results[fused_key + '_' + var_name].append(rho_val)
            
        if line.startswith('exp3 |') or line.startswith('NEW MLP') or line.startswith('NEW LR KEY'):
             current_config = None

    print("Experiment 1 Means:")
    vars = ['time_to_icu_hours', 'peak_creatinine', 'min_hemoglobin', 'peak_potassium', 'min_glucose', 'peak_troponin', 'peak_bnp']
    for fuse in ['Unfused', 'Fused']:
        print(f"--- {fuse} ---")
        for v in vars:
            vals = results[fuse + '_' + v]
            if vals:
                print(f"{v}: {np.mean(vals):.3f} (from {len(vals)} values)")
            else:
                 print(f"{v}: MISSING")

if __name__ == '__main__':
    parse_metrics('metrics_dump_exp1_truelab.txt')
