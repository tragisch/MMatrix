import csv
from collections import defaultdict

def load_data(file):
    data = []
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['case'] in ['conv_medium', 'conv_large', 'pw_medium'] and \
               row['framework'] in ['c_gemm', 'c_mps_zero_copy_sync', 'c_mps_true_async_boundary', 'pytorch_mps', 'mlx']:
                data.append((row['case'], row['framework'], float(row['median_ms'])))
    return data

d1 = load_data('/tmp/cross_run1_r5.csv')
d2 = load_data('/tmp/cross_run2_r5.csv')

combined = defaultdict(list)
for c, f, m in d1 + d2:
    combined[(c, f)].append(m)

results = {}
for (c, f), vals in combined.items():
    if c not in results: results[c] = {}
    results[c][f] = sorted(vals)[len(vals)//2] # median of the runs

frameworks = ['c_gemm', 'c_mps_zero_copy_sync', 'c_mps_true_async_boundary', 'pytorch_mps', 'mlx']
m_matrix_fws = ['c_gemm', 'c_mps_zero_copy_sync', 'c_mps_true_async_boundary']
ext_fws = ['pytorch_mps', 'mlx']

print(f"{'case':<12} | {'c_gemm':>8} | {'zcs':>8} | {'tab':>8} | {'pt_mps':>8} | {'mlx':>8} | {'best_m':>8} | {'best_e':>8} | {'fac':>5}")
print("-" * 110)

for case in ['conv_medium', 'conv_large', 'pw_medium']:
    row = results.get(case, {})
    vals = [row.get(f, 0) for f in frameworks]
    best_m = min([row.get(f, float('inf')) for f in m_matrix_fws])
    best_e = min([row.get(f, float('inf')) for f in ext_fws])
    factor = best_e / best_m if best_m > 0 and best_m != float('inf') else 0
    
    line = f"{case:<12} | " + " | ".join([f"{v:8.2f}" for v in vals])
    line += f" | {best_m:8.2f} | {best_e:8.2f} | {factor:5.2f}"
    print(line)
