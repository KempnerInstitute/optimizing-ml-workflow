import os
import re
import csv
from collections import OrderedDict

def extract_dynamic_params(param_line):
    param_line = param_line.strip()
    if param_line.startswith("Parameters:"):
        param_line = param_line[len("Parameters:"):].strip()

    tokens = param_line.split()
    params = OrderedDict()
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("--"):
            key = tokens[i].lstrip("-")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                params[key] = tokens[i + 1]
                i += 2
            else:
                params[key] = True  # flag without value
                i += 1
        else:
            # Positional argument (e.g., model name)
            params["positional_arg_{}".format(i)] = tokens[i]
            i += 1
    return params

def extract_lowest_val_loss(log_lines):
    val_losses = []
    for line in log_lines:
        match = re.search(r"Val Loss = ([0-9.]+)", line)
        if match:
            val_losses.append(float(match.group(1)))
    return min(val_losses) if val_losses else None

def process_experiment_dirs(base_path, output_csv):
    all_rows = []
    all_keys = set()

    for subdir in os.listdir(base_path):
        sub_path = os.path.join(base_path, subdir)
        if not os.path.isdir(sub_path):
            continue
        
        job_info_path = os.path.join(sub_path, "job_info.txt")
        log_path = os.path.join(sub_path, "training_log.txt")
        
        if not os.path.exists(job_info_path) or not os.path.exists(log_path):
            continue

        with open(job_info_path, "r") as f:
            lines = f.readlines()
        param_line = next((line for line in lines if "Parameters:" in line), "")
        params = extract_dynamic_params(param_line)

        with open(log_path, "r") as f:
            log_lines = f.readlines()
        lowest_val_loss = extract_lowest_val_loss(log_lines)

        params["lowest_val_loss"] = lowest_val_loss
        params["subdirectory_path"] = sub_path
        all_keys.update(params.keys())
        all_rows.append(params)

    all_keys = list(sorted(all_keys))  # consistent column order
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

base_path = "./results"  # Replace with actual path
output_csv = "./results_hpo_summary.csv"
process_experiment_dirs(base_path, output_csv)


