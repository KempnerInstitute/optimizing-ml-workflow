#  Distributed Hyperparameter Sweep with SLURM Array Jobs

This workflow demonstrates how to perform a distributed hyperparameter sweep using SLURM array jobs for CIFAR-10 training with PyTorch. Parameter combinations are automatically generated and executed in parallel using the SLURM workload manager. 


---

## Step 1: Define Sweep Parameters

The hyperparameter space is defined in `generate_param_comb.py`. You can edit this script to specify which parameters and values you'd like to sweep over.

Example snippet from `generate_param_comb.py`:

```python
param_grid = {
    "learning_rate": [0.01, 0.1],
    "batch_size": [32, 64],
    "optimizer": ["adam", "sgd"],
    "weight_decay": [0.0001, 0.001],
    "model": ["resnet18", "alexnet"]
}
```

---

##  Step 2: Generate Parameter Combinations

Once you've defined your sweep space, run the script to generate all combinations:

```bash
python generate_param_comb.py
```

This will create a file named:

```
parameter_combinations.csv
```

Each row in this CSV corresponds to a different training configuration that will be launched as a separate SLURM array job task.

---

##  Step 3: Submit SLURM Array Job

Use the provided SLURM script to launch the parameter sweep:

```bash
sbatch torch_vision_cifar10_arr.slrm
```

Each array task will read a specific row from `parameter_combinations.csv` and run the training script `torch_vision_cifar10.py` with the corresponding parameters.

---

##  Requeue Failed Jobs

Sometimes jobs may fail due to hardware or runtime issues. You can requeue failed jobs using one of the following bash snippets:

### Option 1: Use `sacct`

```bash
job_id=<ENTER JOB ID>
for i in $(sacct -j $job_id | grep FAILED | grep -v batch | awk '{ print $1 }'); do 
    scontrol requeue $i 
done
```

### Option 2: Use `squeue`

```bash
job_id=<ENTER JOB ID>
for i in $(squeue -j $job_id --array --states=FAILED | awk 'NR > 1 { print $1 }'); do 
    scontrol requeue $i 
done
```

In both methods:
- Replace `<ENTER JOB ID>` with your actual SLURM array job ID.
- These commands identify failed subjobs and requeue them individually with `scontrol requeue`.

---

## 📁 Output and Logging

Each job will write its output (`.out`) and error (`.err`) files with names including the SLURM array task ID inside the directory "log_slurm". 


---

##  Analyzing Results

After the sweep is complete, you can parse and aggregate the results—such as training logs—into a summary CSV file using the script extract_sweep_results.py. The output CSV will be sorted by validation loss, making it easy to identify the best-performing configuration. You can then retrieve the optimal parameters and corresponding model checkpoint paths from the final column of the CSV.



## Additional information

Here is a generic example of doing an Slurm array job hyperparameter tuning (https://github.com/KempnerInstitute/examples/array-job)

