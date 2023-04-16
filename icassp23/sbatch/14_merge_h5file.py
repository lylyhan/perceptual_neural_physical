import os

# Define constants.
script_name = os.path.basename(__file__)
script_path = os.path.abspath(os.path.join("..", script_name))

# Create folder.
job_name = os.path.basename(__file__)[:-3]
sbatch_dir = os.path.join(".", job_name)
os.makedirs(sbatch_dir, exist_ok=True)

file_name = job_name.split("_")[0] + ".sbatch"
file_path = os.path.join(sbatch_dir, file_name)

# Generate file.
with open(file_path, "w") as f:
    cmd_args = [script_path]

    f.write("#!/bin/bash\n")
    f.write("\n")
    f.write("#BATCH --job-name=" + script_name + "\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks-per-node=1\n")
    f.write("#SBATCH --cpus-per-task=15\n")
    f.write("#SBATCH --time=8:00:00\n")
    f.write("#SBATCH --mem=8GB\n")
    f.write("#SBTACH -A aej@v100\n")
    f.write("#SBATCH --output=" + job_name + "_%j.out\n")
    f.write("\n")
    f.write("module purge\n")
    f.write("module load pytorch-gpu/py3/1.12.1\n")
    f.write("\n")
    f.write(" ".join(["python"] + cmd_args) + "\"\n")
    f.write("\n")
