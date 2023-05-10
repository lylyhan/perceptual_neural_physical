import os

# Define constants.
script_name = os.path.basename(__file__)
script_path = os.path.abspath(os.path.join("..", script_name))
save_dir = "/gpfswork/rech/rwb/ufg99no/data/icassp23_data"

# Create folder.
job_name = os.path.basename(__file__)[:-3]
sbatch_dir = os.path.join(".", job_name)
os.makedirs(sbatch_dir, exist_ok=True)

file_name = job_name.split("_")[0] + ".sbatch"
file_path = os.path.join(sbatch_dir, file_name)

# Generate file.
with open(file_path, "w") as f:
    cmd_args = [script_path, save_dir]

    f.write("#!/bin/bash\n")
    f.write("\n")
    f.write("#BATCH --job-name=" + script_name + "\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks-per-node=1\n")
    f.write("#SBATCH --cpus-per-task=4\n")
    f.write("#SBATCH --time=1:00:00\n")
    f.write("#SBATCH --mem=8GB\n")
    f.write("#SBTACH -A aej@v100\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --output=" + job_name + "_%j.out\n")
    f.write("\n")
    f.write("module purge\n")
    f.write("module load cuda/11.6.2\n")
    f.write("module load ffmpeg/4.2.4\n")
    f.write("\n")
    f.write(" ".join(["python"] + cmd_args) + "\"\n")
    f.write("\n")
