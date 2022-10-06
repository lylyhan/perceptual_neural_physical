import os

# Define constants.
script_name = os.path.basename(__file__)
script_path = os.path.abspath(os.path.join("..", script_name))
save_dir = "/scratch/vl1019/icassp23_data"

# Create folder.
sbatch_dir = os.path.join(".", os.path.basename(__file__)[:-3])
os.makedirs(sbatch_dir, exist_ok=True)

file_name = script_name[:2] + ".sbatch"
file_path = os.path.join(sbatch_dir, file_name)

# Generate file.
with open(file_path, "w") as f:
    cmd_args = [script_path, save_dir]

    f.write("#!/bin/bash\n")
    f.write("\n")
    f.write("#BATCH --job-name=" + script_name + "\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --tasks-per-node=1\n")
    f.write("#SBATCH --cpus-per-task=4\n")
    f.write("#SBATCH --time=24:00:00\n")
    f.write("#SBATCH --mem=8GB\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --output=" + job_name + "_%j.out\n")
    f.write("\n")
    f.write("module purge\n")
    f.write("module load cuda/11.6.2\n")
    f.write("module load ffmpeg/4.2.4\n")
    f.write("\n")
    f.write(" ".join([
        "singularity exec",
        "--overlay /scratch/vl1019/overlay-50G-10M.ext3:ro",
        "/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif",
        "/bin/bash",
        "-c",
        "\"source",
            "/scratch/vl1019/env.sh;",
            "python"] + cmd_args) + "\"\n")
    f.write("\n")
