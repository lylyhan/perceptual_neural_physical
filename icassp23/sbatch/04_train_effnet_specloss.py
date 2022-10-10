import os

# Define constants.
script_name = os.path.basename(__file__)[:-3] # remove .py
script_path = os.path.abspath(os.path.join("..", script_name)) + ".py"
save_dir = "/scratch/vl1019/icassp23_data"
n_inits = 10

# Create folder.
sbatch_dir = os.path.join(".", script_name)
os.makedirs(sbatch_dir, exist_ok=True)

for init_id in range(n_inits):
    job_name = "_".join([script_name, "init-" + str(init_id)])

    file_name = job_name + ".sbatch"
    file_path = os.path.join(sbatch_dir, file_name)

    # Generate file.
    with open(file_path, "w") as f:
        cmd_args = [script_path, save_dir, str(init_id)]

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
            "singularity exec --nv",
            "--overlay /scratch/vl1019/overlay-50G-10M.ext3:ro",
            "/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif",
            "/bin/bash",
            "-c",
            "\"source",
                "/scratch/vl1019/env.sh;",
                "python"] + cmd_args) + "\"\n")
        f.write("\n")


# Open shell file.
file_path = os.path.join(sbatch_dir, script_name.split("_")[0] + ".sh")

with open(file_path, "w") as f:
    # Print header.
    f.write(
        "# This shell script trains EfficientNet on multiscale spectral loss."
    )
    f.write("\n")

    # Loop over folds: training and validation.
    for init_id in range(n_inits):
        # Define job name.
        job_name = "_".join(
            [script_name[:2], "init-" + str(n_inits).zfill(len(str(init_id)))]
        )
        sbatch_str = "sbatch " + job_name + ".sbatch"
        # Write SBATCH command to shell file.
        f.write(sbatch_str + "\n")

# Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(file_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(file_path, mode)
