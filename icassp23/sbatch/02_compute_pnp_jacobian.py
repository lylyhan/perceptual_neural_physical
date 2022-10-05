import os
import sys

sys.path.append("../src")


# Define constants.
id_max = 100000
n_threads = 500
n_per_th = id_max // n_threads
script_name = os.path.basename(__file__)
script_path = os.path.abspath(os.path.join("..", script_name))
save_dir = "/scratch/vl1019/icassp23_data"

# Create folder.
sbatch_dir = os.path.join(".", script_name[:-3])
os.makedirs(sbatch_dir, exist_ok=True)


for n_thread in range(n_threads):

    job_name = "_".join(
        [script_name[:2], "thread-" + str(n_thread).zfill(len(str(n_threads)))]
    )
    file_name = job_name + ".sbatch"
    file_path = os.path.join(sbatch_dir, file_name)

    # Generate file.
    with open(file_path, "w") as f:
        id_start = n_thread * n_per_th
        id_end = (n_thread + 1) * n_per_th
        if n_thread == n_threads - 1:
            id_end = max(id_end, id_max)
        cmd_args = [script_path, save_dir, str(id_start), str(id_end)]

        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write("#BATCH --job-name=" + script_name + "\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --tasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --time=4:00:00\n")
        f.write("#SBATCH --mem=32GB\n")
        f.write("#SBATCH --output=" + job_name + "_%j.out\n")
        f.write("\n")
        f.write("module purge\n")
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


# Open shell file.
file_path = os.path.join(sbatch_dir, script_name[:2] + ".sh")

with open(file_path, "w") as f:
    # Print header.
    f.write(
        "# This shell script computes scattering features and "
        "the associated Riemannian metric."
    )
    f.write("\n")

    # Loop over folds: training and validation.
    for n_thread in range(n_threads):
        # Define job name.
        job_name = "_".join(
            [script_name[:2], "thread-" + str(n_thread).zfill(len(str(n_threads)))]
        )
        sbatch_str = "sbatch " + job_name + ".sbatch"
        # Write SBATCH command to shell file.
        f.write(sbatch_str + "\n")

# Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(file_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(file_path, mode)
