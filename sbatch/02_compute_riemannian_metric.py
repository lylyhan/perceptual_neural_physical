import os
import sys

sys.path.append("../src")


# Define constants.
id_max = 100000
n_threads = 100
n_per_th = id_max // n_threads
script_name = os.path.basename(__file__)
script_path = os.path.abspath(os.path.join("..", "icassp23", script_name))
csv_path = os.path.abspath(os.path.join("..", "data"))
data_path = "/scratch/vl1019/icassp23_data"

# Create folder.
sbatch_dir = os.path.join(".", os.path.basename(__file__)[:-3])
os.makedirs(sbatch_dir, exist_ok=True)


for n_thread in range(n_threads):

    job_name = "_".join(
        [script_name[:2], "thread-" + str(n_thread).zfill(len(str(n_threads)))]
    )
    file_name = job_name + ".sbatch"
    file_path = os.path.join(sbatch_dir, file_name)

    # Generate file.
    with open(file_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write("#BATCH --job-name=" + script_name[:2] + "\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --tasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=4\n")
        f.write("#SBATCH --time=1:00:00\n")
        f.write("#SBATCH --mem=16GB\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH --output=" + job_name + "_%j.out\n")
        f.write("\n")
        f.write("module purge\n")
        f.write("module load cuda/11.6.2\n")
        f.write("module load ffmpeg/4.2.4\n")
        f.write("\n")

        id_start = n_thread * n_per_th
        id_end = id_start + 1  # (n_thread + 1) * n_per_th
        if n_thread == n_threads - 1:
            id_end = max(id_end, id_max)
        f.write(
            " ".join(
                [
                    "python",
                    script_path,
                    data_path,
                    csv_path,
                    str(id_start),
                    str(id_end) + "\n",
                ]
            )
        )


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
        f.write("\n")

# Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(file_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(file_path, mode)