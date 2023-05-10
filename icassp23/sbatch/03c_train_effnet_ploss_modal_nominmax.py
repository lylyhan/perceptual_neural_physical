import os

# Define constants.
script_name = os.path.basename(__file__)[:-3] # remove .py
script_path = os.path.abspath(os.path.join("..", script_name)) + ".py"
save_dir = "/gpfswork/rech/rwb/ufg99no/data/icassp23_data"
n_inits = 1
batch_size = 256


for init_id in range(n_inits):
    job_name = "_".join([script_name, "init-" + str(init_id)])

    file_name = job_name + ".sbatch"
    file_path = os.path.join(".", file_name)

    # Generate file.
    with open(file_path, "w") as f:
        cmd_args = [script_path, save_dir, str(init_id), str(batch_size)]

        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write("#BATCH --job-name=" + script_name + "\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --tasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=10\n")
        f.write("#SBATCH --time=20:00:00\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBTACH -A aej@v100\n")
        f.write("#SBATCH --output=" + job_name + "_%j.out\n")
        f.write("\n")
        f.write("module purge\n")
        f.write("module load pytorch-gpu/py3/1.12.1\n")
        f.write("\n")
        f.write(" ".join(["python"] + cmd_args) + "\n")
        f.write("\n")


