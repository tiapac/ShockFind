#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=hpcc250512a2
#SBATCH --job-name=shockfind
#SBATCH --output=shockfind_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --partition=medium
# ShockFind on one snapshot (directory or aggregated .h5); every argument goes to run_shockfind.py:
#   sbatch --export=NONE run_shockfind.sbatch /path/to/output_00801.h5
#   sbatch --export=NONE run_shockfind.sbatch /path/to/output_00801 --save-octree /path/to/octrees/
# One process: the tree build is serial, the characterisation uses OpenMP (no mpirun -np > 1).
# Memory: ~70-75 GB for ~93M cells. Elsewhere: change the account, partition, modules and venv.

# --export=NONE gives a bare shell (no /usr/bin on PATH, no `module`): set both up first.
export PATH=/usr/local/bin:/usr/bin:/bin${PATH:+:$PATH}
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/mnt/beegfs/appsx/modules/all/Core
module purge
module load GCC/14.3.0 OpenMPI/5.0.8 HDF5/1.14.6 Python/3.13.5
source /mnt/beegfs/projects/hpcc250512a2/mpacicco/venvs/shockfind313/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /mnt/beegfs/projects/hpcc250512a2/mpacicco/codes/ShockFindOct
# --export=ALL: the step inherits --export=NONE otherwise, losing the venv and modules.
srun --export=ALL -n 1 -c "$SLURM_CPUS_PER_TASK" python run_shockfind.py "$@"
