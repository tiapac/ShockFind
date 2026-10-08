#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=hpcc250512a2
#SBATCH --job-name=sf_neighhl
#SBATCH --output=/mnt/beegfs/projects/hpcc250512a2/mpacicco/codes/ShockFindOct/sf_neighhl_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --partition=medium
# Submit with: sbatch --export=NONE run_neigh_hl.sbatch
# Results: shockfind_results/ next to the output_NNNNN directory.

# --export=NONE gives a bare shell: initialise Lmod before loading modules.
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/mnt/beegfs/appsx/modules/all/Core
module purge
module load GCC/14.3.0 OpenMPI/5.0.8 HDF5/1.14.6 Python/3.13.5
source /mnt/beegfs/projects/hpcc250512a2/mpacicco/venvs/shockfind313/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /mnt/beegfs/projects/hpcc250512a2/mpacicco/codes/ShockFindOct
srun -n 1 -c 36 python src/octree_shockfind_pipeline.py /mnt/beegfs/projects/hpcc250512a2/mpacicco/SimulationLocaBubble/LBDefault+neigh_hl/output_00801
