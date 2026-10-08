#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=hpcc250512a2
#SBATCH --job-name=sf_xthin
#SBATCH --output=/mnt/beegfs/projects/hpcc250512a2/mpacicco/ShockFindOct/sf_xthin_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --partition=short

cd /mnt/beegfs/projects/hpcc250512a2/mpacicco/ShockFindOct
source /mnt/beegfs/projects/hpcc250512a2/mpacicco/venvs/shockfind/bin/activate
srun -n 1 -c 36 python src/octree_shockfind_pipeline.py /mnt/beegfs/projects/hpcc250512a2/mpacicco/SimulationLocaBubble/LBThinDiskExtra/outputs/output_00019
