#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export DESCRIPTION="$1"
export JOB_NAME=optimize_spiral_net_"$DESCRIPTION"_"$DATE"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/spiral/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/spiral/$JOB_NAME.%j.e
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK2/Summer2024

ibrun -n 1 python --description=$DESCRIPTION --num_trials=1 --export_dir=$SCRATCH/data/spiral \
  --export --make_db
EOT
