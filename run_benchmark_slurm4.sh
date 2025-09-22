#!/bin/bash
#SBATCH -J bnchmrk-tip3p2.2.1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --partition=blanca-shirts
#SBATCH --qos=blanca-shirts
#SBATCH --account=blanca-shirts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH -t 0-72:00:00
#SBATCH --output slurm-%x.%A-%a.out
#SBATCH -e slurm-%x.%A-%a.err

ml cuda/11.8
ml anaconda
source ~/.bashrc
conda activate evaluator-050-118

export OE_LICENSE=/home/bamo6610/Documents/Licenses/oe_license.txt
echo "OE_LICENSE is set to: $OE_LICENSE"
echo "Checking CUDA devices..."
nvidia-smi

export FF_V='2.2.1' # <-- Check here and below!
export WATER_MODEL='tip3p' # <-- Check ff file is in folder and you changed job name


for i in 0 1 2; do
  (
    export CUDA_VISIBLE_DEVICES=0
    python benchmark-ext-sage-slurm_v4.py \
      -i training-properties-with-water.json \
      -s stored_data \
      -ff openff-${FF_V}.offxml \
      -wff ${WATER_MODEL}_${FF_V}/${WATER_MODEL}.offxml \
      -o output \
      -r 2 \
      -p $((7100 + i)) \
      -of request-options.json \
      --worker-id $i \
      --num-workers 3
  ) &
done
wait

echo "50 uncorrelated samples, output freq of prod 1000, max prod iterations 4"
sacct --format=jobid,jobname,cputime,elapsed

