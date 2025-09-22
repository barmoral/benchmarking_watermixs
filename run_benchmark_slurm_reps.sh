#!/bin/bash
#SBATCH -J bnchmrk-tip3p_2.2.1
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

# ---- Config you usually tweak ----
export FF_V='2.2.1'         # force field version
export WATER_MODEL='tip3p'   # water model folder prefix
REPLICATES=(1 2 3)          # run these replicates sequentially
BASE_PORT_START=6100        # base port for the first replicate block
PORT_BLOCK_STRIDE=10        # leave gaps to avoid accidental clashes
# ----------------------------------

for r in "${REPLICATES[@]}"; do
  echo "=== Launching replicate r=${r} ==="

  # Give each replicate its own port block: 8200, 8210, 8220, ...
  base_port=$((BASE_PORT_START + (r-1)*PORT_BLOCK_STRIDE))

  for i in 0 1 2; do
    (
      # NOTE: This job requests only 1 GPU. If your node has just one GPU,
      # keep this as 0 so all workers share it. If you *do* have 3 GPUs,
      # switch to: export CUDA_VISIBLE_DEVICES="$i"
      export CUDA_VISIBLE_DEVICES=0

      python benchmark-ext-sage-slurm_v4.py \
        -i training-properties-with-water.json \
        -s stored_data \
        -ff "openff-${FF_V}.offxml" \
        -wff "${WATER_MODEL}_${FF_V}/${WATER_MODEL}.offxml" \
        -o output \
        -r "$r" \
        -p $((base_port + i)) \
        -of request-options.json \
        --worker-id "$i" \
        --num-workers 3
    ) &
  done

  # Wait for the 3 workers of this replicate before starting the next one
  wait
  echo "=== Completed replicate r=${r} ==="
done

echo "All replicates finished."


echo "50 uncorrelated samples, output freq of prod 1000, max prod iterations 4"
sacct --format=jobid,jobname,cputime,elapsed

