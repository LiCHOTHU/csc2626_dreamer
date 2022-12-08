# Running Baselines on MinAtar
```
python test/mdp.py --env freeway_seg --device cpu --exp_suffix debug

sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_seg --device cuda --exp_suffix debug


export SBATCH_TIMELIMIT=1-00:00:00
for SEED in 1 2 3 4 5
do
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway --device cuda --exp_suffix crash0.01 --seed $SEED
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_seg --device cuda --exp_suffix crash0.01 --seed $SEED
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_rgb --device cuda --exp_suffix crash0.01 --seed $SEED
done

```