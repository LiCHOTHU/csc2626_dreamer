# Running Baselines on Freeway
```
export SBATCH_TIMELIMIT=1-00:00:00
for SEED in 1 2 3 4 5
do
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env asterix --device cuda --exp_suffix baseline --seed $SEED
done

export SBATCH_TIMELIMIT=1-00:00:00
for SEED in 1 2 3 4 5
do
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env asterix_rgb --device cuda --exp_suffix baseline --seed $SEED
done

export SBATCH_TIMELIMIT=1-00:00:00
for SEED in 1 2 3 4 5
do
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env asterix_seg --device cuda --exp_suffix baseline --seed $SEED
done
```


# Running Baselines and SlotAtten on Freeway
```
# Baselines
python test/mdp.py --env freeway_seg --device cpu --exp_suffix debug

sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_seg --device cuda --exp_suffix debug

export SBATCH_TIMELIMIT=1-00:00:00
for SEED in 1 2 3 4 5
do
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway --device cuda --exp_suffix crash0.01 --seed $SEED
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_seg --device cuda --exp_suffix crash0.01 --seed $SEED
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_rgb --device cuda --exp_suffix crash0.01 --seed $SEED
done

# With SlotAtten
python mdp.py --env freeway_rgb --device cuda --slot --exp_suffix debug

export SBATCH_TIMELIMIT=1-12:00:00
for SEED in 1 2 3 4 5
do
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_rgb --device cuda --slot --exp_suffix slot_crash0.01 --seed $SEED
done

# With 1 slot
export SBATCH_TIMELIMIT=1-12:00:00
for SEED in 1 2 3 4 5
do
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_rgb --device cuda --slot_1slot --exp_suffix slot1_crash0.01 --seed $SEED
done

# With 1 slot but smaller slot size
export SBATCH_TIMELIMIT=1-12:00:00
for SEED in 1 2 3 4 5
do
    sbatch /home/guqiao/src/csc2626_dreamer/scripts/run_cedar.bash python test/mdp.py --env freeway_rgb --device cuda --slot_1slot --exp_suffix slot1_small_crash0.01 --seed $SEED
done


python test/mdp.py --env freeway_rgb --device cuda --slot_1slot --exp_suffix debug

```