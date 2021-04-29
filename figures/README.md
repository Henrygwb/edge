Files for generating figures. By default assumes that `fid_all.npz` and `stab_all.npz` are in this directory. That can be changed with `--metric_dir`. 

To use defaults and generate both figs:

`python gen_figs.py --stab --fid`

That is the same as

`python gen_figs.py --stab --fid --output_dir ./outputs --metric_dir .`

-------

To plot trajectories:

`python visualize_trajectory.py --data ./Pong-v0_traj_0.npz --out trajectories.png --from_idx 150 --to_idx 200`

or

`python visualize_trajectory.py --data ./Pong-v0_traj_0.npz --out trajectories.png` to plot all images
