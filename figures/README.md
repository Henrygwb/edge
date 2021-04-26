Files for generating figures. By default assumes that `fid_all.npz` and `stab_all.npz` are in this directory. That can be changed with `--metric_dir`. 

To use defaults and generate both figs:

`python gen_figs.py --stab --fid`

That is the same as

`python gen_figs.py --stab --fid --output_dir ./outputs --metric_dir .`
