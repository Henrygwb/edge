# Data Generation

## Code for generating and processing agent trajectories.

See run.py for info

For reproducibility, a pip environment has been exported into `requirements.txt`. Otherwise, just make sure you have the following packages and it should be fine:
- torch
- torchvision
- gym
- stable_baselines3[extra]
- pyyaml

## Running Atari games

Currently run.py supports running and saving trajectories in Atari games. Usage:

```
$ python src/data/run.py -h
usage: run.py [-h] --log_dir LOG_DIR [--model_dir MODEL_DIR] --game GAME [--episodes EPISODES] [--max_timesteps MAX_TIMESTEPS] [--render_game] [--use_pretrained_model]

optional arguments:
  -h, --help            show this help message and exit
  --log_dir LOG_DIR     Log directory path
  --model_dir MODEL_DIR
                        Model directory path
  --game GAME           Game to run
  --episodes EPISODES   Number of episodes
  --max_timesteps MAX_TIMESTEPS
                        Max timesteps per episode
  --render_game         Render live game during runs
  --use_pretrained_model
                        Use a pretrained model
```

I recommend navigating to this directory and running: `python run.py --log_dir logs/ --game SeaquestNoFrameskip-v4 --render_game --use_pretrained_model --model_dir models --episodes 5` to start.