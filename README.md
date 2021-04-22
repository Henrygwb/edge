# Trajectory-level explanation using Nonparametric Bayesian.

## Code structure.

The proposed explanation model and the four baselines are in `explainer`.
- `DGP_XRL.py`: our proposed rnn-based deep kernel learning model. 
- `RationaleNet_XRL.py`: Rational Net baseline (self-explainable input perturbation model).
- `RnnAttn_XRL.py`: Attention-based RNN with attention weight as the explanations.
- `RnnSaliency_XRL.py`: RNN + saliency methods. 
- `Rudder_XRL.py`: reward distribution as explanation.
Each object has the following functions:
- `train()`: train the approximation model to model the correlation between the input trajectories and the final rewards, it will display the training accuracy.
- `test()`: test the trained model accuracy on the test trajectories.
- `get_explanations`: compute the time step importances for the input trajectories.
- `save`: save the trained model.
- `load`: load a well trained model.

Key parameters (the instruction of most parameters can be found in the inline comments):
- `encoder_type`: 'CNN' or 'MLP', if the observation is environment frame snapshot (image), use 'CNN', it will use CNN to transform the input observation ([n_traj, seq_len, input_channels, input_dim, input_dim], torch.float32) into the observation encoding ([n_traj, seq_len, encode_dim]). It will also use an embedding layer to transform the categorical action ([n_traj, seq_len], torch.long) into the action embedding. Then, it concatenate the observation encoding and action embedding and output the final hidden representation. Note that this cnn structure is designed for Atari games, if currently only support input_dim=80/84 and do not support continous actions. If using a different input dim, change the '4' in  `self.cnn_out_dim = 4 * 4 * 16 + embed_dim` in line 54 of `rnn_utils.py` to the current encoded dim. If using continous actions, change the embeding layer in line 35 of `rnn_utils.py` to an MLP. if the observation and action are feature vectors, use 'MLP', it will concatenate the observations and actions and then run an MLP.    
- `likelhood_type`: 'classification' or 'regression', if final rewards are discrete, using 'classification', otherwise using 'regression'.
- `rnn_cell_type`: 'GRU' or 'LSTM', default as 'GRU' for better efficiency.
- `hiddens`: MLP structure or the RNN hidden dim in the CNN+RNN, suggest using the policy network structure and keep it the same for all the explainers.

Key parameters for each explainer:
- `Rudder_XRL.py`: the commen parameters discussed above (Rudder does not distinguish classification and regression).    
- `RnnSaliency_XRL.py`: set `rnn_cell_type='LSTM'` and `use_input_attention=True` (These two options are only used for this explainer).
- `RnnAttn_XRL.py`: `attention_type ='tanh'` as default.
- `RationaleNet_XRL.py`: the commen parameters discussed above.
- `DGP_XRL.py`: many GP strategy options, use the default choice in `pong.py` or turn off all the options. 

The `atari_pong` contains the explanation pipeline, pretrained agents, and the explanation results (approximation model and time step importance). `pong.py` has the explanation pipline.

## Explanation workflow (refer to `atari_pong`).
- Step 1: make a new folder for the game you are working on (we keep one (type of) game(s) in one folder) with the following subfolders: `agents`, `trajs`, `exp_model_results` or naming them with your own style.
- Step 2: set up the game env, load the pretrained agent, and collect trajectories by running the agent in the environment.
  - Note 1: Run and save the trajectories when collecting them at the first time and load the collected traj for future usages (make sure all the models are trained on the same set of trajectories).
  - Note 2: each trajectory means each game round (The agent fails/loses/dies and the game env restarts). Do not directly splitting the trajectories based on the `done` flag given by the game env. In some games, the agent may have multiple lives or the game runs multiple rounds before returning a `done`. Do a double check and set up the specific spltting flag for such games. 
  - Note 3: Save the original observations and the preprocessed ones used as policy network inputs (For better visualization purpose).
  - Note 4: the trajectories have varied lengthes, pad them into the same length: pad at the front, not the end; pad with a meaningless number (Be careful with 0, '-1' and '1', it will cause confusion for rewards and categorial actions).
  - Note 5: control the traj length with some parameter like `max_ep_len` and discard the trajs that do not finish at the maximum length.
  - Note 6: save every traj with a `.npz` file to prevent the out of memory issue: 
  - Note 7: the shape of the save items: 
    - Observations: [max_seq_len, input_channel, input_dim, input_dim] or [max_seq_len, input_dim].
    - States (preprocessed observations): [max_seq_len, input_channel, input_dim, input_dim] or [max_seq_len, input_dim].
    - Actions: [max_seq_len] or [max_seq_len, act_dim].
    - Rewards: [max_seq_len].
    - Value function outputs: [max_seq_len].
    - Final reward: [1].
    - max_ep_length: actual maximum traj length.
    - traj_count: total number of collected trajs.
- Step 3: load and preprocess the trajectories. 
  - Note 1: change the padded values in obs with `0` and preprocess them into states using the policy network preprocessing method.
  - Note 2: categorical actions: add the action values by `1` and change the padded value to `0`, and record the total number of actions as all the possible actions + 1 (`np.unique(acts)`).
  - Note 3: record the actual length of each trajectories.
  - Note 4: obtain the final rewards. Discrete final rewards: change the final rewards to class labels if it has negative values (e.g., [-1, 0, 1] -> [0, 1, 2]), and record the number of classes.
  - Note 5: Prepare the training and testing set. Be careful with the data type: torch.float32 for continuous variables, torch.long for integers (discrete variables).
- Step 4: record the value function outputs as the first baseline results.
- Step 5: run the baselines mentioned above with a different choice of `--explainer`: 'rudder', 'saliency', 'attention', 'rationale', save the trained model and obtained explanations, and training/testing accuracy and runtimes. 
- Step 6: run our method and save/record the same things.
- Step 7: Quantitative evaluation.
  - Approximation accuracy (precision, recall, f1).
  - Fidelity w.r.t. the explanation model: (In total four metrics (four methods of perturbing the input traj), each one computes three/one values in classification/regression. For classification, we compute the fidelity value of each traj, the prediction diff of each traj before/after perturbation, and the classification accuracy of the perturbed trajs. For regression, we compute the abs prediction diff of each traj before/after perturbation.
  - Fidelity w.r.t. the RL task. 
  - Sensitivity/stablity: two parameters number of samples used for compute stability (default as 10), eps (perturbation strength) adjust it to meantain the noise value range as about (5%) of the input obs value range where the noise value range is (0, eps). 
  - Efficiency/runtime: training and explanation runtime.
  
