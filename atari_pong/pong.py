import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import numpy as np
import gym, argparse
from scipy.misc import imresize
from explainer.DGP_XRL import DGPXRL
from explainer.Rudder_XRL import Rudder
from explainer.RnnAttn_XRL import RnnAttn
from explainer.gp_utils import VisualizeCovar
from atari_pong.utils import NNPolicy, rollout
from explainer.RnnSaliency_XRL import RnnSaliency
from explainer.RationaleNet_XRL import RationaleNet



parser = argparse.ArgumentParser()
parser.add_argument("--explainer", type=str, default='value')

args = parser.parse_args()


def obs_resize(obs):
    obs = obs[:, :, 35:195].mean(-1)
    obs_new = np.zeros((obs.shape[0], obs.shape[1], 80, 80))
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            obs_new[i, j] = imresize(obs[i, j], (80, 80)).astype(np.float32)/255.
    del obs
    return obs_new


# Setup env, load the target agent, and collect the trajectories.
env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs/Pong-v0.npz'
#traj_path = None
num_traj = 15

if traj_path is None:
    # Load agent, build environment, and play an episode.
    env = gym.make(env_name)
    env.seed(1)

    model = NNPolicy(channels=1, num_actions=env.action_space.n)
    _ = model.try_load(agent_path, checkpoint='*.tar')
    torch.manual_seed(1)

    history = rollout(model, env, num_traj=num_traj, max_ep_len=200, render=False)

    np.savez_compressed('trajs/'+env_name+'.npz', observations=history['observations'], actions=history['actions']
                        , rewards=history['rewards'], values=history['values'])
else:
    history = np.load(traj_path)

obs = history['observations']
acts = history['actions']
rewards = history['rewards']
values = history['values']

del history

# Preprocess the trajs: change the padding -1 back to 0, resize obs to state, extract the final reward.
acts[acts==-20] = -1
acts = acts + 1
traj_lens = np.count_nonzero(acts, axis=1)
obs[obs==-20] = 0
obs = obs_resize(obs)
rewards[rewards==-20] = 0
final_rewards = rewards[:, -1].astype('int32') # get the final reward of each traj.
final_rewards[final_rewards==-1] = 0
final_rewards[final_rewards==1] = 1

# Get the shared parameters, prepare training/testing data.
num_class = len(np.unique(final_rewards))
seq_len = values.shape[1]
input_dim = obs.shape[3]
n_action = np.max(acts) + 1
train_size = int(obs.shape[0]*0.8)

obs = torch.tensor(obs[..., None, :, :], dtype=torch.float32)
acts = torch.tensor(acts, dtype=torch.long)
rewards = torch.tensor(rewards, dtype=torch.float32)
final_rewards = torch.tensor(final_rewards, dtype=torch.long)

train_set = torch.utils.data.TensorDataset(obs[0:train_size, ...], acts[0:train_size, ...], final_rewards[0:train_size, ...])
test_set = torch.utils.data.TensorDataset(obs[train_size:, ...], acts[train_size:, ...], final_rewards[train_size:, ...])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False)

hiddens = [4]
encoder_type = 'CNN'
rnn_cell_type = 'GRU'
n_epoch = 2
save_path = 'exp_model_results/'

if args.explainer == 'value':
    # Explainer 1 - Value function.
    sal_value = (values - np.min(values, axis=1)[:, None]) / \
                (np.max(values, axis=1)[:, None] - np.min(values, axis=1)[:, None])
    np.savez_compressed(save_path+'value_exp.npz', sal=sal_value)

elif args.explainer == 'rudder':
    # Explainer 2 - Rudder.
    rudder_explainer = Rudder(seq_len=seq_len, input_dim=input_dim, hiddens=hiddens, n_action=n_action,
                              encoder_type=encoder_type)
    name = 'rudder_' + encoder_type + '_' + rnn_cell_type
    rudder_explainer.train(train_loader, n_epoch, save_path=save_path+name+'_model.data')
    rudder_explainer.test(test_loader)
    rudder_explainer.load(save_path+name+'_model.data')
    rudder_explainer.test(test_loader)
    sal_rudder = rudder_explainer.get_explanations(obs, acts, final_rewards)
    np.savez_compressed(save_path+name+'_exp.npz', sal=sal_rudder)

elif args.explainer == 'saliency':
    # Explainer 3 - RNN + Saliency.
    likelihood_type = 'classification'
    use_input_attention = True
    saliency_explainer = RnnSaliency(seq_len, input_dim, likelihood_type, hiddens, n_action, encoder_type=encoder_type,
                                     num_class=2, rnn_cell_type=rnn_cell_type, use_input_attention=use_input_attention,
                                     normalize=False)
    name = 'saliency_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + str(use_input_attention)

    saliency_explainer.train(train_loader, n_epoch, save_path=save_path+name+'_model.data')
    saliency_explainer.test(test_loader)
    saliency_explainer.load(save_path+name+'_model.data')
    saliency_explainer.test(test_loader)
    for back2rnn in [False, True]:
        sal_g = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='gradient',
                                                    back2rnn=back2rnn)
        sal_ig = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='integrated_gradient',
                                                     back2rnn=back2rnn)
        sal_ig_unified = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='unifintgrad',
                                                             back2rnn=back2rnn)
        sal_sg = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='smoothgrad',
                                                     back2rnn=back2rnn)
        sal_sg_exp = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='expgrad',
                                                         back2rnn=back2rnn)
        sal_sg_var = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='vargrad',
                                                         back2rnn=back2rnn)
        if back2rnn:
            np.savez_compressed(save_path+name+'_exp_rnn_layer.npz', sal_g=sal_g, sal_ig=sal_ig,
                                sal_ig_unified=sal_ig_unified, sal_sg=sal_sg, sal_sg_exp=sal_sg_exp,
                                sal_sg_var=sal_sg_var)
        else:
            np.savez_compressed(save_path+name+'_exp_input_layer.npz', sal_g=sal_g, sal_ig=sal_ig,
                                sal_ig_unified=sal_ig_unified, sal_sg=sal_sg, sal_sg_exp=sal_sg_exp,
                                sal_sg_var=sal_sg_var)

elif args.explainer == 'attention':
    # Explainer 4 - AttnRNN.
    likelihood_type = 'classification'
    attention_type = 'tanh'
    name = 'attention_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + attention_type

    attention_explainer = RnnAttn(seq_len, input_dim, likelihood_type, hiddens, n_action, encoder_type=encoder_type,
                                  num_class=2, attention_type=attention_type, normalize=False)

    attention_explainer.train(train_loader, n_epoch, save_path=save_path+name+'_model.data')
    attention_explainer.test(test_loader)
    attention_explainer.load(save_path+name+'_model.data')
    attention_explainer.test(test_loader)
    sal_attention = attention_explainer.get_explanations(obs, acts, final_rewards)
    np.savez_compressed(save_path+name+'_exp.npz', sal=sal_attention)

elif args.explainer == 'rationale':
    # Explainer 5 - RationaleNet.
    likelihood_type = 'classification'
    name = 'rationale_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type

    rationale_explainer = RationaleNet(seq_len, input_dim, likelihood_type, hiddens, n_action, encoder_type=encoder_type,
                                       num_class=2, normalize=False)
    rationale_explainer.train(train_loader, n_epoch, save_path=save_path+name+'_model.data')
    rationale_explainer.test(test_loader)
    rationale_explainer.load(save_path+name+'_model.data')
    rationale_explainer.test(test_loader)
    sal_rationale = rationale_explainer.get_explanations(obs, acts, final_rewards)
    np.savez_compressed(save_path+name+'_exp.npz', sal=sal_rationale)

elif args.explainer == 'dgp':
    # Explainer 6 - DGP.
    likelihood_type = 'regression'
    rnn_cell_type = 'GRU'
    optimizer = 'adam'
    num_inducing_points = 20
    using_ngd = True # Whether to use natural gradient descent.
    using_ksi = False # Whether to use KSI approximation, using this with other options as False.
    using_ciq = True # Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD.
    using_sor = False # Whether to use SoR approximation, not applicable for KSI and CIQ.
    using_OrthogonallyDecouple = False # Using together NGD may cause numerical issue.
    grid_bound = [(-3, 3)] * hiddens[-1] * 2

    dgp_explainer = DGPXRL(train_loader=train_loader, seq_len=seq_len, input_dim=input_dim, hiddens=hiddens,
                           likelihood_type=likelihood_type, lr=0.01, optimizer_type=optimizer, n_epoch=n_epoch,
                           gamma=0.1, num_inducing_points=num_inducing_points, n_action=n_action, grid_bounds=grid_bound,
                           encoder_type=encoder_type, inducing_points=None, mean_inducing_points=None,
                           num_class=num_class, rnn_cell_type=rnn_cell_type, using_ngd=using_ngd, using_ksi=using_ksi,
                           using_ciq=using_ciq, using_sor=using_sor, using_OrthogonallyDecouple=using_OrthogonallyDecouple)
    name = 'dgp_' + likelihood_type + '_' + rnn_cell_type + '_' + \
           str(num_inducing_points)+'_'+ str(using_ngd) + '_' + str(using_ngd) + '_' \
           + str(using_ksi) + '_' + str(using_ciq) + '_' + str(using_sor) + '_' \
           + str(using_OrthogonallyDecouple)
    dgp_explainer.train(save_path+name+'_model.data')
    dgp_explainer.test(test_loader)
    dgp_explainer.load(save_path+name+'_model.data')
    dgp_explainer.test(test_loader)
    sal_rationale, covariance = dgp_explainer.get_explanations(obs, acts, final_rewards)
    # np.savez_compressed(save_path+name+'_exp.npz', sal=sal_rationale, full_covar=covariance[0], traj_cova=covariance[1],
    #                     step_covar=covariance[2])

    # VisualizeCovar(covariance[0], save_path+name+'_dgp_full_covar.pdf')
    # VisualizeCovar(covariance[1], save_path+name+'_dgp_traj_covar.pdf')
    # VisualizeCovar(covariance[2], save_path+name+'_dgp_step_covar.pdf')

