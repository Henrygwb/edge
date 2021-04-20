import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3"
import torch
import numpy as np
import gym, argparse
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


# Setup env, load the target agent, and collect the trajectories.
env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs/' + env_name
# traj_path = None
num_traj = 30
max_ep_len = 200

if traj_path is None:
    # Load agent, build environment, and play an episode.
    env = gym.make(env_name)
    env.seed(1)

    model = NNPolicy(channels=1, num_actions=env.action_space.n)
    _ = model.try_load(agent_path, checkpoint='*.tar')
    torch.manual_seed(1)

    rollout(model, env, num_traj=num_traj, max_ep_len=max_ep_len, save_path='trajs/'+env_name,render=False)


# Get the shared parameters, prepare training/testing data.
num_class = 2
seq_len = int(np.load('trajs/' + env_name + '_max_length.npy'))
input_dim = 80
n_action = 7
len_diff = max_ep_len - seq_len
total_data_idx = np.arange(int(np.load('trajs/' + env_name + '_num_traj.npy')))
train_idx = total_data_idx[0:int(total_data_idx.shape[0]*0.7), ]
test_idx = total_data_idx[int(total_data_idx.shape[0]*0.7):, ]
exp_idx = total_data_idx[0:int(total_data_idx.shape[0]*0.6), ]

hiddens = [4]
encoder_type = 'CNN'
rnn_cell_type = 'GRU'
n_epoch = 2
batch_size = 4
save_path = 'exp_model_results/'
likelihood_type = 'classification'

if args.explainer == 'value':
    # Explainer 1 - Value function.
    values = []
    n_batch = int(exp_idx.shape[0]/batch_size)+1
    for batch in range(n_batch):
        for idx in exp_idx[batch*batch_size:min((batch+1)*batch_size, exp_idx.shape[0]), ]:
            value_tmp = np.load(traj_path + '_traj_' + str(idx) + '.npz')['values']
            values.append(value_tmp[len_diff:])

    values = np.array(values)
    sal_value = (values - np.min(values, axis=1)[:, None]) / \
                (np.max(values, axis=1)[:, None] - np.min(values, axis=1)[:, None])
    np.savez_compressed(save_path+'value_exp.npz', sal=sal_value)

elif args.explainer == 'rudder':
    # Explainer 2 - Rudder.
    rudder_explainer = Rudder(seq_len=seq_len, len_diff=len_diff, input_dim=input_dim, hiddens=hiddens,
                              n_action=n_action, encoder_type=encoder_type)
    name = 'rudder_' + encoder_type + '_' + rnn_cell_type
    rudder_explainer.train(train_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')
    rudder_explainer.test(test_idx, batch_size, traj_path)
    rudder_explainer.load(save_path+name+'_model.data')
    rudder_explainer.test(test_idx, batch_size, traj_path)
    sal_rudder_all, fid_all, stab_all, mean_time = rudder_explainer.exp_fid_stab(exp_idx, batch_size, traj_path,
                                                                                 likelihood_type)
    print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_all[0])))
    print('Std fid of the zero-one normalization: {}'.format(np.std(fid_all[0])))

    print('Mean fid of the top 10 normalization: {}'.format(np.mean(fid_all[1])))
    print('Std fid of the top 10 normalization: {}'.format(np.std(fid_all[1])))

    print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_all[2])))
    print('Std fid of the top 25 normalization: {}'.format(np.std(fid_all[2])))

    print('Mean fid of the top 50 normalization: {}'.format(np.mean(fid_all[3])))
    print('Std fid of the top 50 normalization: {}'.format(np.std(fid_all[4])))

    print('Mean stab: {}'.format(np.mean(stab_all)))
    print('Std stab: {}'.format(np.std(stab_all)))

    print('Mean exp time: {}'.format(mean_time))

    np.savez_compressed(save_path+name+'_exp.npz', sal=sal_rudder_all, fid=fid_all, stab=stab_all, time=mean_time)

elif args.explainer == 'saliency':
    # Explainer 3 - RNN + Saliency.
    use_input_attention = True
    saliency_explainer = RnnSaliency(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                     encoder_type=encoder_type, num_class=2, rnn_cell_type='LSTM',
                                     use_input_attention=use_input_attention, normalize=False)
    name = 'saliency_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + str(use_input_attention)

    saliency_explainer.train(train_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')
    saliency_explainer.test(test_idx, batch_size, traj_path)
    saliency_explainer.load(save_path+name+'_model.data')
    saliency_explainer.test(test_idx, batch_size, traj_path)
    all_methods = ['gradient', 'integrated_gradient', 'unifintgrad', 'smoothgrad', 'expgrad', 'vargrad']
    fid_all_methods = []
    for back2rnn in [True, False]:
        for saliency_methond in all_methods:
            sal_saliency_all, fid_all, stab_all, acc_all, mean_time = saliency_explainer.exp_fid_stab(
                exp_idx, batch_size, traj_path, back2rnn, saliency_methond, n_samples=2, n_stab_samples=1)
            fid_all_methods.append(np.mean(fid_all))
            if back2rnn:
                np.savez_compressed(save_path + name + '_' + saliency_methond + '_exp_rnn_layer.npz',
                                    sal=sal_saliency_all, fid=fid_all, stab=stab_all, time=mean_time, acc=acc_all)
            else:
                np.savez_compressed(save_path + name + '_' + saliency_methond + '_exp_input_layer.npz',
                                    sal=sal_saliency_all, fid=fid_all, stab=stab_all, time=mean_time, acc=acc_all)
    best_method_idx = np.argmin(fid_all_methods)
    if best_method_idx < 6:
        saliency_methond = all_methods[best_method_idx]
        sal_best = np.load(save_path + name + '_' + saliency_methond + '_exp_rnn_layer.npz')['sal']
        fid_best = np.load(save_path + name + '_' + saliency_methond + '_exp_rnn_layer.npz')['fid']
        stab_best = np.load(save_path + name + '_' + saliency_methond + '_exp_rnn_layer.npz')['stab']
        time_best = np.load(save_path + name + '_' + saliency_methond + '_exp_rnn_layer.npz')['time']
        acc_best = np.load(save_path + name + '_' + saliency_methond + '_exp_rnn_layer.npz')['acc']
    else:
        saliency_methond = all_methods[best_method_idx - 6]
        sal_best = np.load(save_path + name + '_' + saliency_methond + '_exp_input_layer.npz')['sal']
        fid_best = np.load(save_path + name + '_' + saliency_methond + '_exp_input_layer.npz')['fid']
        stab_best = np.load(save_path + name + '_' + saliency_methond + '_exp_input_layer.npz')['stab']
        time_best = np.load(save_path + name + '_' + saliency_methond + '_exp_input_layer.npz')['time']
        acc_best = np.load(save_path + name + '_' + saliency_methond + '_exp_input_layer.npz')['acc']

    print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_best[0])))
    print('Std fid of the zero-one normalization: {}'.format(np.std(fid_best[0])))
    print('Acc fid of the zero-one normalization: {}'.format(acc_best[0]))

    print('Mean fid of the top 10 normalization: {}'.format(np.mean(fid_best[1])))
    print('Std fid of the top 10 normalization: {}'.format(np.std(fid_best[1])))
    print('Acc fid of the top 10 normalization: {}'.format(acc_best[1]))

    print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_best[2])))
    print('Std fid of the top 25 normalization: {}'.format(np.std(fid_best[2])))
    print('Acc fid of the top 25 normalization: {}'.format(acc_best[2]))

    print('Mean fid of the top 50 normalization: {}'.format(np.mean(fid_best[3])))
    print('Std fid of the top 50 normalization: {}'.format(np.std(fid_best[3])))
    print('Acc fid of the top 50 normalization: {}'.format(acc_best[3]))

    print('Mean stab: {}'.format(np.mean(stab_best)))
    print('Std stab: {}'.format(np.std(stab_best)))

    print('Mean exp time: {}'.format(time_best))

    np.savez_compressed(save_path + name + '_exp_best.npz',
                        sal=sal_best, fid=fid_best, stab=stab_best, time=time_best, acc=acc_best)

elif args.explainer == 'attention':
    # Explainer 4 - AttnRNN.
    attention_type = 'tanh'
    name = 'attention_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + attention_type

    attention_explainer = RnnAttn(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                  encoder_type=encoder_type, num_class=2, attention_type=attention_type,
                                  normalize=False)

    attention_explainer.train(train_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')
    attention_explainer.test(test_idx, batch_size, traj_path)
    attention_explainer.load(save_path+name+'_model.data')
    attention_explainer.test(test_idx, batch_size, traj_path)

    sal_attention_all, fid_all, stab_all, acc_all, mean_time = attention_explainer.exp_fid_stab(exp_idx,
                                                                                                batch_size, traj_path)
    print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_all[0])))
    print('Std fid of the zero-one normalization: {}'.format(np.std(fid_all[0])))
    print('Acc fid of the zero-one normalization: {}'.format(acc_all[0]))

    print('Mean fid of the top 10 normalization: {}'.format(np.mean(fid_all[1])))
    print('Std fid of the top 10 normalization: {}'.format(np.std(fid_all[1])))
    print('Acc fid of the top 10 normalization: {}'.format(acc_all[1]))

    print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_all[2])))
    print('Std fid of the top 25 normalization: {}'.format(np.std(fid_all[2])))
    print('Acc fid of the top 25 normalization: {}'.format(acc_all[2]))

    print('Mean fid of the top 50 normalization: {}'.format(np.mean(fid_all[3])))
    print('Std fid of the top 50 normalization: {}'.format(np.std(fid_all[3])))
    print('Acc fid of the top 50 normalization: {}'.format(acc_all[3]))

    print('Mean stab: {}'.format(np.mean(stab_all)))
    print('Std stab: {}'.format(np.std(stab_all)))

    print('Mean exp time: {}'.format(mean_time))

    np.savez_compressed(save_path+name+'_exp.npz', sal=sal_attention_all, fid=fid_all, stab=stab_all, time=mean_time,
                        acc=acc_all)

elif args.explainer == 'rationale':
    # Explainer 5 - RationaleNet.
    name = 'rationale_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type

    rationale_explainer = RationaleNet(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                       encoder_type=encoder_type, num_class=2, normalize=False)

    rationale_explainer.train(train_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')
    rationale_explainer.test(test_idx, batch_size, traj_path)
    rationale_explainer.load(save_path+name+'_model.data')
    rationale_explainer.test(test_idx, batch_size, traj_path)

    sal_rationale_all, fid_all, stab_all, acc_all, mean_time = rationale_explainer.exp_fid_stab(exp_idx,
                                                                                                batch_size, traj_path)
    print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_all[0])))
    print('Std fid of the zero-one normalization: {}'.format(np.std(fid_all[0])))
    print('Acc fid of the zero-one normalization: {}'.format(acc_all[0]))

    print('Mean fid of the top 10 normalization: {}'.format(np.mean(fid_all[1])))
    print('Std fid of the top 10 normalization: {}'.format(np.std(fid_all[1])))
    print('Acc fid of the top 10 normalization: {}'.format(acc_all[1]))

    print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_all[2])))
    print('Std fid of the top 25 normalization: {}'.format(np.std(fid_all[2])))
    print('Acc fid of the top 25 normalization: {}'.format(acc_all[2]))

    print('Mean fid of the top 50 normalization: {}'.format(np.mean(fid_all[3])))
    print('Std fid of the top 50 normalization: {}'.format(np.std(fid_all[3])))
    print('Acc fid of the top 50 normalization: {}'.format(acc_all[3]))

    print('Mean stab: {}'.format(np.mean(stab_all)))
    print('Std stab: {}'.format(np.std(stab_all)))

    print('Mean exp time: {}'.format(mean_time))

    np.savez_compressed(save_path+name+'_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all, time=mean_time,
                        acc=acc_all)

elif args.explainer == 'dgp':
    # Explainer 6 - DGP.
    rnn_cell_type = 'GRU'
    optimizer = 'adam'
    num_inducing_points = 20
    using_ngd = True # Whether to use natural gradient descent.
    using_ksi = False # Whether to use KSI approximation, using this with other options as False.
    using_ciq = True # Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD.
    using_sor = False # Whether to use SoR approximation, not applicable for KSI and CIQ.
    using_OrthogonallyDecouple = False # Using together NGD may cause numerical issue.
    grid_bound = [(-3, 3)] * hiddens[-1] * 2

    dgp_explainer = DGPXRL(train_len=train_idx.shape[0], seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                           hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type=optimizer,
                           n_epoch=n_epoch, gamma=0.1, num_inducing_points=num_inducing_points, n_action=n_action,
                           grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
                           mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
                           using_ngd=using_ngd, using_ksi=using_ksi, using_ciq=using_ciq, using_sor=using_sor,
                           using_OrthogonallyDecouple=using_OrthogonallyDecouple)

    name = 'dgp_' + likelihood_type + '_' + rnn_cell_type + '_' + \
           str(num_inducing_points)+'_'+ str(using_ngd) + '_' + str(using_ngd) + '_' \
           + str(using_ksi) + '_' + str(using_ciq) + '_' + str(using_sor) + '_' \
           + str(using_OrthogonallyDecouple)

    dgp_explainer.train(train_idx, batch_size, traj_path, save_path=save_path+name+'_model.data')
    dgp_explainer.test(test_idx, batch_size, traj_path)
    dgp_explainer.load(save_path+name+'_model.data')
    dgp_explainer.test(test_idx, batch_size, traj_path)

    sal_rationale_all, covar_all, fid_all, stab_all, acc_all, mean_time = dgp_explainer.exp_fid_stab(exp_idx,
                                                                                                     batch_size,
                                                                                                     traj_path)
    print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_all[0])))
    print('Std fid of the zero-one normalization: {}'.format(np.std(fid_all[0])))
    print('Acc fid of the zero-one normalization: {}'.format(acc_all[0]))

    print('Mean fid of the top 10 normalization: {}'.format(np.mean(fid_all[1])))
    print('Std fid of the top 10 normalization: {}'.format(np.std(fid_all[1])))
    print('Acc fid of the top 10 normalization: {}'.format(acc_all[1]))

    print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_all[2])))
    print('Std fid of the top 25 normalization: {}'.format(np.std(fid_all[2])))
    print('Acc fid of the top 25 normalization: {}'.format(acc_all[2]))

    print('Mean fid of the top 50 normalization: {}'.format(np.mean(fid_all[3])))
    print('Std fid of the top 50 normalization: {}'.format(np.std(fid_all[3])))
    print('Acc fid of the top 50 normalization: {}'.format(acc_all[3]))

    print('Mean stab: {}'.format(np.mean(stab_all)))
    print('Std stab: {}'.format(np.std(stab_all)))

    print('Mean exp time: {}'.format(mean_time))

    np.savez_compressed(save_path+name+'_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all, time=mean_time,
                        acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1], step_covar=covar_all[2])

    # VisualizeCovar(covariance[0], save_path+name+'_dgp_full_covar.pdf')
    # VisualizeCovar(covariance[1], save_path+name+'_dgp_traj_covar.pdf')
    # VisualizeCovar(covariance[2], save_path+name+'_dgp_step_covar.pdf')


