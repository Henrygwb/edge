import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import numpy as np
from explainer.DGP_XRL import DGPXRL
from explainer.Rudder_XRL import Rudder
from explainer.RnnAttn_XRL import RnnAttn
from explainer.RnnSaliency_XRL import RnnSaliency
from explainer.RationaleNet_XRL import RationaleNet

# Setup env, load the target agent, and collect the trajectories.
env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs_exp/' + env_name
num_traj = 2120
max_ep_len = 200

# Get the shared parameters, prepare training/testing data.
num_class = 2
seq_len = 200
input_dim = 80
n_action = 7
len_diff = max_ep_len - seq_len
exp_idx = np.arange(int(np.load(traj_path + '_num_traj.npy')))

hiddens = [4]
encoder_type = 'CNN'
rnn_cell_type = 'GRU'
n_epoch = 100
batch_size = 40
save_path = 'models/'
likelihood_type = 'classification'
n_stab_samples = 10


# Explainer 1 - Value function.
values = []
n_batch = int(exp_idx.shape[0] / batch_size)

for batch in range(n_batch):
    for idx in exp_idx[batch * batch_size:(batch + 1) * batch_size, ]:
        value_tmp = np.load(traj_path + '_traj_' + str(idx) + '.npz')['values']
        values.append(value_tmp[len_diff:])

values = np.array(values)
sal_value = (values - np.min(values, axis=1)[:, None]) / \
            (np.max(values, axis=1)[:, None] - np.min(values, axis=1)[:, None] + 1e-16)
print(sal_value.shape)
np.savez_compressed(save_path + 'value_exp.npz', sal=sal_value)


# Explainer 2 - Rudder.
rudder_explainer = Rudder(seq_len=seq_len, len_diff=len_diff, input_dim=input_dim, hiddens=hiddens,
                          n_action=n_action, encoder_type=encoder_type)
name = 'rudder_' + encoder_type + '_' + rnn_cell_type
rudder_explainer.load(save_path + name + '_model.data')
rudder_explainer.test(exp_idx, batch_size, traj_path)
sal_rudder_all, fid_all, stab_all, abs_all, mean_time = rudder_explainer.exp_fid_stab(exp_idx, batch_size,
                                                                                      traj_path,
                                                                                      likelihood_type,
                                                                                      n_stab_samples)
print('=============================================')
print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_all[0])))
print('Std fid of the zero-one normalization: {}'.format(np.std(fid_all[0])))
print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(abs_all[0])))
print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(abs_all[0])))

print('=============================================')
print('Mean fid of the top 5 normalization: {}'.format(np.mean(fid_all[1])))
print('Std fid of the top 5 normalization: {}'.format(np.std(fid_all[1])))
print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(abs_all[1])))
print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(abs_all[1])))

print('=============================================')
print('Mean fid of the top 15 normalization: {}'.format(np.mean(fid_all[2])))
print('Std fid of the top 15 normalization: {}'.format(np.std(fid_all[2])))
print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(abs_all[2])))
print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(abs_all[2])))

print('=============================================')
print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_all[3])))
print('Std fid of the top 25 normalization: {}'.format(np.std(fid_all[3])))
print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(abs_all[3])))
print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(abs_all[3])))

print('=============================================')
print('Mean stab: {}'.format(np.mean(stab_all)))
print('Std stab: {}'.format(np.std(stab_all)))

print('=============================================')
print('Mean exp time: {}'.format(mean_time))

np.savez_compressed(save_path + name + '_exp.npz', sal=sal_rudder_all, fid=fid_all, stab=stab_all,
                    abs_diff=abs_all, time=mean_time)


# Explainer 3 - RNN + Saliency.
use_input_attention = True
saliency_explainer = RnnSaliency(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                 encoder_type=encoder_type, num_class=2, rnn_cell_type='LSTM',
                                 use_input_attention=use_input_attention, normalize=False)
name = 'saliency_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + str(use_input_attention)
saliency_explainer.load(save_path + name + '_model.data')
saliency_explainer.test(exp_idx, batch_size, traj_path)

sal_best, fid_best, stab_best, acc_best, abs_diff_best, time_best = saliency_explainer.exp_fid_stab(
    exp_idx, batch_size, traj_path, True, 'gradient', n_samples=15, n_stab_samples=n_stab_samples)

print('=============================================')
print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_best[0])))
print('Std fid of the zero-one normalization: {}'.format(np.std(fid_best[0])))
print('Acc fid of the zero-one normalization: {}'.format(acc_best[0]))
print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(abs_diff_best[0])))
print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(abs_diff_best[0])))

print('=============================================')
print('Mean fid of the top 5 normalization: {}'.format(np.mean(fid_best[1])))
print('Std fid of the top 5 normalization: {}'.format(np.std(fid_best[1])))
print('Acc fid of the top 5 normalization: {}'.format(acc_best[1]))
print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(abs_diff_best[1])))
print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(abs_diff_best[1])))

print('=============================================')
print('Mean fid of the top 15 normalization: {}'.format(np.mean(fid_best[2])))
print('Std fid of the top 15 normalization: {}'.format(np.std(fid_best[2])))
print('Acc fid of the top 15 normalization: {}'.format(acc_best[2]))
print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(abs_diff_best[2])))
print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(abs_diff_best[2])))

print('=============================================')
print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_best[3])))
print('Std fid of the top 25 normalization: {}'.format(np.std(fid_best[3])))
print('Acc fid of the top 25 normalization: {}'.format(acc_best[3]))
print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(abs_diff_best[3])))
print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(abs_diff_best[3])))

print('=============================================')
print('Mean stab: {}'.format(np.mean(stab_best)))
print('Std stab: {}'.format(np.std(stab_best)))

print('=============================================')
print('Mean exp time: {}'.format(time_best))

np.savez_compressed(
    save_path + name + '_exp_best.npz', sal=sal_best, fid=fid_best,
    stab=stab_best, time=time_best, acc=acc_best, abs_diff=abs_diff_best)


# Explainer 4 - Attention.
attention_type = 'tanh'
name = 'attention_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + attention_type

attention_explainer = RnnAttn(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                              encoder_type=encoder_type, num_class=2, attention_type=attention_type,
                              normalize=False)

attention_explainer.load(save_path + name + '_model.data')
attention_explainer.test(exp_idx, batch_size, traj_path)

sal_attention_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = attention_explainer.exp_fid_stab(
    exp_idx, batch_size, traj_path, n_stab_samples)

print('=============================================')
print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_all[0])))
print('Std fid of the zero-one normalization: {}'.format(np.std(fid_all[0])))
print('Acc fid of the zero-one normalization: {}'.format(acc_all[0]))
print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(abs_diff_all[0])))
print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(abs_diff_all[0])))

print('=============================================')
print('Mean fid of the top 5 normalization: {}'.format(np.mean(fid_all[1])))
print('Std fid of the top 5 normalization: {}'.format(np.std(fid_all[1])))
print('Acc fid of the top 5 normalization: {}'.format(acc_all[1]))
print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(abs_diff_all[1])))
print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(abs_diff_all[1])))

print('=============================================')
print('Mean fid of the top 15 normalization: {}'.format(np.mean(fid_all[2])))
print('Std fid of the top 15 normalization: {}'.format(np.std(fid_all[2])))
print('Acc fid of the top 15 normalization: {}'.format(acc_all[2]))
print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(abs_diff_all[2])))
print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(abs_diff_all[2])))

print('=============================================')
print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_all[3])))
print('Std fid of the top 25 normalization: {}'.format(np.std(fid_all[3])))
print('Acc fid of the top 25 normalization: {}'.format(acc_all[3]))
print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(abs_diff_all[3])))
print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(abs_diff_all[3])))

print('=============================================')
print('Mean stab: {}'.format(np.mean(stab_all)))
print('Std stab: {}'.format(np.std(stab_all)))

print('=============================================')
print('Mean exp time: {}'.format(mean_time))

np.savez_compressed(save_path + name + '_exp.npz', sal=sal_attention_all, fid=fid_all, stab=stab_all,
                    time=mean_time,
                    acc=acc_all, abs_diff=abs_diff_all)


# Explainer 5 - Rationale net.

name = 'rationale_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type

rationale_explainer = RationaleNet(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                   encoder_type=encoder_type, num_class=2, normalize=False)

rationale_explainer.load(save_path + name + '_model.data')
rationale_explainer.test(exp_idx, batch_size, traj_path)

sal_rationale_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = rationale_explainer.exp_fid_stab(
    exp_idx, batch_size, traj_path, n_stab_samples)

print('=============================================')
print('Mean fid of the zero-one normalization: {}'.format(np.mean(fid_all[0])))
print('Std fid of the zero-one normalization: {}'.format(np.std(fid_all[0])))
print('Acc fid of the zero-one normalization: {}'.format(acc_all[0]))
print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(abs_diff_all[0])))
print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(abs_diff_all[0])))

print('=============================================')
print('Mean fid of the top 5 normalization: {}'.format(np.mean(fid_all[1])))
print('Std fid of the top 5 normalization: {}'.format(np.std(fid_all[1])))
print('Acc fid of the top 5 normalization: {}'.format(acc_all[1]))
print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(abs_diff_all[1])))
print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(abs_diff_all[1])))

print('=============================================')
print('Mean fid of the top 15 normalization: {}'.format(np.mean(fid_all[2])))
print('Std fid of the top 15 normalization: {}'.format(np.std(fid_all[2])))
print('Acc fid of the top 15 normalization: {}'.format(acc_all[2]))
print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(abs_diff_all[2])))
print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(abs_diff_all[2])))

print('=============================================')
print('Mean fid of the top 25 normalization: {}'.format(np.mean(fid_all[3])))
print('Std fid of the top 25 normalization: {}'.format(np.std(fid_all[3])))
print('Acc fid of the top 25 normalization: {}'.format(acc_all[3]))
print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(abs_diff_all[3])))
print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(abs_diff_all[3])))

print('=============================================')
print('Mean stab: {}'.format(np.mean(stab_all)))
print('Std stab: {}'.format(np.std(stab_all)))

print('=============================================')
print('Mean exp time: {}'.format(mean_time))

np.savez_compressed(save_path + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
                    time=mean_time,
                    acc=acc_all, abs_diff=abs_diff_all)


# Explainer 6 - DGP.

save_path = 'models/dgp/'
likelihood_type = 'classification'
n_stab_samples = 10

# Explainer 6 - DGP.
grid_bound = [(-3, 3)] * hiddens[-1] * 2
likelihood_sample_size = 8

model_1 = 'dgp_classification_GRU_100_False_False_False_False_False_False_model.data'
dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                       hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
                       n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
                       grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
                       mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
                       using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
                       using_OrthogonallyDecouple=False, weight_x=False, lambda_1=0.001)

dgp_explainer.load(save_path+model_1)
dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
name = model_1[:-11]
sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
    exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)

np.savez_compressed(save_path + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
                    time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
                    step_covar=covar_all[2], abs_diff=abs_diff_all)

dgp_1_fid_results = np.load(save_path + 'dgp_classification_GRU_100_False_False_False_False_False_False_exp.npz')
dgp_1_sal = dgp_1_fid_results['sal']
dgp_1_fid = dgp_1_fid_results['fid']
dgp_1_stab = dgp_1_fid_results['stab']
dgp_1_diff = dgp_1_fid_results['abs_diff']
dgp_1_time = dgp_1_fid_results['time']
dgp_1_acc = dgp_1_fid_results['acc']

print('=============================================')
print('Mean fid of the zero-one normalization: {}'.format(np.mean(dgp_1_fid[0])))
print('Std fid of the zero-one normalization: {}'.format(np.std(dgp_1_fid[0])))
print('Acc fid of the zero-one normalization: {}'.format(dgp_1_acc[0]))
print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(dgp_1_diff[0])))
print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(dgp_1_diff[0])))

print('=============================================')
print('Mean fid of the top 5 normalization: {}'.format(np.mean(dgp_1_fid[1])))
print('Std fid of the top 5 normalization: {}'.format(np.std(dgp_1_fid[1])))
print('Acc fid of the top 5 normalization: {}'.format(dgp_1_acc[1]))
print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(dgp_1_diff[1])))
print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(dgp_1_diff[1])))

print('=============================================')
print('Mean fid of the top 15 normalization: {}'.format(np.mean(dgp_1_fid[2])))
print('Std fid of the top 15 normalization: {}'.format(np.std(dgp_1_fid[2])))
print('Acc fid of the top 15 normalization: {}'.format(dgp_1_acc[2]))
print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(dgp_1_diff[2])))
print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(dgp_1_diff[2])))

print('=============================================')
print('Mean fid of the top 25 normalization: {}'.format(np.mean(dgp_1_fid[3])))
print('Std fid of the top 25 normalization: {}'.format(np.std(dgp_1_fid[3])))
print('Acc fid of the top 25 normalization: {}'.format(dgp_1_acc[3]))
print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(dgp_1_diff[3])))
print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(dgp_1_diff[3])))

print('=============================================')
print('Mean stab: {}'.format(np.mean(dgp_1_stab)))
print('Std stab: {}'.format(np.std(dgp_1_stab)))

print('=============================================')
print('Mean exp time: {}'.format(dgp_1_time))



model_2 = 'dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_model.data'
dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                       hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
                       n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
                       grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
                       mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
                       using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
                       using_OrthogonallyDecouple=False, weight_x=False, lambda_1=0.1)


dgp_explainer.load(save_path+'dgp/'+model_2)
dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
name = model_2[:-11]
sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
    exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)

np.savez_compressed(save_path + 'dgp/' + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
                    time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
                    step_covar=covar_all[2], abs_diff=abs_diff_all)

dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
dgp_2_fid = dgp_2_fid_results['fid']
dgp_2_stab = dgp_2_fid_results['stab']
dgp_2_diff = dgp_2_fid_results['abs_diff']
dgp_2_time = dgp_2_fid_results['time']
dgp_2_acc = dgp_2_fid_results['acc']

print('=============================================')
print('Mean fid of the zero-one normalization: {}'.format(np.mean(dgp_2_fid[0])))
print('Std fid of the zero-one normalization: {}'.format(np.std(dgp_2_fid[0])))
print('Acc fid of the zero-one normalization: {}'.format(dgp_2_acc[0]))
print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(dgp_2_diff[0])))
print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(dgp_2_diff[0])))

print('=============================================')
print('Mean fid of the top 5 normalization: {}'.format(np.mean(dgp_2_fid[1])))
print('Std fid of the top 5 normalization: {}'.format(np.std(dgp_2_fid[1])))
print('Acc fid of the top 5 normalization: {}'.format(dgp_2_acc[1]))
print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(dgp_2_diff[1])))
print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(dgp_2_diff[1])))

print('=============================================')
print('Mean fid of the top 15 normalization: {}'.format(np.mean(dgp_2_fid[2])))
print('Std fid of the top 15 normalization: {}'.format(np.std(dgp_2_fid[2])))
print('Acc fid of the top 15 normalization: {}'.format(dgp_2_acc[2]))
print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(dgp_2_diff[2])))
print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(dgp_2_diff[2])))

print('=============================================')
print('Mean fid of the top 25 normalization: {}'.format(np.mean(dgp_2_fid[3])))
print('Std fid of the top 25 normalization: {}'.format(np.std(dgp_2_fid[3])))
print('Acc fid of the top 25 normalization: {}'.format(dgp_2_acc[3]))
print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(dgp_2_diff[3])))
print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(dgp_2_diff[3])))

print('=============================================')
print('Mean stab: {}'.format(np.mean(dgp_2_stab)))
print('Std stab: {}'.format(np.std(dgp_2_stab)))

print('=============================================')
print('Mean exp time: {}'.format(dgp_2_time))


model_3 = 'dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_model.data'
dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                       hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
                       n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
                       grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
                       mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
                       using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
                       using_OrthogonallyDecouple=False, weight_x=True, lambda_1=0.001)


dgp_explainer.load(save_path+model_3)
dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
name = model_3[:-11]
sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
  exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)

np.savez_compressed(save_path + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
                    time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
                    step_covar=covar_all[2], abs_diff=abs_diff_all)

dgp_3_fid_results = np.load(save_path + 'dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_exp.npz')
dgp_3_sal = dgp_3_fid_results['sal']
dgp_3_fid = dgp_3_fid_results['fid']
dgp_3_stab = dgp_3_fid_results['stab']
dgp_3_diff = dgp_3_fid_results['abs_diff']
dgp_3_time = dgp_3_fid_results['time']
dgp_3_acc = dgp_3_fid_results['acc']

print('=============================================')
print('Mean fid of the zero-one normalization: {}'.format(np.mean(dgp_3_fid[0])))
print('Std fid of the zero-one normalization: {}'.format(np.std(dgp_3_fid[0])))
print('Acc fid of the zero-one normalization: {}'.format(dgp_3_acc[0]))
print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(dgp_3_diff[0])))
print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(dgp_3_diff[0])))

print('=============================================')
print('Mean fid of the top 5 normalization: {}'.format(np.mean(dgp_3_fid[1])))
print('Std fid of the top 5 normalization: {}'.format(np.std(dgp_3_fid[1])))
print('Acc fid of the top 5 normalization: {}'.format(dgp_3_acc[1]))
print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(dgp_3_diff[1])))
print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(dgp_3_diff[1])))

print('=============================================')
print('Mean fid of the top 15 normalization: {}'.format(np.mean(dgp_3_fid[2])))
print('Std fid of the top 15 normalization: {}'.format(np.std(dgp_3_fid[2])))
print('Acc fid of the top 15 normalization: {}'.format(dgp_3_acc[2]))
print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(dgp_3_diff[2])))
print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(dgp_3_diff[2])))

print('=============================================')
print('Mean fid of the top 25 normalization: {}'.format(np.mean(dgp_3_fid[3])))
print('Std fid of the top 25 normalization: {}'.format(np.std(dgp_3_fid[3])))
print('Acc fid of the top 25 normalization: {}'.format(dgp_3_acc[3]))
print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(dgp_3_diff[3])))
print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(dgp_3_diff[3])))

print('=============================================')
print('Mean stab: {}'.format(np.mean(dgp_3_stab)))
print('Std stab: {}'.format(np.std(dgp_3_stab)))

print('=============================================')
print('Mean exp time: {}'.format(dgp_3_time))
