import torch
import timeit
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def exp_fid2nn_zero_one(obs, acts, rewards, explainer, saliency, preds_orin):
    obs.requires_grad = False
    acts.requires_grad = False

    if type(saliency) == np.ndarray:
        saliency = torch.tensor(saliency, dtype=torch.float32)

    if len(obs.shape) == 5:
        saliency = saliency[:, :, None, None, None]
    else:
        saliency = saliency[:, :, None]

    if explainer.likelihood_type == 'classification':
        preds_sal, acc = explainer.predict(saliency * obs, acts, rewards)      
        return -np.log(preds_sal), acc, np.abs(preds_sal-preds_orin[0])
    else:
        preds_sal = explainer.predict(saliency * obs, acts, rewards)
        return np.abs(preds_sal-preds_orin)


def exp_fid2nn_topk(obs, acts, rewards, explainer, saliency, preds_orin, num_fea):
    obs.requires_grad = False
    acts.requires_grad = False

    if type(saliency) == torch.Tensor:
        saliency = saliency.cpu().detach().numpy()

    importance_id_sorted = np.argsort(saliency, axis=1)[:, ::-1] # high to low.
    nonimportance_id = importance_id_sorted[:, num_fea:]
    nonimportance_id = nonimportance_id.copy()
    mask = torch.ones_like(acts, dtype=torch.long)
    for j in range(acts.shape[0]):
        mask[j, nonimportance_id[j, ]] = 0

    if explainer.likelihood_type == 'classification':
        preds_sal, acc = explainer.predict(obs * mask[:, :, None, None, None], acts * mask, rewards)
        return -np.log(preds_sal), acc, np.abs(preds_sal-preds_orin[0])
    else:
        preds_sal = explainer.predict(obs * mask[:, :, None, None, None], acts * mask, rewards)
        return np.abs(preds_sal-preds_orin)


def exp_stablity(obs, acts, rewards, explainer, saliency, num_sample=5, eps=0.05):
    # eps: perturbation strength, in pong game, the input value range is (0, 1), add noise range between (0, 1).
    def get_l2_diff(x, y):
        diff_square = np.square((x - y))
        if len(x.shape) > 2:
            diff_square_sum = np.apply_over_axes(np.sum, diff_square, list(range(2, len(x.shape))))
            diff_square_sum = diff_square_sum.reshape(x.shape[0], x.shape[1])
        else:
            diff_square_sum = diff_square
        diff_square_sum = diff_square_sum.sum(-1)
        diff = np.sqrt(diff_square_sum)
        return diff
    stab = []
    for _ in range(num_sample):
        noise = torch.rand(obs.shape) * eps
        noisy_obs = obs + noise
        diff_x = get_l2_diff(obs.cpu().detach().numpy(), noisy_obs.cpu().detach().numpy())
        diff_x[diff_x==0] = 1e-8
        noisy_saliency = explainer.get_explanations_by_tensor(noisy_obs, acts, rewards)
        diff_x_sal = get_l2_diff(saliency, noisy_saliency)
        stab.append((diff_x_sal / diff_x))
    stab = np.array(stab)
    stab = np.max(stab, axis=0)

    return stab


def draw_fid_fig(metric_values, save_path):
    """
    :params: metric_values: [num_method(5), 4, num_traj]
    """
    label_list = []
    value_list = []
    explainer_list = []

    for metric_type in ['ZeroOne', 'Top10', 'Top25', 'Top50']:

        for i in range(metric_values.shape[2]):
            label_list.append(metric_type)
            if metric_type == 'ZeroOne':
                value_list.append(metric_values[0, 0, i])
            if metric_type == 'Top10':
                value_list.append(metric_values[0, 1, i])
            if metric_type == 'Top25':
                value_list.append(metric_values[0, 2, i])
            if metric_type == 'Top50':
                value_list.append(metric_values[0, 3, i])

            explainer_list.append('Rudder')

        for i in range(metric_values.shape[2]):
            label_list.append(metric_type)
            if metric_type == 'ZeroOne':
                value_list.append(metric_values[1, 0, i])
            if metric_type == 'Top10':
                value_list.append(metric_values[1, 1, i])
            if metric_type == 'Top25':
                value_list.append(metric_values[1, 2, i])
            if metric_type == 'Top50':
                value_list.append(metric_values[1, 3, i])

            explainer_list.append('Saliency')

        for i in range(metric_values.shape[2]):
            label_list.append(metric_type)
            if metric_type == 'ZeroOne':
                value_list.append(metric_values[2, 0, i])
            if metric_type == 'Top10':
                value_list.append(metric_values[2, 1, i])
            if metric_type == 'Top25':
                value_list.append(metric_values[2, 2, i])
            if metric_type == 'Top50':
                value_list.append(metric_values[2, 3, i])

            explainer_list.append('Attn')

        for i in range(metric_values.shape[2]):
            label_list.append(metric_type)
            if metric_type == 'ZeroOne':
                value_list.append(metric_values[3, 0, i])
            if metric_type == 'Top10':
                value_list.append(metric_values[3, 1, i])
            if metric_type == 'Top25':
                value_list.append(metric_values[3, 2, i])
            if metric_type == 'Top50':
                value_list.append(metric_values[3, 3, i])

            explainer_list.append('RatNet')

        for i in range(metric_values.shape[2]):
            label_list.append(metric_type)
            if metric_type == 'ZeroOne':
                value_list.append(metric_values[4, 0, i])
            if metric_type == 'Top10':
                value_list.append(metric_values[4, 1, i])
            if metric_type == 'Top25':
                value_list.append(metric_values[4, 2, i])
            if metric_type == 'Top50':
                value_list.append(metric_values[4, 3, i])

            explainer_list.append('Our')

    data_pd = pd.DataFrame({'Metric': value_list, 'Label': label_list, 'explainer': explainer_list})
    figure = plt.figure(figsize=(20, 6))
    ax = sns.boxplot(x="Label", y="Metric", hue="explainer", data=data_pd, whis=2,
                     hue_order=['Rudder', 'Saliency', 'Attn', 'RatNet', 'Our'])

    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={'size': 30})
    ax.set_ylabel('Metric', fontsize=35)
    ax.set_xlabel('')
    ax.tick_params(axis='both', which='major', labelsize=35)
    pp = PdfPages(save_path)
    pp.savefig(figure, bbox_inches='tight')
    pp.close()
    return 0


def draw_stab_fig(metric_values, save_path):
    """
    :params: metric_values: [num_method(5), num_traj]
    """
    value_list = []
    explainer_list = []

    for i in range(metric_values.shape[1]):
        value_list.append(metric_values[0, i])
        explainer_list.append('Rudder')

    for i in range(metric_values.shape[2]):
        value_list.append(metric_values[1, i])
        explainer_list.append('Saliency')

    for i in range(metric_values.shape[2]):
        value_list.append(metric_values[2, i])
        explainer_list.append('Attn')

    for i in range(metric_values.shape[2]):
        value_list.append(metric_values[3, i])
        explainer_list.append('RatNet')

    for i in range(metric_values.shape[2]):
        value_list.append(metric_values[4, i])
        explainer_list.append('Our')

    data_pd = pd.DataFrame({'Metric': value_list, 'explainer': explainer_list})
    figure = plt.figure(figsize=(20, 6))
    ax = sns.boxplot(x="explainer", y="Metric", data=data_pd, whis=2)

    ax.set_ylabel('Metric', fontsize=35)
    ax.set_xlabel('explainer')
    ax.tick_params(axis='both', which='major', labelsize=35)
    pp = PdfPages(save_path)
    pp.savefig(figure, bbox_inches='tight')
    pp.close()
    return 0
