import torch
import timeit
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
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

    mask_obs = torch.ones_like(obs, dtype=torch.float32)
    mask_acts = torch.ones_like(acts, dtype=torch.long)

    for j in range(acts.shape[0]):
        mask_acts[j, nonimportance_id[j,]] = 0
        mask_obs[j, nonimportance_id[j,]] = 0

    if explainer.likelihood_type == 'classification':
        preds_sal, acc = explainer.predict(obs * mask_obs, acts * mask_acts, rewards)
        return -np.log(preds_sal), acc, np.abs(preds_sal-preds_orin[0])
    else:
        preds_sal = explainer.predict(obs * mask_obs, acts * mask_acts, rewards)
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


def truncate_importance(importance, num_importance, percentile=50, thredshold=4):
    """
    :param: importance: ordered importance index from high to low.
    truncate the importance steps into a continuous sequence with a reasonable diff between two consecutive steps.
    e.g. [183,  38, 193, 190, 110, 182, 187, 188, 184, 178] -> [178, 182, 184, 187, 188, 190, 193]
         [38, 39, 40, 183,  38, 193, 190, 110, 182, 187, 188, 184, 178] -> [38, 39, 40, 182, 184, 187, 188, 190]
    """
    importance_selected = importance[:num_importance]
    sorted = np.sort(importance_selected)
    diff = sorted[1:] - sorted[:-1]
    diff_thred = np.percentile(diff, percentile)
    max_diff = np.max(diff)
    if max_diff <= thredshold:
        sorted_final = sorted
    else:
        idx = np.where(diff <= diff_thred)[0]
        x = set()

        for i in range(len(idx)):
            x.add(sorted[idx[i]])
            x.add(sorted[idx[i] + 1])
        sorted_final = np.sort(list(x))

    return np.arange(sorted_final[0], sorted_final[-1]+1)


def draw_fid_fig(fid_data, explainers, metrics, save_path, box_plot=True, log_scale=True):
    """
    :params: fid_data: [num_method, num_metrics, num_traj]
    """
    label_list = []
    value_list = []
    explainer_list = []
    for explainer_idx, explainer_type in enumerate(explainers):
        for idx, metric_type in enumerate(metrics):
            for metric in fid_data[explainer_idx][idx]:
                label_list.append(metric_type)
                explainer_list.append(explainer_type)
                value_list.append(metric)

    data_pd = pd.DataFrame({'Metric': value_list, 'Label': label_list, 'explainer': explainer_list})
    figure = plt.figure(figsize=(20, 6))

    if box_plot:
        ax = sns.boxplot(x="Label", y="Metric", hue="explainer", data=data_pd,
                         hue_order=explainers)
    else:
        ax = sns.barplot(x="Label", y="Metric", hue="explainer", data=data_pd,
                         hue_order=explainers)
        if log_scale:
            ax.set_yscale("log")
            ax.set_yticks([0.01, 0.1, 1, 10])
            # ax.set_yticks([1e-6, 1e-4, 1e-2, 1, 1e2]) # cartpole
            # ax.set_yticks([1e-4, 1e-2, 1, 1e2]) # pendulum

    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={'size': 30})
    ax.set_ylabel('Metric', fontsize=35)
    ax.set_xlabel('')
    ax.tick_params(axis='both', which='major', labelsize=35)
    pp = PdfPages(save_path)
    pp.savefig(figure, bbox_inches='tight')
    pp.close()


def draw_stab_fig(stab_data, explainers, save_path, box_plot=True):
    """
    :params: metric_values: [num_method(5), num_traj]
    """
    value_list = []
    explainer_list = []
    for explainer_idx, explainer_type in enumerate(explainers):
        for metric_val in stab_data[explainer_idx]:
            value_list.append(metric_val)
            explainer_list.append(explainer_type)

    data_pd = pd.DataFrame({'Metric': value_list, 'explainer': explainer_list})
    figure = plt.figure(figsize=(20, 6))
    if box_plot:
        ax = sns.boxplot(x="explainer", y="Metric", data=data_pd)
    else:
        ax = sns.barplot(x="explainer", y="Metric", data=data_pd)
        ax.set_yticks([0.05, 0.1])
    ax.set_ylabel('Metric', fontsize=35)
    ax.set_xlabel('explainer')
    ax.tick_params(axis='both', which='major', labelsize=35)
    pp = PdfPages(save_path)
    pp.savefig(figure, bbox_inches='tight')
    pp.close()


def draw_fid_fig_t(fid_data, explainers, metrics, save_path, box_plot=True, log_scale=True):
    """
    :params: fid_data: [num_metrics, num_method, num_traj]
    """
    label_list = []
    value_list = []
    explainer_list = []
    for idx, metric_type in enumerate(metrics):
        for explainer_idx, explainer_type in enumerate(explainers):
            for metric in fid_data[idx][explainer_idx]:
                label_list.append(metric_type)
                explainer_list.append(explainer_type)
                value_list.append(metric)

    data_pd = pd.DataFrame({'Metric': value_list, 'Label': label_list, 'explainer': explainer_list})
    figure = plt.figure(figsize=(20, 6))

    if box_plot:
        ax = sns.boxplot(x="Label", y="Metric", hue="explainer", data=data_pd,
                         hue_order=explainers)
    else:
        ax = sns.barplot(x="Label", y="Metric", hue="explainer", data=data_pd,
                         hue_order=explainers)
        if log_scale:
            ax.set_yscale("log")
            ax.set_yticks([0.01, 10])

    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={'size': 30})
    ax.set_ylabel('Metric', fontsize=35)
    ax.set_xlabel('')
    ax.tick_params(axis='both', which='major', labelsize=35)
    pp = PdfPages(save_path)
    pp.savefig(figure, bbox_inches='tight')
    pp.close()


def compute_rl_fid(diff, len, diff_max, len_max=200, eps=0.001, weight=1):
    diff = diff / diff_max
    len = len / len_max
    diff[diff == 0] =eps
    len[len == 0] =eps
    diff_log = np.log(diff)
    len_log = np.log(len)
    return len_log - weight*diff_log

