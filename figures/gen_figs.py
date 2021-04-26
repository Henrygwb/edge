import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_dir', type=str, required=False, default='.', help='input directory')
    parser.add_argument('--output_dir', type=str, required=False, default='./outputs', help='output directory')
    parser.add_argument('--fid', action="store_true", help="Generate fid figure")
    parser.add_argument('--stab', action="store_true", help="Generate stab figure")
    return parser.parse_args()

def draw_fid_fig(args):
    """
    :params: metric_values: [num_method(5), 4, num_traj]
    """
    label_list = []
    value_list = []
    explainer_list = []
    fid_data = np.load(Path(args.metric_dir)/'fid_all.npz')
    for explainer_type in fid_data.files:
        for idx,metric_type in enumerate(['ZeroOne', 'Top5', 'Top15', 'Top25']):
            for metric in fid_data[explainer_type][idx]:
                label_list.append(metric_type)
                explainer_list.append(explainer_type)
                value_list.append(metric)
            
    data_pd = pd.DataFrame({'Metric': value_list, 'Label': label_list, 'explainer': explainer_list})
    figure = plt.figure(figsize=(20, 6))
    ax = sns.barplot(x="Label", y="Metric", hue="explainer", data=data_pd,
                     hue_order=['rudder', 'saliency', 'attention', 'rationale', 'dgp'])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={'size': 30})
    ax.set_ylabel('Metric', fontsize=35)
    ax.set_xlabel('')
    ax.tick_params(axis='both', which='major', labelsize=35)
    pp = PdfPages(Path(args.output_dir)/'fid.pdf')
    pp.savefig(figure, bbox_inches='tight')
    pp.close()

def draw_stab_fig(args):
    """
    :params: metric_values: [num_method(5), num_traj]
    """
    value_list = []
    explainer_list = []
    stab_data = np.load(Path(args.metric_dir)/'stab_all.npz')
    for explainer_type in stab_data.files:
        for metric_val in stab_data[explainer_type]:
            value_list.append(metric_val)
            explainer_list.append(explainer_type)

    data_pd = pd.DataFrame({'Metric': value_list, 'explainer': explainer_list})
    figure = plt.figure(figsize=(20, 6))
    ax = sns.barplot(x="explainer", y="Metric", data=data_pd)
    ax.set_ylabel('Metric', fontsize=35)
    ax.set_xlabel('explainer')
    ax.tick_params(axis='both', which='major', labelsize=35)
    pp = PdfPages(Path(args.output_dir)/'stab.pdf')
    pp.savefig(figure, bbox_inches='tight')
    pp.close()

if __name__ == '__main__':
    args = get_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    if args.fid:
        draw_fid_fig(args)
    if args.stab:
        draw_stab_fig(args)