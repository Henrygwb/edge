import numpy as np
import matplotlib.pyplot as plt

import argparse

def visualize(data_file, disp_range, output):
    arr = np.load(data_file)
    obs = arr['observations']

    min_idx = max(0 if disp_range[0] is None else disp_range[0], 0)
    max_idx = min(len(obs) if disp_range[1] is None else disp_range[1], len(obs))

    assert min_idx < max_idx
    obs = obs[min_idx:max_idx]

    num_imgs = max_idx - min_idx
    print(f'{num_imgs} images')
    grid_length = int(np.sqrt(num_imgs))
    grid_width = int(num_imgs/grid_length)
    while grid_length*grid_width != num_imgs:
        grid_length += 1
        grid_width = int(num_imgs/grid_length)
    print(f'{grid_length}x{grid_width} Plot')
    
    height = 20
    width = 10
    plt.figure(figsize=(width,height))

    for idx in range(obs.shape[0]):
        img = obs[idx]
        ax = plt.subplot(int(num_imgs / grid_width + 1), grid_width, idx + 1)
        ax.set_axis_off()
        ax.set_title(f'{idx + min_idx + 1}')
        plt.imshow(img)
    plt.tight_layout()
    plt.savefig(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Data location")
    parser.add_argument('--out', type=str, required=True, help="output location")
    parser.add_argument('--from_idx', type=int, required=False, default=None, help="Min observation to plot")
    parser.add_argument('--to_idx', type=int, required=False, default=None, help="Max observation to plot")
    args = parser.parse_args()
    data, from_idx, to_idx = args.data, args.from_idx, args.to_idx
    disp_range = (from_idx, to_idx)
    visualize(data, disp_range, args.out)