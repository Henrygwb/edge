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
    grid_height = int(np.sqrt(num_imgs))
    grid_width = int(num_imgs/grid_height)
    while grid_height*grid_width != num_imgs:
        grid_height += 1
        grid_width = int(num_imgs/grid_height)
    print(f'{grid_height}x{grid_width} Plot')
    from matplotlib import gridspec

    height = 10
    width = 5
    plt.figure(figsize=(width,height))
    gs = gridspec.GridSpec(grid_height, grid_width,
         wspace=0.1, hspace=0.2, 
         top=1.-0.5/(grid_height+1), bottom=0.5/(grid_height+1), 
         left=0.5/(grid_width+1), right=1-0.5/(grid_width+1))

    for i in range(grid_height):
        for j in range(grid_width):
            img = obs[i*grid_width + j]
            ax = plt.subplot(gs[i,j])
            ax.set_axis_off()
            ax.set_title(i*grid_width + j + min_idx + 1, fontsize=6, pad=0)
            ax.imshow(img)
            
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