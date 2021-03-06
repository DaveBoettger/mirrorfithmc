import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import scipy

def plot_dataset_3d(ds, ax=None, marker=None, invert_marker=False, color_val=None, cmap=None, vmin=None, vmax=None, dist_unit = 'mm', cbar_label='', tick_font_size=6, label_font_size=12, label_pad=None, point_size=20):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    ax.tick_params(labelsize=tick_font_size)

    if color_val is not None:
        if cmap is None:
            cmap = matplotlib.cm.get_cmap()
        if vmin is None:
            vmin = np.min(color_val)
        if vmax is None:
            vmax = np.max(color_val)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        mapper.set_array(color_val)
        colors = mapper.to_rgba(color_val)
    typename = type(ds).__name__.split('.')
    if 'DatasetArrays' in typename:
        pos = ds.pos
    elif 'DatasetTensors' in typename:
        pos = ds.pos.eval()
    elif 'Dataset' in typename:
        if marker is not None:
            ds = ds.subset_from_marker(marker, invert=invert_marker)
        plot_array = ds.to_arrays()
        pos = plot_array.pos
    elif type(ds) == np.ndarray:
        pos = ds

    if color_val is not None:
        ax.scatter(pos[0,:], pos[1,:], pos[2,:],c=colors, s=point_size, depthshade=False)
    else:
        ax.scatter(pos[0,:], pos[1,:], pos[2,:], s=point_size, depthshade=False)
    if color_val is not None:
        cb = fig.colorbar(mapper)
        cb.set_label(cbar_label, fontsize=label_font_size)
    ax.set_xlabel(f'X [{dist_unit}]', fontsize=label_font_size, labelpad=label_pad)
    ax.set_ylabel(f'Y [{dist_unit}]', fontsize=label_font_size, labelpad=label_pad)
    ax.set_zlabel(f'Z [{dist_unit}]', fontsize=label_font_size, labelpad=label_pad)

    return ax

def plot_dict_correlation(data, plot_keys=None, plot_labels=None, plot_credible_levels=False, figsize=(15,15), tick_font_size=6, label_font_size=12, label_pad=20,figaxes=None, point_color='b'):
    if plot_keys is None:
        plot_keys = list(data.keys())
    if plot_labels is None:
        plot_labels = plot_keys
    if figaxes is None:
        fig,axes = matplotlib.pyplot.subplots(nrows=len(plot_keys), ncols=len(plot_keys), figsize=figsize)
    else:
        fig, axes = figaxes
    for one,lone,axone in zip(plot_keys,plot_labels,axes):
        for two,ltwo,ax in zip(plot_keys,plot_labels,axone):
            ax.tick_params(labelsize=tick_font_size)
            if two == plot_keys[0]:
                #In the first column, so label y-axis:
                ax.set_ylabel(lone, labelpad=label_pad,fontsize=label_font_size)
            if one == plot_keys[0]:
                #In fist row, so label x-axis:
                ax.set_title(ltwo,fontsize=label_font_size)
            if one == two:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, f'$\sigma={np.std(data[one]):.2f}$', horizontalalignment='center', verticalalignment='center', fontsize=label_font_size, transform=ax.transAxes)
                continue
            if plot_credible_levels:
                levels = find_credible_levels(data[two],data[one])
                print('found levels')
                sns.kdeplot(data[two],data[one], cmap='Blues',shade=True, shade_lowest=False, levels=levels, ax=ax)
                ax.scatter(np.mean(data[two]), np.mean(data[one]),c='r')
            else:
                ax.plot(data[two],data[one],f'{point_color}.')
    return fig, axes

def plot_dist(data_dict, keys=None, labels=None, shape=None, figsize=5, label_font_size=12, tick_font_size=6):

    if keys is None:
        keys = list(data_dict.keys())
    if labels is None:
        labels = keys

    if shape is None:
        min_side = int(np.ceil(np.sqrt(len(keys))))
        nrows=min_side
        ncols=min_side
    else:
        nrows = shape[0]
        ncols = shape[1]
    fig,axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(figsize*ncols,figsize*nrows))
    for k,lbl,ax in zip(keys, labels, axes.flat):
        print(k)
        ax.set_yticks([])
        ax.tick_params(labelsize=tick_font_size)
        sns.distplot(data_dict[k],ax=ax)
        ax.set_xlabel(lbl, fontsize=label_font_size)
        ax.set_ylabel('Probability Density', fontsize=label_font_size)
    return fig,axes

def find_credible_levels(x,y,contour_targets=[.997,.954,.683]):
    '''Adapted from https://stackoverflow.com/questions/35225307/set-confidence-levels-in-seaborn-kdeplot'''

    # Make a 2d normed histogram
    H,xedges,yedges=np.histogram2d(x,y,bins=50,normed=True)

    norm=H.sum() # Find the norm of the sum

    # Set target levels as percentage of norm
    targets = [norm*contour for contour in contour_targets]

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target

    levels = []
    # Find levels by summing histogram to objective
    for target in targets:
        levels.append(scipy.optimize.bisect(objective, H.min(), H.max(), args=(target,)))

    # For nice contour shading with seaborn, define top level
    levels.insert(0,H.min())
    levels.append(H.max())
    return levels
