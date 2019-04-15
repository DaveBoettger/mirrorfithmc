import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

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

def plot_dict_correlation(data, plot_keys=None, plot_labels=None, figsize=(15,15), tick_font_size=6, label_font_size=12, label_pad=20):
    if plot_keys is None:
        plot_keys = list(data.keys())
    if plot_labels is None:
        plot_labels = plot_keys
    fig,axes = matplotlib.pyplot.subplots(nrows=len(plot_keys), ncols=len(plot_keys), figsize=figsize)
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
            ax.plot(data[two],data[one],'.')
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
    print(data_dict['std'])
    fig,axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(figsize*ncols,figsize*nrows))
    for k,lbl,ax in zip(keys, labels, axes.flat):
        print(k)
        ax.set_yticks([])
        ax.tick_params(labelsize=tick_font_size)
        sns.distplot(data_dict[k],ax=ax)
        ax.set_xlabel(lbl, fontsize=label_font_size)
        ax.set_ylabel('Probability Density', fontsize=label_font_size)
    return fig,axes
