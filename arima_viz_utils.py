import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_timeseries2d(timeseries, dates, suptitle, savepath):
    min_ts = np.nanmin(timeseries)
    max_ts = np.nanmax(timeseries)
    n_frames = timeseries.shape[0]
    fr_shape = timeseries.shape[1:]
    n_subplt_perfig = 25
    n_plt_per_axis = int(np.sqrt(n_subplt_perfig))
    
    if (n_frames % n_subplt_perfig == 0):
        n_figs = int(n_frames / n_subplt_perfig)
    else:
        n_figs = int(np.ceil(n_frames / n_subplt_perfig))
    
    fig_n = 0
    n_y, n_x = 0, 0
    fig, axs = plt.subplots(n_plt_per_axis, n_plt_per_axis, figsize=(50, 40))
    
    for i, fr in enumerate(timeseries):
        t = dates[i]
        
        title = f"{t[:4]}-{t[4:6]}-{t[6:]}"

        if (n_x < n_plt_per_axis):
            cmap = axs[n_y, n_x].imshow(fr, cmap='jet', vmin=min_ts, vmax=max_ts, interpolation=None, aspect='auto')
            axs[n_y, n_x].set_title(title, fontweight="bold", size=40)
            n_x += 1

        else:
            n_y += 1
            n_x = 0

            if (n_y < n_plt_per_axis):
                cmap = axs[n_y, n_x].imshow(fr, cmap='jet', vmin=min_ts, vmax=max_ts, interpolation=None, aspect='auto')
                axs[n_y, n_x].set_title(title, fontweight="bold", size=40)
                n_x += 1
            else:
                cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
                cbar_ax.tick_params(labelsize=50) 
                fig.colorbar(cmap, cax=cbar_ax)

                fig.suptitle(suptitle, fontsize=50)
                plt.savefig(f"{savepath}_{fig_n}.png")
                fig, axs = plt.subplots(n_plt_per_axis, n_plt_per_axis, figsize=(50, 40))
                n_y, n_x = 0, 0
                fig_n += 1

                cmap = axs[n_y, n_x].imshow(fr, cmap='jet', vmin=min_ts, vmax=max_ts, interpolation=None, aspect='auto')
                axs[n_y, n_x].set_title(title, fontweight="bold", size=40)
                n_x += 1
                
    if (n_y != 0 or n_x != 0):
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        cbar_ax.tick_params(labelsize=50)
        fig.colorbar(cmap, cax=cbar_ax)


def get_coord_position(size, n_points):
    space = size // (n_points+1)
    rest = size % (n_points+1)
    
    if (rest % 2 == 0):
        init = int(space + (rest/2))
        #end = int(size - init)
    else:
        init = int(space + np.ceil(rest/2))
        #end = int(size - (space + np.ceil(rest/2)))
    
    coord = []
    for i in range(0, n_points):
        coord.append((init + (i*space)))
        
    return coord
    
def get_ticks_and_labels(val_range, n_ticks, start_date_idx, dates):  
    xticks_pos = np.linspace(val_range[0], val_range[1], n_ticks, dtype='int')
    xticks_labels = [dates[i+start_date_idx] for i in xticks_pos]
    return xticks_pos, xticks_labels

def extract_series(timeseries, x_pos, y_pos):
    sampled_series = []
    # Lists useful for plotting
    x_coord, y_coord = [], []
    coord = []
    for y in y_pos: 
        for x in x_pos:
            sampled_series.append(timeseries[:, y, x])
            y_coord.append(y)
            x_coord.append(x)
            coord.append((x,y))
    return sampled_series, x_coord, y_coord, coord

def plot_timeseries1d(pred, real, dates, n_xaxis, n_yaxis, suptitle, savepath):
    assert (real.shape == pred.shape)
    
    ysize, xsize = real.shape[1], real.shape[2]
    x_pos, y_pos = [], []
    
    # Extracting series
    x_pos = get_coord_position(xsize, n_xaxis)
    y_pos = get_coord_position(ysize, n_yaxis)
    sampled_series_pred, x_coord, y_coord, coord = extract_series(pred, x_pos, y_pos)
    sampled_series_real, x_coord, y_coord, coord = extract_series(real, x_pos, y_pos)
            
    # Example frame for reference
    plt.figure()
    plt.title("Observing Series")
    plt.plot(x_coord, y_coord, 'kX')
    plt.imshow(real[-1, :, :], vmin=np.min(real), vmax=np.max(real), cmap='jet')
    plt.colorbar()
    plt.savefig(os.path.join(savepath, 'sampled_pix.png'))
    
    ## Plotting series side by side
    fig_n = 0
    n_y, n_x = 0, 0
    fig, axs = plt.subplots(n_yaxis, n_xaxis, figsize=(40, 40))
    nt = real.shape[0]
    xticks_pos, xticks_labels = get_ticks_and_labels([0, nt-1], 5, 0, dates)
    
    assert (len(sampled_series_pred) == (n_xaxis * n_yaxis) == len(sampled_series_real))

    for i, real in enumerate(sampled_series_real):
        pred = sampled_series_pred[i]

        if (n_x < n_xaxis):
            axs[n_y, n_x].plot(real, 'b-', label='Real Values')
            axs[n_y, n_x].plot(pred, 'r-', label='Predicted Values')

            axs[n_y, n_x].set_xticks(xticks_pos)
            axs[n_y, n_x].set_xticklabels(xticks_labels)
            axs[n_y, n_x].set_ylabel('Displacement')
            axs[n_y, n_x].set_xlabel('Time Sample')
            handles, labels = axs[n_y, n_x].get_legend_handles_labels()
            axs[n_y, n_x].legend(handles, labels, loc='lower right')
            n_x += 1
        else:
            n_y += 1
            n_x = 0

            if (n_y < n_yaxis):
                axs[n_y, n_x].plot(real, 'b-', label='Real Values')
                axs[n_y, n_x].plot(pred, 'r-', label='Predicted Values')
                
                axs[n_y, n_x].set_xticks(xticks_pos)
                axs[n_y, n_x].set_xticklabels(xticks_labels)
                axs[n_y, n_x].set_ylabel('Displacement')
                axs[n_y, n_x].set_xlabel('Time Sample')
                n_x += 1
            else:
                break

    fig.suptitle(suptitle, fontsize=50)
    plt.savefig(os.path.join(savepath, '1d_pred_vs_obs.png'))

    return 

