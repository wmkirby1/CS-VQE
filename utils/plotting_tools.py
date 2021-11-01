import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

plt.style.use('ggplot')

def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)
    # order the factors
    if val > val2:
        val, val2 = val2, val

    return val, val2


def plot_cs_vqe_convergence(data, title):
    """
    """
    fig, axs = plt.subplots(nrows = data['rows'], ncols = data['cols'], figsize = (6*data['cols'],6*data['rows']))

    for index, grid in enumerate(data['grid_pos']):
        if type(grid) != tuple:
            grid_ref = str(tuple(grid))
            grid = tuple(map(int, grid_ref[1:-1].split(', ')))
        else:
            grid_ref = grid

        vqe_result = data[grid_ref]
        
        X = vqe_result['numfev']
        Y = vqe_result['energy']

        # plot results in corresponding subfigure
        l1 = axs[grid].plot(X, Y, color='black', zorder=2, label='CS-VQE optimisation')
        l2 = axs[grid].hlines(data['noncon'], X[0], X[-1], color='r', zorder=0, label='Noncontextual ground state energy')
        l3 = axs[grid].hlines(data['truegs'], X[0], X[-1], color='g', zorder=1, label='True ground state energy')
        # creating legend labels for target and convergence value
        l4 = axs[grid].plot([1], [0], color='purple', ls='--', zorder=3, label='Target Value')
        l5 = axs[grid].plot([1], [0], color='b', ls='--', zorder=4, label='Convergence Value')
        l6 = axs[grid].plot([1], [0], color='pink', zorder=5, label='Chemical accuracy')
        
        #axs[grid].hlines(vqe_result['projected_target'], X[0], X[-1], color='orange', ls='--')
        # expectation value of A
        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #axs[grid].text(0.7, 0.45, '<ψ|A|ψ>='+str(round(vqe_result['A_expct'], 10)), transform = axs[grid].transAxes, bbox=props)

        #axs[grid].set_xticks(X)
        if vqe_result['num_sim_q'] == data['num_qubits']:
            axs[grid].set_title("Full VQE")
        else:  
            axs[grid].set_title("Simulating %i/%i qubits" % (vqe_result['num_sim_q'], data['num_qubits']))
        #axs[grid].set_xticklabels([])

        # plotting zoomed portion of graph to observe convergence
        X_zoom = []
        Y_zoom = []
        for index, t in enumerate(Y):
            if t-0.01 < data['noncon']:
                X_zoom.append(X[index])
                Y_zoom.append(t)

        if X_zoom == []:
            X_zoom = [0, 1]
            Y_zoom = [0, 0]

        # location for the zoomed portion
        ax_box = axs[grid].get_position()
        ax_origin = (ax_box.get_points())[1]
        sf_size = 0.1
        sub_axes = plt.axes([ax_origin[0]-sf_size*1.1, ax_origin[1]-sf_size*1.4, sf_size*1.1, sf_size*1.4])
        
        # plot the zoomed portion
        sub_axes.set_ylim((data['truegs']-0.01, data['noncon']+0.01))
        sub_axes.plot(X_zoom, Y_zoom, color='black')
        # lines fixed at noncontextual ground energy and true ground energy
        sub_axes.hlines(data['noncon'], X_zoom[0], X_zoom[-1], color='r')
        sub_axes.hlines(data['truegs'], X_zoom[0], X_zoom[-1], color='g')
        sub_axes.hlines(data['truegs']+0.0016, X_zoom[0], X_zoom[-1], color='pink')
        # plotting the target value given the number of qubits simulated
        sub_axes.hlines(vqe_result['target'], X_zoom[0], X_zoom[-1], color='purple', ls='--')
        sub_axes.text(x=median(X_zoom), y=vqe_result['target']-0.005, s= 'target = '+str(round(vqe_result['target'], 4)), size='small')
        # plotting the convergence value
        if vqe_result['result']<data['noncon']+0.1:
            sub_axes.hlines(vqe_result['result'], X_zoom[0], X_zoom[-1], color='b', ls='--')
            sub_axes.text(x=X_zoom[0], y=vqe_result['result']-0.005, s= 'min = '+str(round(Y[-1], 4)), size='small')
        #sub_axes.hlines(vqe_result['projected_target'], X_zoom[0], X_zoom[-1], color='orange', ls='--')
        
        if data['rows'] != 1:
            if grid[0] == data['rows']-1:
                axs[grid].set_xlabel('Optimisation count',fontsize=16)
            if grid[1] == 0:
                axs[grid].set_ylabel('Energy (Ha)',fontsize=18)
        else:
            axs[grid].set_xlabel('Optimisation count',fontsize=16)
            if grid == 0:
                axs[grid].set_ylabel('Energy (Ha)',fontsize=18)

    handles, labels = axs[grid].get_legend_handles_labels()
    order = [0,5,2,4,1,3]

    fig.legend([handles[i] for i in order],
            [labels[i] for i in order],
                loc="lower center",   # Position of legend
                borderaxespad=0.1,    # Small spacing around legend box
                ncol=3,
                bbox_to_anchor=(0.5, -0.02),
                fancybox=True, 
                shadow=True,
                prop={'size': 15})

    #fig.suptitle(title, fontsize=20, y=0.96)
    
    return fig


def plot_cs_vqe_convergence_alt(data, title=None, max_num_plots=None,x=None,y=None,plot_index=None):
    """
    """
    if max_num_plots is not None:
        if (x is None) and (y is None):
            y, x = factor_int(max_num_plots)
        if y==1:
            grids = list(range(max_num_plots))
        else:
            grids = list(itertools.product(range(y), range(x)))
        if plot_index is None:
            grid_map = data['grid_pos'][-max_num_plots:]
        else:
            grid_map = data['grid_pos']
    else:
        x = data['cols']
        y = data['rows']
        grids = data['grid_pos']
    fig, axs = plt.subplots(nrows = y, ncols = x, figsize = (6*x,6*y))

    for index, grid in enumerate(grids):
        if max_num_plots is None:
            if y!=1 and type(grid) != tuple:
                grid_ref = str(tuple(grid))
                grid_ref = tuple(map(int, grid_ref[1:-1].split(', ')))
            else:
                grid_ref = grid
        else:
            if y!=1:
                if plot_index is None:
                    grid_ref = str(tuple(grid_map[index]))
                    grid_ref = tuple(map(int, grid_ref[1:-1].split(', ')))
                else:
                    i = plot_index[index]
                    grid_ref = str(tuple(grid_map[i]))
                    grid_ref = tuple(map(int, grid_ref[1:-1].split(', ')))
            else:
                grid_ref = grid

        if y!=1:
            vqe_result = data[str(grid_ref)]
        else:
            vqe_result = data[grid_ref]
            
        X = vqe_result['numfev']
        Y = vqe_result['energy']
        
        # plotting zoomed portion of graph to observe convergence
        X_zoom = []
        Y_zoom = []
        for index, t in enumerate(Y):
            if t-0.01 < data['noncon']:
                X_zoom.append(X[index])
                Y_zoom.append(t)

        if X_zoom == []:
            X_zoom = [0, 1]
            Y_zoom = [0, 0]

        axs[grid].set_ylim((data['truegs']-0.01, data['noncon']+0.01))
        # plot results in corresponding subfigure
        l1 = axs[grid].plot(X_zoom, Y_zoom, color='black', zorder=2, label='VQE optimisation')
        l2 = axs[grid].hlines(data['noncon'], X_zoom[0], X_zoom[-1], color='r', zorder=0, label='Noncontextual ground state energy')
        l3 = axs[grid].hlines(data['truegs'], X_zoom[0], X_zoom[-1], color='g', zorder=1, label='True ground state energy')
        # creating legend labels for target and convergence value
        l4 = axs[grid].hlines(vqe_result['target'], X_zoom[0], X_zoom[-1], color='purple', ls='--', zorder=3, label='Contextual target')
        l5 = axs[grid].hlines(data['truegs']+0.0016, X_zoom[0], X_zoom[-1], color='pink', zorder=5, label='Chemical accuracy')
        axs[grid].text(x=0.9*median(X_zoom), y=vqe_result['target']-0.003, s= 'Contextual target: '+str(round(vqe_result['target'], 4)), size='medium')
        if vqe_result['result']<data['noncon']+0.1:
            axs[grid].text(x=X_zoom[0], y=vqe_result['result']-0.003, s= 'VQE result: '+str(round(Y[-1], 4)), size='medium')
        #axs[grid].hlines(vqe_result['projected_target'], X[0], X[-1], color='orange', ls='--')
        # expectation value of A
        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #axs[grid].text(0.7, 0.45, '<ψ|A|ψ>='+str(round(vqe_result['A_expct'], 10)), transform = axs[grid].transAxes, bbox=props)

        #axs[grid].set_xticks(X)
        if vqe_result['num_sim_q'] == data['num_qubits']:
            axs[grid].set_title("Full VQE", fontsize=18)
        else:  
            axs[grid].set_title("Simulating %i/%i qubits" % (vqe_result['num_sim_q'], data['num_qubits']),fontsize=18)
        #axs[grid].set_xticklabels([])
        
        if data['rows'] != 1:
            if y==1:
                axs[grid].set_xlabel('Optimisation count',fontsize=18)
                if grid==0:
                    axs[grid].set_ylabel('Energy (Ha)',fontsize=18)
            else:
                if grid[0] == y-1:
                    axs[grid].set_xlabel('Optimisation count',fontsize=18)
                if grid[1] == 0:
                    axs[grid].set_ylabel('Energy (Ha)',fontsize=18)
        else:
            axs[grid].set_xlabel('Optimisation count',fontsize=18)
            if grid == 0:
                axs[grid].set_ylabel('Energy (Ha)',fontsize=18)

    handles, labels = axs[grid].get_legend_handles_labels()
    order = [0,4,2,3,1]

    fig.legend([handles[i] for i in order],
               [labels[i] for i in order],
                loc="lower center",   # Position of legend
                borderaxespad=0.1,    # Small spacing around legend box
                ncol=3,
                bbox_to_anchor=(0.5, -0.01),
                fancybox=True, 
                shadow=True,
                prop={'size': 18})

    #fig.suptitle(title, fontsize=20, y=0.96)
    
    return fig


def plot_parameter_settings(data, title=None):
    """
    """
    params = list(data['params'].keys())
    num_params = len(params)

    fig, axs = plt.subplots(nrows = 2, ncols = 1,
                            sharex=True,
                            figsize=(8, 8))

    axs[0].set_title('Parameter settings')
    for index, p in enumerate(params):
        c = plt.cm.jet(index/(num_params))
        axs[0].plot(data['numfev'], data['params'][p], label=p, color=c)
    axs[0].set_ylabel('Parameter value')

    X, Y = data['numfev'], data['energy']

    axs[1].set_title('Optimiser output')
    axs[1].plot(X, Y, color='black')
    axs[1].set_xlabel('Optimisation count')
    axs[1].set_ylabel('Energy (Ha)')

    # plot results in corresponding subfigure
    l1 = axs[1].plot(X, Y, color='black', zorder=2, label='CS-VQE optimisation')
    # creating legend labels for target and convergence value
    l2 = axs[1].plot([1], [data['noncon']], color='r', zorder=0, label='Noncontextual ground state energy')
    l3 = axs[1].plot([1], [data['truegs']], color='g', zorder=1, label='True ground state energy')
    l4 = axs[1].plot([1], [data['target']], color='purple', ls='--', zorder=3, label='Target Value')
    l5 = axs[1].plot([1], [data['result']], color='b', ls='--', zorder=4, label='Convergence Value')
    l6 = axs[1].plot([1], [data['truegs']+0.0016], color='pink', zorder=5, label='Chemical accuracy')
            

    # plotting zoomed portion of graph to observe convergence
    X_zoom = []
    Y_zoom = []
    for index, t in enumerate(Y):
        if t-0.01 < data['noncon']:
            X_zoom.append(X[index])
            Y_zoom.append(t)

    if X_zoom == []:
        X_zoom = [1, 2]
        Y_zoom = [0, 0]

    # location for the zoomed portion
    ax_box = axs[1].get_position()
    ax_origin = (ax_box.get_points())[1]
    sf_size = 0.2
    sub_axes = plt.axes([ax_origin[0]-sf_size*1.4, ax_origin[1]-sf_size*1.3, sf_size*1.4, sf_size*1.3])

    # plot the zoomed portion
    sub_axes.set_ylim((data['truegs']-0.01, data['noncon']+0.01))
    sub_axes.plot(X_zoom, Y_zoom, color='black')
    # lines fixed at noncontextual ground energy and true ground energy
    sub_axes.hlines(data['noncon'], X_zoom[0], X_zoom[-1], label='test',color='r')
    sub_axes.hlines(data['truegs'], X_zoom[0], X_zoom[-1], color='g')
    sub_axes.hlines(data['truegs']+0.0016, X_zoom[0], X_zoom[-1], color='pink')
    # plotting the target value given the number of qubits simulated
    sub_axes.hlines(data['target'], X_zoom[0], X_zoom[-1], color='purple', ls='--')
    sub_axes.text(x=median(X_zoom), y=data['target']-0.005, s= 'target = '+str(round(data['target'], 4)), size='small')
    # plotting the convergence value
    if data['result']<data['noncon']+0.1:
        sub_axes.hlines(data['result'], X_zoom[0], X_zoom[-1], color='b', ls='--')
        sub_axes.text(x=X_zoom[0], y=data['result']-0.005, s= 'min = '+str(round(Y[-1], 4)), size='small')
    #sub_axes.hlines(vqe_result['projected_target'], X_zoom[0], X_zoom[-1], color='orange', ls='--')

    axs[0].legend(loc="right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
            ncol=3,
            bbox_to_anchor=(1.5, 0.5),
            fancybox=True, 
            shadow=True)

    axs[1].legend(loc="right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
            ncol=1,
            bbox_to_anchor=(1.5, 0.5),
            fancybox=True, 
            shadow=True)

    if title is not None:
        fig.suptitle(title, fontsize=15, y=0.98)

    return fig


def plot_parameter_settings_alt(data, title=None, log_error=False):
    """
    """
    params = list(data['params'].keys())
    num_params = len(params)

    fig, axs = plt.subplots(nrows = 2, ncols = 1,
                            sharex=True,
                            figsize=(8, 8))

    axs[1].set_title('Parameter settings')
    for index, p in enumerate(params):
        c = plt.cm.jet(index/(num_params))
        axs[1].plot(data['numfev'], data['params'][p], label=p, color=c, linewidth=1)
    axs[1].set_xlabel('Optimisation count')
    axs[1].set_ylabel('Parameter value')

    X, Y = data['numfev'], [data['energy']]
    Y = [v-data['truegs'] for v in data['energy']]

    axs[0].set_title('%s %s optimiser output'%(data['backnd'],data['optmzr']))
    axs[0].set_ylabel('Error (Ha)')#'Energy (Ha)')

    # plot results in corresponding subfigure
    if data['stddev']:
        stddev_upper = [a+b for a,b in list(zip(Y, data['stddev']))]
        stddev_lower = [a-b for a,b in list(zip(Y, data['stddev']))]
        if log_error:
            stddev_upper = [np.log10(abs(s)) for s in stddev_upper]
            stddev_lower = [np.log10(abs(s)) for s in stddev_lower]
        axs[0].fill_between(X, stddev_lower, stddev_upper,alpha=0.8, label='Standard deviation')
    
    convrge = data['result']-data['truegs']
    chemacc = 0.0016
    if log_error:
        Y = [np.log10(abs(y)) for y in Y]
        convrge = np.log10(abs(convrge))
        chemacc = np.log10(chemacc)
    # creating legend labels for target and convergence value
    #l2 = axs[1].plot([1], [data['noncon']], color='r', zorder=0, label='Noncontextual ground state energy')
    l1 = axs[0].plot(X, Y, color='black', label='CS-VQE optimisation', linewidth=1)
    l2 = axs[0].hlines(convrge,0,X[-1], color='b', ls=':', label='Convergence Value')
    l3 = axs[0].hlines(chemacc,0,X[-1], color='green',label='Chemical accuracy')
    #l4 = axs[0].hlines(data['truegs'], 0, X[-1], color='g', label='True ground state energy')
    #l4 = axs[1].plot([1], [data['target']], color='purple', ls='--', zorder=3, label='Target Value')
    
    
    axs[1].legend(loc="right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
            ncol=3,
            bbox_to_anchor=(1.5, 0.5),
            fancybox=True, 
            shadow=True)

    axs[0].legend(loc="right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
            ncol=1,
            bbox_to_anchor=(1.38, 0.5),
            fancybox=True, 
            shadow=True)

    if title is not None:
        fig.suptitle(title, fontsize=15, y=0.98)

    return fig