import math
import itertools
import matplotlib.pyplot as plt
import cs_vqe_classes.cs_vqe_circuit as cs_circ
from statistics import median
from copy import deepcopy


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
        
        X = vqe_result['counts']
        Y = vqe_result['values']

        # plot results in corresponding subfigure
        l1 = axs[grid].plot(X, Y, color='black', zorder=2, label='CS-VQE optimisation')
        l2 = axs[grid].hlines(data['gs_noncon_energy'], X[0], X[-1], color='r', zorder=0, label='Noncontextual ground state energy')
        l3 = axs[grid].hlines(data['true_gs'], X[0], X[-1], color='g', zorder=1, label='True ground state energy')
        # creating legend labels for target and convergence value
        l4 = axs[grid].plot([1], [0], color='purple', ls='--', zorder=3, label='Target Value')
        l5 = axs[grid].plot([1], [0], color='b', ls='--', zorder=4, label='Convergence Value')
        l6 = axs[grid].plot([1], [0], color='pink', zorder=5, label='Chemical accuracy')


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
            if t-0.01 < data['gs_noncon_energy']:
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
        sub_axes.set_ylim((data['true_gs']-0.01, data['gs_noncon_energy']+0.01))
        sub_axes.plot(X_zoom, Y_zoom, color='black')
        # lines fixed at noncontextual ground energy and true ground energy
        sub_axes.hlines(data['gs_noncon_energy'], X_zoom[0], X_zoom[-1], color='r')
        sub_axes.hlines(data['true_gs'], X_zoom[0], X_zoom[-1], color='g')
        sub_axes.hlines(data['true_gs']+0.0015, X_zoom[0], X_zoom[-1], color='pink')
        # plotting the target value given the number of qubits simulated
        sub_axes.hlines(vqe_result['target'], X_zoom[0], X_zoom[-1], color='purple', ls='--')
        sub_axes.text(x=median(X_zoom), y=vqe_result['target']-0.005, s= 'target = '+str(round(vqe_result['target'], 4)), size='small')
        # plotting the convergence value
        if vqe_result['result']<data['gs_noncon_energy']+0.1:
            sub_axes.hlines(vqe_result['result'], X_zoom[0], X_zoom[-1], color='b', ls='--')
            sub_axes.text(x=X_zoom[0], y=vqe_result['result']-0.005, s= 'min = '+str(round(Y[-1], 4)), size='small')

        if data['rows'] != 1:
            if grid[0] == 1:
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
                ncol=3)

    fig.suptitle(title, fontsize=16)
    
    return fig