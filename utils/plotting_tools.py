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


def plot_cs_vqe_convergence(hamiltonian, terms_noncon, anz_terms, num_qubits, order, max_sim_q, min_sim_q=0, show_circ=False):
    rows, cols = factor_int(max_sim_q-min_sim_q)

    fig, axs = plt.subplots(nrows = rows, ncols = cols, figsize = (6*cols,6*rows))
    if rows == 1:
        grid_pos = range(cols)
    else:
        grid_pos = list(itertools.product(range(rows), range(cols)))

    circs = cs_circ.cs_vqe_circuit(hamiltonian, terms_noncon, num_qubits, order)
    gs_noncon_energy = circs.gs_noncon_energy
    true_gs = circs.true_gs

    for index, grid in enumerate(grid_pos):

        num_sim_q = index+1+min_sim_q
        print('Simulating %i out of %i qubits...' % (num_sim_q, num_qubits))
        vqe_result, target_energy, X, Y = circs.CS_VQE(anz_terms, num_sim_q)
        print('VQE result:', vqe_result, '|', 'Target energy:', target_energy, '\n')

        # plot results in corresponding subfigure
        l1 = axs[grid].plot(X, Y, color='black', zorder=2)
        l2 = axs[grid].hlines(gs_noncon_energy, X[0], X[-1], color='r', zorder=0)
        l3 = axs[grid].hlines(true_gs, X[0], X[-1], color='g', zorder=1)
        # creating legend labels for target and convergence value
        l4 = axs[grid].plot([1], [0], color='purple', ls='--', zorder=3)
        l5 = axs[grid].plot([1], [0], color='b', ls='--', zorder=4)

        
        axs[grid].set_xticks(X)
        if num_sim_q == num_qubits:
            axs[grid].set_title("Full VQE")
        else:  
            axs[grid].set_title("%i qubits + ancilla" % num_sim_q)
        axs[grid].set_xticklabels([])
        
        # plotting zoomed portion of graph to observe convergence
        X_zoom = []
        Y_zoom = []
        for index, t in enumerate(Y):
            if t-0.01 < gs_noncon_energy:
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
        sub_axes.set_ylim((true_gs-0.01, gs_noncon_energy+0.01))
        sub_axes.plot(X_zoom, Y_zoom, color='black')
        # lines fixed at noncontextual ground energy and true ground energy
        sub_axes.hlines(gs_noncon_energy, X_zoom[0], X_zoom[-1], color='r')
        sub_axes.hlines(true_gs, X_zoom[0], X_zoom[-1], color='g')
        # plotting the target value given the number of qubits simulated
        sub_axes.hlines(target_energy, X_zoom[0], X_zoom[-1], color='purple', ls='--')
        sub_axes.text(x=median(X_zoom), y=target_energy-0.004, s= 'target = '+str(round(target_energy, 4)), size='small')
        # plotting the convergence value
        if vqe_result+0.1<gs_noncon_energy:
            sub_axes.hlines(vqe_result, X_zoom[0], X_zoom[-1], color='b', ls='--')
            sub_axes.text(x=X_zoom[0], y=vqe_result-0.004, s= 'min = '+str(round(Y[-1], 4)), size='small')
        
        fig.legend([l1,l2,l3,l4,l5],
                labels=['CS-VQE optimisation','Target Value','Convergence Value','Noncontextual ground state energy','True ground state energy'],
                loc="lower center",   # Position of legend
                borderaxespad=0.1,    # Small spacing around legend box
                ncol=5)

    if rows != 1:
        if grid[0] == 1:
            axs[grid].set_xlabel('Optimisation count',fontsize=16)
        if grid[1] == 0:
            axs[grid].set_ylabel('Energy (Ha)',fontsize=18)
    else:
        axs[grid].set_xlabel('Optimisation count',fontsize=16)
        if grid == 0:
            axs[grid].set_ylabel('Energy (Ha)',fontsize=18)

    
    return fig