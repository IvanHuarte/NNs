import numpy as np
import matplotlib.pyplot as plt

def dump_callback(logger, settings, plot =  True, write = False):

    size = settings['size']
    theta = settings['theta']
    phi = settings['phi']
    time_exe = settings['time_exe']
    write_folder = settings['write_folder']
    architecture_display = settings['architecture']
    opt_name = settings['optimizer']
    learning_rate = settings['learning_rate']
    sim_label = settings['sim_label']
    
    # Extract some results
    E_hist = np.array(logger['Energy']['Mean']).real
    dev_E_hist = np.array(logger['Energy']['Sigma']).real
    E_best = min(logger['Energy']['Mean']).real
    if hasattr(logger, 'E_ED'):
        E_gr = np.array(logger['E_ED']).real
        error=np.abs(E_hist-E_gr)/np.abs(E_gr)

    # Calculate the variance score
    var= np.array(logger['Energy']['Variance']).real
    vscore = int(np.prod(settings['size']))*var//(E_hist**2)

    setup_sim = f"E_best: {E_best:.4f} \nopt: {opt_name} \nl_rate: {learning_rate} \ntime_exe: {time_exe:.2f}"
    if hasattr(logger, 'E_ED'):
        setup_sim = setup_sim + f" \nE_ED: {E_gr:.4e}"

    if plot:

        _, ax = plt.subplots(3,1,figsize=(8, 18))
        ax[0].set_title(f"Convergence 4x4 theta=9 phi=72")
        ax[0].errorbar(range(len(E_hist)), E_hist, yerr=dev_E_hist, fmt='none', ecolor='r', label='E_stdev')
        ax[0].plot(E_hist, color='blue', label='E')

        if hasattr(logger, 'E_ED'):
            ax[0].hlines(E_gr,0,len(E_hist), color='green', label='ED Energy')

        ax[0].text(0.45, 0.93, architecture_display, transform=ax[0].transAxes, fontsize=12, color='k', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax[0].text(0.9, 0.75, setup_sim, transform=ax[0].transAxes, fontsize=10, color='k', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax[0].legend()
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Energy', fontsize=12)
        ax[0].grid()

        ax[1].plot(error, color='red', label='E')
        ax[1].set_yscale('log')
        ax[1].legend()
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Error', fontsize=12)
        ax[1].grid()


        ax[2].plot(vscore, color='purple', label='Vscore')
        ax[2].set_yscale('log')
        ax[2].legend()
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel('Vscore', fontsize=12)
        ax[2].grid()
        plt.tight_layout()
        plt.savefig(write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}.jpeg", dpi=600)
        plt.close()

    # Save the data
    if write:
        np.savetxt(write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}_E_hist.txt",
                np.array(E_hist))
        np.savetxt(write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}_error.txt",
                np.array(error))
        np.savetxt(write_folder + f"Callback_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}_vscore.txt",
                np.array(vscore))
        

def save_results(results, vstate, setup, x_ED = None):
    size = setup['size']
    N = int(np.prod(setup['size']))
    theta = setup['theta']
    phi = setup['phi']
    write_folder = setup['write_folder']
    architecture = setup['architecture']
    opt_name = setup['optimizer']
    learning_rate = setup['learning_rate']

    

    np.savetxt(write_folder + f"Oxalate_{size[0]}x{size[1]}_theta_{theta}_phi_{phi}_{sim_label}_results.txt",
            results)