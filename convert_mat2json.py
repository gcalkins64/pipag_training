import numpy as np
import json
from scipy.io import loadmat, savemat
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    # Constants
    savePath = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data'
    R_eq = 25559e3 # m, Uranus
    mu = 5.7940*10**15,  # m^3/s^2
    Nruns = 5000

    # Near Escape
    # removeFlags = [2]  # 0 capture, 1 escape, 2 crash
    # tag = 'UOP_inc_lit_disps'
    # dataPath = '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250425173914_UOP_inc_lit_disps_R0_C5000_Puranus_O1_Fenergy_FFTrue_DP2'
    # tFind = 1001  # Final time step index
    # norm = 142248870.22982892

    # Near Crash
    # removeFlags = [1]  # 0 capture, 1 escape, 2 crash
    # tag = 'UOP_near_crash'
    # dataPath = '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250425235334_UOP_near_crash_R0_C5000_Puranus_O2_Fenergy_FFTrue_DP2'
    # tFind = 987  # Final time step index
    # norm = 142145066.67666337

    # Agressive Near Crash
    removeFlags = [1]  # 0 capture, 1 escape, 2 crash
    tag = 'UOP_near_crash_steeper'
    dataPath = '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250426100446_UOP_near_crash_steeper_R0_C5000_Puranus_O2_Fenergy_FFTrue_DP2'
    tFind = 867  # Final time step index
    norm = 142141053.1824186

    flagDownsample = True
    flagEnergy = True  # if true, use energy, if false, use velocity
    flagScale = True # if true, scale the data, if false, don't scale

    if flagDownsample:
        downsampleNum = 64
    else:
        downsampleNum = tFind

    if not flagEnergy:
        # Vel norms and offsets
        norm = 10_000  # m/s
        offset = 18_000  # m/s
        dataName = 'vel'
    else:
        # energy norms and offsets
        offset = 0 # TODO
        dataName = 'energy'

    # Load in runs 0 to 5000 from folder that are in mat files
    datas = []
    ras = []
    for run in trange(Nruns):
        # Load in the data
        data = loadmat(f'{dataPath}/run_' + str(run) + '.mat')
        datas.append(data)
        ras.append(data['ra'])

    data_dict = {}
    data_mat = np.zeros((Nruns, downsampleNum))
    if flagDownsample:
        idx = np.linspace(0, tFind - 1, downsampleNum, dtype=int)
    else:
        idx = np.linspace(0, tFind - 1, tFind, dtype=int)
    goodInds = []
    for run in range(Nruns):
        data = datas[run]
        vel = data['x'][3, :tFind]
        r = data['x'][0, :tFind]
        rp = data['rp']

        if ras[run] < 0: # Escape
            label = 1
        elif np.isnan(ras[run]) or (rp - R_eq) < 100e3:
            label = 2
        else: # Capture
            label = 0

        if label in removeFlags:  # Remove from dataset
            continue
        else:
            goodInds.append(run)

        if flagEnergy:
            data = vel**2 / 2 - mu / r
        else:
            data = vel

        if flagScale:
            normed_data = (data - offset) / norm
        else:
            normed_data = data

        data_dict[f'sample{run}'] = {dataName: normed_data[idx].tolist(), 'label': label}
        data_mat[run, :] = normed_data[idx]

    suffix = f'{dataName}_{"scaled_" if flagScale else ""}{"downsampled_" if flagDownsample else ""}'
    with open(f'{savePath}/{tag}_{Nruns}_data_{suffix}.json', 'w') as f:
        json.dump(data_dict, f)

    Nruns = len(data_dict)

    # Save datamat as a mat file
    save_path = f'{savePath}/{tag}_{Nruns}_data_{suffix}.mat'
    savemat(save_path, {'data': data_mat})

    # Plot the data
    sns.set_theme('notebook', style='whitegrid', palette='Paired', rc={"lines.linewidth": 2.5, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, "font.family": "serif"})
    fig, ax = plt.subplots()
    for run in goodInds:
        label = data_dict[f'sample{run}']['label']
        color = f"C{label}"
        ax.plot(data_mat[run, :], color=color)
    line1 = plt.Line2D([0], [0], color='C0', label='Capture')
    line2 = plt.Line2D([0], [0], color='C1', label='Escape')
    line3 = plt.Line2D([0], [0], color='C2', label='Crash')
    ax.legend(handles=[line1, line2, line3])
    plt.ylabel(f'{r"Velocity (m/s)" if not flagEnergy else r"Energy (m$^2$/s$^2$)"}')
    plt.xlabel('Time step')
    if flagEnergy:
        plt.hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
    plt.savefig(f'/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs/{tag}_{Nruns}_data_{suffix}.png', dpi=300)

    # Get maximum initial energy
    max_energy = 0
    mean_energy_0 = 0
    for run in goodInds:
        energy = data_mat[run, 0]
        if energy > max_energy:
            max_energy = energy
        mean_energy_0 += energy
    mean_energy_0 /= len(goodInds)
    print(f'Maximum initial energy: {max_energy}')
    print(f'Mean initial energy: {mean_energy_0}')



if __name__ == '__main__':
    main()
    plt.show()
