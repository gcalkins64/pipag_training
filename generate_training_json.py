import numpy as np
import json
from scipy.io import loadmat, savemat
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Determine proportional splits for escape and crash
def split_proportional(indices, proportions):
    np.random.shuffle(indices)
    total = len(indices)
    n_train = int(proportions[0] * total)
    n_val = int(proportions[1] * total)
    n_test = total - n_train - n_val
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:]
    )


def main():
    np.random.seed(1)
    # Constants
    savePath = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data'
    R_eq = 25559e3 # m, Uranus
    mu = 5.7940*10**15,  # m^3/s^2
    Nruns = 2000

    # Near Escape
    # removeFlags = [2]  # 0 capture, 1 escape, 2 crash
    # tag = 'UOP_inc_lit_disps'
    # dataPath = '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250425173914_UOP_inc_lit_disps_R0_C5000_Puranus_O1_Fenergy_FFTrue_DP2'
    # tFind = 1001  # Final time step index
    # norm = 142248870.22982892

    # Near Crash
    # removeFlags = []  # 0 capture, 1 escape, 2 crash
    # tag = 'UOP_near_crash_extend'
    # dataPath = '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250425235334_UOP_near_crash_R0_C5000_Puranus_O2_Fenergy_FFTrue_DP2'
    # tFind = 2001  # Final time step index
    # cutoffFlag = False  # if true, cut off the data based on the final time step, if false pad the data
    # norm = 142145066.67666337

    # Agressive Near Crash
    # removeFlags = [1]  # 0 capture, 1 escape, 2 crash
    # tag = 'UOP_near_crash_steeper'
    # dataPath = '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250426100446_UOP_near_crash_steeper_R0_C5000_Puranus_O2_Fenergy_FFTrue_DP2'
    # tFind = 867  # Final time step index
    # norm = 142141053.1824186

    # All 3 modes Uniform
    # removeFlags = []  # 0 capture, 1 escape, 2 crash
    # tag = 'UOP_uniform_pGRAM'
    # dataPath = '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250512120942_UOP_uniform_pGRAM_R42_C5000_Puranus_O1_Fenergy_FFTrue_DP2_GMVAEFalse'
    # tFind = 1001  # Final time step index
    # cutoffFlag = False  # if true, cut off the data based on the final time step, if false pad the data
    # norm = 143005107.62401044

    # Combine agressive near crash and near escape
    # removeFlags = []  # 0 capture, 1 escape, 2 crash
    # tag = 'UOP_near_crash_steeper_near_escape_COMBINED'
    # dataPaths = ['/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250425173914_UOP_inc_lit_disps_R0_C5000_Puranus_O1_Fenergy_FFTrue_DP2', '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250426100446_UOP_near_crash_steeper_R0_C5000_Puranus_O2_Fenergy_FFTrue_DP2']
    # tFind = 1001
    # cutoffFlag = False  # if true, cut off the data based on the final time step, if false pad the data
    # norm = 142248639.65469068

    # Polynomial truth
    # removeFlags = []
    # tag = 'UOP_poly_truth'
    # dataPaths = ['/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250513140548_UOP_uniform_poly_R42_C5000_Puranus_O1_Fenergy_FFTrue_DP-1_GMVAEFalse']
    # tFind = 1001
    # cutoffFlag = False  # if true, cut off the data based on the final time step, if false pad the data
    # norm = 142380579.93189436

    # near-crash pGRAM Normal
    # removeFlags = []
    # tag = '1_near_crash_fnpag'
    # dataPaths = [
    #     '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250526170111_1_near_crash_fnpag_R12_C2000_Puranus_O2_Fenergy_FFTrue_DP0_GMVAEFalse']
    # tFind = 1501
    # cutoffFlag = False  # if true, cut off the data based on the final time step, if false pad the data
    # norm = 142193148.77261898

    # near-escape pGRAM Normal
    removeFlags = [2]
    tag = '1_near_escape_fnpag'
    dataPaths = [
        '/Users/gracecalkins/Local_Documents/local_code/pipag/data/20250526202825_1_near_escape_fnpag_R12_C2000_Puranus_O1_Fenergy_FFTrue_DP0_GMVAEFalse']
    tFind = 1501
    cutoffFlag = False  # if true, cut off the data based on the final time step, if false pad the data
    norm = 142193148.77261898


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
    for dataPath in dataPaths:
        for run in trange(Nruns, desc=f"Loading from {dataPath}"):
            # Load in the data
            data = loadmat(f'{dataPath}/run_' + str(run) + '.mat')
            datas.append(data)
            ras.append(data['ra'])

    data_dict = {}
    data_mat = np.zeros((Nruns*len(dataPaths), downsampleNum))
    if flagDownsample:
        idx = np.linspace(0, tFind - 1, downsampleNum, dtype=int)
    else:
        idx = np.linspace(0, tFind - 1, tFind, dtype=int)
    goodInds = []
    save_ind = 0
    labels = []
    for run in range(Nruns*len(dataPaths)):
        data = datas[run]
        if cutoffFlag:
            vel = data['x'][3, :tFind]
            r = data['x'][0, :tFind]
        else:
            # Pad out the velocity and position data with the last value
            if tFind < data['x'].shape[1]:
                vel = data['x'][3, :tFind]
                r = data['x'][0, :tFind]
            else:
                vel = np.pad(data['x'][3, :], (0, tFind - data['x'].shape[1]), 'edge')
                r = np.pad(data['x'][0, :], (0, tFind - data['x'].shape[1]), 'edge')

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

        data_dict[f'sample{save_ind}'] = {dataName: normed_data[idx].tolist(), 'label': label}
        data_mat[save_ind, :] = normed_data[idx]
        save_ind += 1
        labels.append(label)

    suffix = f'{dataName}_{"scaled_" if flagScale else ""}{"downsampled_" if flagDownsample else ""}'
    with open(f'{savePath}/{tag}_{Nruns}_data_{suffix}.json', 'w') as f:
        json.dump(data_dict, f)

    Nruns = len(data_dict)

    # Save datamat as a mat file
    save_path = f'{savePath}/{tag}_{Nruns}_data_{suffix}.mat'
    savemat(save_path, {'data': data_mat})

    # Get 1024 training, 128 validation, and 128 test sample indices including ALL crashes and escapes proportionally distributed between the three sets and filling the rest with the captures
    labels = np.array(labels)
    n_train, n_test, n_val = 1024, 128, 128
    n_total = n_train + n_test + n_val

    # Step 1: Separate indices by label
    capture_idx = np.where(labels == 0)[0]
    escape_idx = np.where(labels == 1)[0]
    crash_idx = np.where(labels == 2)[0]

    # Print the number of captures, escapes, and crashs
    print(f"Number of captures: {len(capture_idx)}")
    print(f"Number of escapes: {len(escape_idx)}")
    print(f"Number of crashes: {len(crash_idx)}")

    # 1024 + 128 + 128 = 1280 total samples
    proportions = [n_train / n_total, n_test / n_total, n_val / n_total]  # [train, val, test]

    if 1 not in removeFlags:
        escape_train, escape_val, escape_test = split_proportional(escape_idx, proportions)
    else:
        escape_train, escape_val, escape_test = np.array([]), np.array([]), np.array([])
    if 2 not in removeFlags:
        crash_train, crash_val, crash_test = split_proportional(crash_idx, proportions)
    else:
        crash_train, crash_val, crash_test = np.array([]), np.array([]), np.array([])

    # Step 3: Compute how many capture samples are needed to fill each set
    train_needed = n_train - len(escape_train) - len(crash_train)
    val_needed = n_test - len(escape_val) - len(crash_val)
    test_needed = n_val - len(escape_test) - len(crash_test)

    np.random.shuffle(capture_idx)
    capture_train = capture_idx[:train_needed]
    capture_val = capture_idx[train_needed:train_needed + val_needed]
    capture_test = capture_idx[train_needed + val_needed:train_needed + val_needed + test_needed]

    # Step 4: Combine and shuffle
    train_indices = np.concatenate([escape_train, crash_train, capture_train])
    val_indices = np.concatenate([escape_val, crash_val, capture_val])
    test_indices = np.concatenate([escape_test, crash_test, capture_test])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Results
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Put the train, val, and test indices into a single sequential list and save it as a json
    all_inds = np.concatenate([train_indices, val_indices, test_indices])
    output = {'sample_list': all_inds.tolist()}

    with open(f'{savePath}/{tag}_{Nruns}_inds_{suffix}.json', "w") as f:
        json.dump(output, f, indent=2)

    # Plot the data
    sns.set_theme('notebook', style='whitegrid', palette='Paired', rc={"lines.linewidth": 2.5, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, "font.family": "serif"})
    fig, ax = plt.subplots()
    for ii, run in enumerate(goodInds):
        label = data_dict[f'sample{ii}']['label']
        color = f"C{label}"
        ax.plot(data_mat[ii, :], color=color)
    line1 = plt.Line2D([0], [0], color='C0', label='Capture')
    line2 = plt.Line2D([0], [0], color='C1', label='Escape')
    line3 = plt.Line2D([0], [0], color='C2', label='Crash')
    ax.legend(handles=[line1, line2, line3])
    plt.ylabel(f'{r"Velocity (m/s)" if not flagEnergy else r"Energy (m$^2$/s$^2$)"}')
    plt.xlabel('Time step')
    if flagEnergy:
        plt.hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
    plt.savefig(f'/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs/{tag}_{Nruns}_data_{suffix}.png', dpi=300)


    # Plot the training, validation, and testing data in three subplots
    fig, axs = plt.subplots(1,3, figsize=(8,5), sharey=True)

    for ii, run in enumerate(train_indices):
        label = data_dict[f'sample{ii}']['label']
        color = f"C{label}"
        axs[0].plot(data_mat[ii, :], color=color)
    axs[0].set_title('Training Data')

    for ii, run in enumerate(val_indices):
        label = data_dict[f'sample{ii}']['label']
        color = f"C{label}"
        axs[1].plot(data_mat[ii, :], color=color)
    axs[1].set_title('Validation Data')
    axs[1].set_yticklabels([])

    for ii, run in enumerate(test_indices):
        label = data_dict[f'sample{ii}']['label']
        color = f"C{label}"
        axs[2].plot(data_mat[ii, :], color=color)
    axs[2].set_title('Testing Data')
    axs[2].set_yticklabels([])
    axs[0].set_ylabel(f'{r"Velocity (m/s)" if not flagEnergy else r"Energy (m$^2$/s$^2$)"}')
    axs[0].set_xlabel('Time step')
    axs[1].set_xlabel('Time step')
    axs[2].set_xlabel('Time step')
    if flagEnergy:
        axs[0].hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
        axs[1].hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
        axs[2].hlines(0, 0, downsampleNum, colors='r', linestyles='dashed')
    axs[0].legend(handles=[line1, line2, line3])
    plt.tight_layout()
    plt.savefig(f'/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs/{tag}_{Nruns}_data_{suffix}_train_val_test.png', dpi=300)

    # Get maximum initial energy
    max_energy = 0
    mean_energy_0 = 0
    for run in range(Nruns):
        energy = data_mat[run, 0]
        if energy > max_energy:
            max_energy = energy
        mean_energy_0 += energy
    mean_energy_0 /= Nruns
    print(f'Maximum initial energy: {max_energy}')
    print(f'Mean initial energy: {mean_energy_0}')



if __name__ == '__main__':
    main()
    plt.show()
