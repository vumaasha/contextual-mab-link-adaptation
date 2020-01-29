import itpp
import numpy as np

''' Find the SINR for the given CQI to approximately achieve the given BLER target
'''
def estimate_sinr_from_cqi(awgn_data, cqi):

    REF_BLER_TARGET = 0.1

    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler = awgn_data['snr_vs_bler']

    _, nrof_cqi = awgn_snr_vs_bler.shape
    
    bler = awgn_snr_vs_bler[:, cqi]

    if cqi == 0:
        return np.min(awgn_snr_range_dB)
    elif cqi == nrof_cqi - 1:
        return np.max(awgn_snr_range_dB)

    index1 = np.max(np.argwhere(REF_BLER_TARGET < bler))
    index2 = np.min(np.argwhere(REF_BLER_TARGET > bler))
    
    estimated_sinr_dB = (awgn_snr_range_dB[index1] + awgn_snr_range_dB[index2]) / 2.0

    return estimated_sinr_dB

def determine_mcs_from_sinr(awgn_data, sinr_dB, bler_target):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler = awgn_data['snr_vs_bler']

    _, nrof_cqi = awgn_snr_vs_bler.shape

    tbs = [ 20, 20, 40, 64, 84, 104, 124, 148, 168, 148, 188, 232, 272, 316, 356, 400, 408, 472, 536, 600, 660, 724 ]
    
    bler_at_snr = determine_bler_at_sinr(awgn_data, sinr_dB)

    # Find the largest MCS that has BLER less than the BLER target
    # The CQIs are evaluated in decreasing order and first value that predicts a BLER < 0.1
    # is returned.
    largest_mcs = 0
    for i in range(nrof_cqi):
        current_mcs = nrof_cqi - i - 1
        if bler_at_snr[current_mcs] < bler_target:
            largest_mcs = current_mcs
            break 
        else:
            continue

    # Determine the expected tput for all valid MCSs
    best_mcs = 0
    best_expected_tput = 0
    for i in range( largest_mcs ):
        expected_tput = ( 1 - bler_at_snr[ i ] ) * float( tbs[ i ] )
        if expected_tput > best_expected_tput:
            best_expected_tput = expected_tput
            best_mcs = i
    
    return best_mcs

def determine_bler_at_sinr(awgn_data, sinr_dB):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler = awgn_data['snr_vs_bler']

    _, nrof_cqi = awgn_snr_vs_bler.shape
    
    bler_at_sinr = itpp.vec(nrof_cqi)

    for i in range(nrof_cqi):
        bler = awgn_snr_vs_bler[:, i]
        if sinr_dB <= np.min(awgn_snr_range_dB):
            bler_at_sinr[i] = 1.0
        elif sinr_dB >= np.max(awgn_snr_range_dB):
            bler_at_sinr[i] = 0.0
        else:
            index1 = np.max(np.argwhere(awgn_snr_range_dB < sinr_dB))
            index2 = np.min(np.argwhere(awgn_snr_range_dB > sinr_dB))

            bler_at_sinr[i] = (bler[index1] + bler[index2]) / 2.0

    return bler_at_sinr

