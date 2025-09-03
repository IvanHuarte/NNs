from netket.operator.spin import sigmaz
import netket.experimental as nkx
import numpy as np

# Vstate calculations

def renyi_vs(vstate):
    """
    Renyi's entropy
    """
    N = vstate.hilbert.size
    renyi = nkx.observable.Renyi2EntanglementEntropy(
        vstate.hilbert, np.arange(0, N / 2 + 1, dtype=int)
    )   
    return vstate.expect(renyi).mean
    
def M_vs(vstate):
    """
    First magnetization momentum (Ferro)
    """
    N = vstate.hilbert.size
    magn = sum([sigmaz(vstate.hilbert, i) / N for i in range(N)])
    return vstate.expect(magn).mean.real

def M2_vs(vstate):
    """
    Second magnetization momentum (Ferro) 
    """
    N = vstate.hilbert.size
    magn2 = sum([sigmaz(vstate.hilbert, i) / N for i in range(N)])
    return vstate.expect(magn2 @ magn2).mean.real

def Ms_vs(vstate):
    """
    First staggered magnetization momentum (AntiFerro)
    """
    N = vstate.hilbert.size
    magn_s = sum(
        [(-1) ** i * sigmaz(vstate.hilbert, i) / N for i in range(N)]
    )
    return vstate.expect(magn_s).mean.real

def Ms2_vs(vstate):
    """
    Second staggered magnetization momentum (AntiFerro)
    """
    N = vstate.hilbert.size
    magn_s2 = sum(
        [(-1) ** i * sigmaz(vstate.hilbert, i) / N for i in range(N)]
    )
    return vstate.expect(magn_s2 @ magn_s2).mean.real

def calc_all_observables_vs(vstate):

    S_renyi = renyi_vs(vstate)
    m = M_vs(vstate)
    ms = Ms_vs(vstate)
    m2 = M2_vs(vstate)
    ms2 = Ms2_vs(vstate)

    return float(S_renyi), float(m), float(ms), float(m2), float(ms2)

# Exact Diagonalization calculations

def compute_spin_matrices(N):
    configs = np.arange(2**N)[:, None]          
    bits = ((configs >> np.arange(N)) & 1)
    sigma_z = 0.5 - bits                              
    return sigma_z  

def M_ED(state):
    """<M> magnetización media"""
    N = int(np.log2(len(state)))
    sigma_z = 2 * compute_spin_matrices(N)
    probs = np.abs(state)**2
    M_exp = np.sum(probs[:, None] * sigma_z, axis=0).mean()  
    return float(M_exp)

def M2_ED(state):
    """<M^2> segundo momento de la magnetización uniforme"""
    N = int(np.log2(len(state)))
    sigma_z = 2 * compute_spin_matrices(N)
    probs = np.abs(state)**2

    # correladores S_i S_j
    corr = (probs[:, None, None] * (sigma_z[:, :, None] * sigma_z[:, None, :])).sum(axis=0)
    M2_exp = corr.sum() / (N**2)
    return float(M2_exp)

def Ms_ED(state):
    """<Ms> magnetización staggered media por sitio"""
    N = int(np.log2(len(state)))
    sigma_z = 2 * compute_spin_matrices(N)
    staggered = (-1)**np.arange(N)
    probs = np.abs(state)**2
    Ms_exp = np.sum(probs[:, None] * (sigma_z * staggered), axis=0).mean()
    return float(Ms_exp)

def Ms2_ED(state):
    """<Ms^2> segundo momento de la magnetización staggered"""
    N = int(np.log2(len(state)))
    sigma_z = 2 * compute_spin_matrices(N)
    probs = np.abs(state)**2
    staggered = (-1)**np.arange(N)

    sigma_z_stag = sigma_z * staggered  # shape (2^N, N)
    corr = (probs[:, None, None] * (sigma_z_stag[:, :, None] * sigma_z_stag[:, None, :])).sum(axis=0)
    Ms2_exp = corr.sum() / (N**2)
    return float(Ms2_exp)

def calc_all_observables_ED(state):

    N = int(np.log2(len(state)))
    staggered = (-1)**np.arange(N)

    sigma_z = 2 * compute_spin_matrices(N)
    probs = np.abs(state)**2

    m = np.sum(probs[:, None] * sigma_z, axis=0).mean()
    ms = np.sum(probs[:, None] * (sigma_z * staggered), axis=0).mean()

    corr = (probs[:, None, None] * (sigma_z[:, :, None] * sigma_z[:, None, :])).sum(axis=0)
    m2 = corr.sum() / (N**2)
    
    sigma_z_stag = sigma_z * staggered
    corr = (probs[:, None, None] * (sigma_z_stag[:, :, None] * sigma_z_stag[:, None, :])).sum(axis=0)
    ms2 = corr.sum() / (N**2)

    return float(m), float(ms), float(m2), float(ms2)
