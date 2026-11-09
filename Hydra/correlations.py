from netket.operator.spin import sigmax, sigmay, sigmaz
from netket.hilbert import Spin
import numpy as np

def correlations_ED(size, phi):
    N = int(np.prod(size))
    hi = Spin(s=0.5, N=N)
    i_idx, j_idx = np.triu_indices(N)

    SS=[]

    for i,j in zip(i_idx,j_idx):

        XX = (phi.conj().T @ ((sigmax(hi, i)*sigmax(hi,j))@ phi)).real
        YY = (phi.conj().T @ ((sigmay(hi, i)*sigmay(hi,j))@ phi)).real
        ZZ = (phi.conj().T @ ((sigmaz(hi, i)*sigmaz(hi,j))@ phi)).real
        SS.append(float(XX+YY+ZZ))

    return np.array(SS)



def correlations_vstate(vstate):

    N = vstate.hilbert.size
    hi = vstate.hilbert
    i_idx, j_idx = np.triu_indices(N)

    SS=[]

    for i,j in zip(i_idx,j_idx):

        XX = vstate.expect(sigmax(hi, i)*sigmax(hi,j)).mean.real
        YY = vstate.expect(sigmay(hi, i)*sigmay(hi,j)).mean.real
        ZZ = vstate.expect(sigmaz(hi, i)*sigmaz(hi,j)).mean.real
        SS.append(float(XX+YY+ZZ))


    return np.array(SS)
