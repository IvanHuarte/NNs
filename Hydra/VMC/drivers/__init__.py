from .netket import VMC_netket, VMC_SR_netket

REGISTRY_VMC = {
    "vmc_netket": VMC_netket,
    "vmc_sr_netket": VMC_SR_netket,
}
