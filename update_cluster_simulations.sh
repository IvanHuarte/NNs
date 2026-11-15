#!/bin/bash

# GARMONEGRO UPDATE
# pile
rsync -av garmonegro:/home/ihuarte/Escritorio/Ivan/NNs/pile/* /home/ihuarte/Escritorio/Ivan/NNs/pile/garmonegro/
# Simulations folder
rsync -av garmonegro:/home/ihuarte/Escritorio/Ivan/NNs/Simulations_claude_cluster/ /home/ihuarte/Escritorio/Ivan/NNs/Simulations_claude_garmonegro/

wait 

# CESGA UPDATE
# pile
rsync -av cesga:/home/csic/atm/ihf/Escritorio/Ivan/NNs/pile/* ~/Escritorio/Ivan/NNs/pile/cesga/
# Simulations folder
rsync -av cesga:/mnt/netapp1/Store_CSIC//home/csic/atm/ihf/Escritorio/Ivan/NNs/Simulations_claude_cluster/ ~/Escritorio/Ivan/NNs/Simulations_claude_cesga/

wait

exit 0

