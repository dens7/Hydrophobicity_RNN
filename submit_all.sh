#!/bin/bash

#sleep 20m

rep_arr=('txf')
density_arr=('voxelCount')
CNStat=All
size='2.0x2.0x0.3nm'
pixels='25x25'
server=staging
ligand_set=all

for density in ${density_arr[@]}; do
	for rep in ${rep_arr[@]}; do

		bash submit_submit_chtc.sh ${density} ${rep} ${CNStat} ${size} ${pixels} ${server} ${ligand_set}

	done
done
