#!/bin/bash

density_rep=$1
rep=$2 # avg or txf
CNStat=$3
size=$4
pixels=$5 #_smear0.5
img_rows="$(echo $pixels | cut -d"x" -f1)"
img_cols="$(echo $pixels | cut -d"x" -f2)"
img_cols="$(echo $img_cols | cut -d"_" -f1)"
server=$6
stack=(1 2 3 4 6 10 15 20 25 30)
surfaces=ideal
ligand_set=$7

for i in ${stack[@]}; do
	#echo $server
	bash submit_chtc.sh ${density_rep} ${i} ${surfaces} ${size} ${pixels} ${CNStat} ${rep} ${img_rows} ${img_cols} ${server} ${ligand_set}
done

