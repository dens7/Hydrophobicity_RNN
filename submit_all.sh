#!/bin/bash

CNStat='All'
size='2.0x2.0x0.3nm'
pixels='25x25' #_smear0.5
img_rows="$(echo $pixels | cut -d"x" -f1)"
img_cols="$(echo $pixels | cut -d"x" -f2)"
img_cols="$(echo "$img_cols" | cut -d"_" -f1)"
stack=(30)
surfaces=ideal
ligand_set=all

for i in ${stack[@]}; do
	bash submit_chtc.sh ${i} ${surfaces} ${size} ${pixels} ${CNStat} "${img_rows}" "${img_cols}" ${ligand_set}
done

