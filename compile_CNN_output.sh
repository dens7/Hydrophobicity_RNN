#!/bin/bash

basedir=$PWD

density_reps=('gauss' 'voxelCount' 'CDsmear')
density_reps=('voxelCount')
data_reps=('txf' 'avg')
#data_reps=('txf')
nStacks=(1 2 3 4 6 10 15 20 25 30)
#nStacks=(4 10)
CNStat=All
dataset=ideal
ligand_set=all
size=3.0x3.0x0.3nm
pixels=20x20 #_smear0.5

for density_rep in ${density_reps[@]}; do
	for data_rep in ${data_reps[@]}; do
		for nStack in ${nStacks[@]}; do

			if [[ $data_rep == 'txf' ]]
			then
				label=${dataset}_${ligand_set}_${density_rep}_t${nStack}f_${size}_${pixels}_${CNStat}
			elif [[ $data_rep == 'avg' ]]
			then
				label=${dataset}_${ligand_set}_${density_rep}_avg${nStack}_${size}_${pixels}_${CNStat}
			elif [[ $data_rep == 'AvgVar' ]]
			then
				label=${dataset}_${ligand_set}_${density_rep}_AvgVar_${nStack}_${size}_${pixels}_${CNStat}
			fi
			CNN_name=${dataset}_${CNStat}/CNN_Input_${label}
			folder_name=RegressionOutput_${label}

			#echo $CNN_name
			#echo $folder_name

			cd ${CNN_name}
			pwd
			mkdir ${folder_name}

			mv regress* ${folder_name}/
			mv model* ${folder_name}/

			tar -zcf ${folder_name}.tar.gz ${folder_name}/
			cd ${basedir}
			pwd
		done
	done
done
