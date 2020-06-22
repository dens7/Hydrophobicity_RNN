#!/bin/bash

jobName=CNN_Trial

data_rep=$1 # voxelCount or gauss or CDsmear
nDeep=$2 # Depth of representation
surfaces=$3 # real or ideal
size=$4 # e.g. 2.0x2.0x0.3nm
pixels=$5 # e.g. 20x20 or 20x20_smear0.5
CNStat=$6 # e.g. All or ext
rep=$7 # avg or txf 
img_rows=$8 # Number of rows
img_cols=$9 # Number of columns
server=${10} # e.g. gluster or staging
ligand_set=${11} # e.g. amide or amide or hydrox

echo $data_rep
dataset=${ligand_set}_${data_rep}_t${nDeep}f_${size}_${pixels}_${CNStat}

test_train_folder=RNN_Input_${surfaces}_${dataset}
init_dir=$PWD/${test_train_folder}

executable=run_regress_leave1out.sh
python_code=RNN_keras.py
nSamples=5 # Number of jobs to be started in the queue
pickle_name="5_fold_indices_40.pickle"
norm_method=frame

req_memory=15 # GB
req_disk=15 # GB

log_folder=log
err_folder=err
output_folder=output

mkdir -p ${test_train_folder}

mkdir -p ${test_train_folder}/${log_folder}
mkdir -p ${test_train_folder}/${err_folder}
mkdir -p ${test_train_folder}/${output_folder}

cp ${python_code} ${test_train_folder}/
cp ${pickle_name} ${test_train_folder}/

if [[ $server == "gluster" || $server == "staging" ]]; then
	echo $server
	cp submit_template_${server}.sub ${jobName}.sub
	temp=test_train_folder
else
	cp submit_template.sub ${jobName}.sub
fi

sed -i s/OUTPUT_FOLDER/${output_folder}/g ${jobName}.sub
sed -i s/LOG_FOLDER/${log_folder}/g ${jobName}.sub
sed -i s/ERR_FOLDER/${err_folder}/g ${jobName}.sub
sed -i s/TEST_TRAIN_DATA/${test_train_folder}/g ${jobName}.sub
sed -i s/NDEEP/${nDeep}/g ${jobName}.sub
sed -i s/EXECUTABLE/${executable}/g ${jobName}.sub
sed -i s/INPUT_IMAGES_FOLDER/${test_train_folder}/g ${jobName}.sub
sed -i s/NSAMPLES/${nSamples}/g ${jobName}.sub
sed -i s/REQ_MEMORY/${req_memory}/g ${jobName}.sub
sed -i s/REQ_DISK/${req_disk}/g ${jobName}.sub
sed -i s/SERVER/${server}/g ${jobName}.sub
sed -i s#INIT_DIR#${init_dir}#g ${jobName}.sub
sed -i s/PICKLE_NAME/${pickle_name}/g ${jobName}.sub
sed -i s/PYTHON_CODE/${python_code}/g ${jobName}.sub
sed -i s/NORM_METHOD/${norm_method}/g ${jobName}.sub
sed -i s/IMG_ROWS/${img_rows}/g ${jobName}.sub
sed -i s/IMG_COLS/${img_cols}/g ${jobName}.sub

sed -i s/PYTHON_CODE/${python_code}/g ${executable}

condor_submit ${jobName}.sub -batch-name ${dataset} 
