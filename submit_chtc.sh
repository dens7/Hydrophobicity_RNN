#!/bin/bash

jobName=RNN_Model


nDeep=$1 # Depth of representation
surfaces=$2 # real or ideal
size=$3 # e.g. 2.0x2.0x0.3nm
pixels=$4 # e.g. 20x20 or 20x20_smear0.5
CNStat=$5 # e.g. All or ext
img_rows=$6 # Number of rows
img_cols=$7 # Number of columns
ligand_set=$8 # e.g. amide or amide or hydrox

dataset=${ligand_set}_voxelCount_t${nDeep}f_${size}_${pixels}_${CNStat}

test_train_folder=RNN_Input_${surfaces}_${dataset}
init_dir=$PWD/${test_train_folder}

executable=run_regress_leave1out.sh
python_code=RNN_Keras.py
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

cp submit_template_staging.sub ${jobName}.sub
temp=test_train_folder

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
sed -i s#INIT_DIR#${init_dir}#g ${jobName}.sub
sed -i s/PICKLE_NAME/${pickle_name}/g ${jobName}.sub
sed -i s/PYTHON_CODE/${python_code}/g ${jobName}.sub
sed -i s/NORM_METHOD/${norm_method}/g ${jobName}.sub
sed -i s/IMG_ROWS/${img_rows}/g ${jobName}.sub
sed -i s/IMG_COLS/${img_cols}/g ${jobName}.sub

sed -i s/PYTHON_CODE/${python_code}/g ${executable}

condor_submit ${jobName}.sub -batch-name ${dataset} 
