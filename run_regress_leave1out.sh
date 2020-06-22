#!/bin/bash

echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"

# untar your Python installation
tar -xzf python.tar.gz
# make sure the script will use your Python installation,
# and the working directory as it's home location
export PATH=$(pwd)/python/bin:$PATH
mkdir home
export HOME=$(pwd)/home

index=$1
index_pickle=$2
nDeep=$3
folder_name=$4/
server=$5
norm_method=$6
img_rows=$7
img_cols=$8

tar_file=$4.tar.gz



if [ $server == gluster ]
then
echo /mnt/gluster/kelkar2/${tar_file}
cp /mnt/gluster/kelkar2/${tar_file} .
elif [ $server == staging ]
then
echo /staging/kelkar2/${tar_file}
cp /staging/kelkar2/${tar_file} .
fi

tar -xzf ${tar_file}

#input_folder=CNN_Input_${label}/

#output_folder=RegressionOutput_${label}/

#output_folder=""

echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"

echo $PWD

python Lenet5_keras.py ${index} ${index_pickle} ${nDeep} ${folder_name} ${norm_method} ${img_rows} ${img_cols} #&> log${1}.out

rm log${1}.out

rm ${tar_file}


