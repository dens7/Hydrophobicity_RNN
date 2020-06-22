#!/bin/bash
source ~/.bashrc
basedir=$PWD

ls -d */ &> folders.txt

while read folder_name; do

 cd $folder_name
 #cd log 
 ls -1 | wc -l
 echo $folder_name
 #rm *
 cd $basedir

done < folders.txt
