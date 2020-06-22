#!/bin/bash


while read num; do

	condor_rm $num

done < trial.sh
