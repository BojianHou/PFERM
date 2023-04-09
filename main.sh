#!/bin/sh

source activate FERM
main_dir=/cbica/home/houbo/Projects/fair_ERM
cd $main_dir
python -u main.py >${main_dir}/log/logfile1.log 2>&1