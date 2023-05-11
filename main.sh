#!/bin/sh

parse()
{
	while [ -n "$1" ];
	do
		case $1 in
            -d)
              dataset=$2;
              shift 2;;
            -m)
              main_dir=$2;
              shift 2;;
            -c)
              constraint=$2;
              shift 2;;
            -l)
              lamda=$2;
              shift 2;;
		esac
	done
}

if [ $# -lt 1 ]
then
	help
fi

## Reading arguments
parse $*

source activate FERM
# main_dir=/cbica/home/houbo/Projects/fair_ERM
cd $main_dir

echo "constraint is $constraint"
python -u main.py --constraint $constraint --lamda $lamda \
>${main_dir}/log/logfile_constraint_${constraint}_lamda_${lamda}.log 2>&1