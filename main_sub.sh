#!/bin/sh


echo -e "\nRunning commands on          : `hostname`"
echo -e "Start time                     : `date +%F-%H:%M:%S`\n"
main_dir=/cbica/home/houbo/Projects/fair_ERM

for dataset in toy_new
do
    echo "Processing $dataset ..."

    for constraint in EO DP
    do
        echo "Constraint is $constraint ..."
        for lamda in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            echo "lamda is $lamda ..."
            jid=$(qsub \
                  -terse \
                  -l h_vmem=40G \
                  -pe threaded 2-4 \
                  -o ${main_dir}/sge_output/\$JOB_NAME-\$JOB_ID.stdout \
                  -e ${main_dir}/sge_output/\$JOB_NAME-\$JOB_ID.stderr \
                  ${main_dir}/main.sh -d $dataset -m $main_dir \
                  -c $constraint -l $lamda)
        done
    done
done