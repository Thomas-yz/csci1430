#!/bin/bash
#
#  Execute from the current working directory
#$ -cwd
#
# This requests a single GPU. Unless you write
# Your code in a very specific way, it will not
# benefit from additional GPUs.
#$ -l gpus=1

# This loads the virtual environment
source /course/cs1430/tf_gpu/bin/activate
FILE=$1
# Print out the python file so we have a record. 
# Useful when iterating a file so you can keep track of what 
# You are doing
# uncomment this next line to print the python file to stdout
# cat $FILE
# Shift increments the argv values.
shift
# Tensorpack specific setting to cache ZMQ pipes in /ltmp/ 
# for speed boost
export TENSORPACK_PIPEDIR=/ltmp/
# If you need to compile any CUDA kernels do it on the local FS 
# so it happens faster
export CUDA_CACHE_PATH=/ltmp/
# Runs the python file passing all args and pipes n into the file. 
# -u tells python to not buffer the output to so it is printed
# more often.
echo -e "n\n" | python -u $FILE $@

#Reminder
# qhost to look at machines and the current usage
# qstat to view your running jobs
# qstat -u '*' to view everyone's running jobs
# qsub to run the job
