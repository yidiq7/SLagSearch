#!/bin/bash
# Submit a chain of 10 Slurm jobs where each job depends on the successful completion of the previous one.
# This allows the GA to keep evolving for 400 generations automatically.

NUM_JOBS=10
SCRIPT="GA.script"

echo "Submitting a chain of $NUM_JOBS jobs..."

# Submit the first job and get its Job ID using the --parsable flag
JOB_ID=$(sbatch --parsable $SCRIPT)
echo "Submitted job 1 (Job ID: $JOB_ID)"

# Loop to submit the remaining jobs, each dependent on the previous one
for i in $(seq 2 $NUM_JOBS); do
    JOB_ID=$(sbatch --parsable --dependency=afterany:$JOB_ID $SCRIPT)
    echo "Submitted job $i (Job ID: $JOB_ID)"
done

echo "All $NUM_JOBS jobs successfully submitted to the Slurm queue!"
