python ~/tmp/coco/do.py run-python

tsp -S 1

for PROBLEM in {1..24}; do
    for N_EVALS in 100 500 1000; do
        tsp python run_algorithms.py --problem $PROBLEM --instance 1 --dim 10 --n_eval $N_EVALS
    done    
done