#python ~/tmp/coco/do.py run-python
from pymoo.problems import get_problem
from tqdm import tqdm
import pandas as pd
from utils import *
from pymoo.vendor.vendor_coco import COCOProblem
import random
import os

coco_instance=1
dim=5
n_samples=500


from threading import Lock

# Define a lock
problem_creation_lock = Lock()

def get_problem(typ, n_var, scale):
    with problem_creation_lock:  # Acquire the lock before creating the problem
        return ScaledCOCOProblem(typ, n_var=n_var, scale=scale)
    
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

for n_evals in [100, 200, 500, 1000]:
    for coco_problem in range(1, 25):
        print(n_evals, coco_problem)
        file = f'scale_correlations/pc_{coco_problem}__pi_{coco_instance}__nevals_{n_evals}.csv'
        if os.path.exists(file):
            print(f'Files {file} already exist.')
        else:
            df = {'problem_class': [], 'problem_instance': [], 'y_diff': [], 'ga_diff': [], 'cmaes_diff': [], 'n_evals': []}
            for run in tqdm(range(1, 1000)):
                scaled = get_scale()
                problem = get_problem(f"bbob-f{coco_problem}-{coco_instance}", n_var=dim, scale=scaled)
                data = run_algorithms(problem, n_runs=1, algorithms=['GA', 'CMAES'], n_eval=n_evals)
                sampling = LHS()
                X = sampling(problem, dim*n_samples).get("X")
                y = problem.evaluate(X).flatten()

                with problem_creation_lock:
                    y_diff = np.abs(np.min(y)-np.max(y))
                    ga_diff = data['GA'][0]-problem.evaluate(problem.pareto_set())[0][0]
                    cmaes_diff = data['CMAES'][0]-problem.evaluate(problem.pareto_set())[0][0]

                df['problem_class'].append(coco_problem)
                df['problem_instance'].append(coco_instance)
                df['y_diff'].append(y_diff)
                df['ga_diff'].append(ga_diff)
                df['cmaes_diff'].append(cmaes_diff)
                df['n_evals'].append(n_evals)
            df = pd.DataFrame(df)
            df.to_csv(file, index=False)