import argparse
import os
from utils import *
from pymoo.problems import get_problem
from tqdm import tqdm
import pandas as pd
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int)
    parser.add_argument("--instance", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--n_eval", type=int)
    args = parser.parse_args()
    n_samples = 100
    save_dir = "runs"
    create_directory_if_not_exist(save_dir)
    file = f'{save_dir}/p_{args.problem}__i_{args.instance}__nevals_{args.n_eval}.csv'
    
    if os.path.isfile(file) == False:
        dfs = []
        for run in range(1, 1000):
            scaled = get_scale()
            problem = ScaledCOCOProblem(f"bbob-f{args.problem}-{args.instance}", n_var=args.dim, scale=scaled)
            data = run_algorithms(problem, n_runs=1, n_eval=args.n_eval)
            pdf = pd.DataFrame(data)
            sampling = LHS()
            X = sampling(problem, args.dim*n_samples).get("X")
            y = problem.evaluate(X).flatten()

            pdf['problem'] = args.problem
            pdf['instance'] = args.instance
            pdf['dim'] = args.dim
            pdf['n_eval'] = args.n_eval
            pdf['optimum'] = problem.evaluate(problem.pareto_set())
            pdf['sample_y_min'] = y.min()
            pdf['sample_y_max'] = y.max()
            dfs.append(pdf)
            
        dfs = pd.concat(dfs)
        dfs.to_csv(file, index=False, float_format='%.15f')
    else:
        print('Skip file', file)

