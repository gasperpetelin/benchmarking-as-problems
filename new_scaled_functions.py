from utils import *
import os.path
from scipy.stats import qmc
from tqdm import tqdm
from pymoo.operators.sampling.lhs import LHS
import polars as pl
import multiprocessing as mp
import random

if __name__ == "__main__":
    save_dir = 'scale_problem'
    create_directory_if_not_exist(save_dir)
    
    problem_dim = 5
    n_runs = 100
    n_evals = 1000
    seed=42
    n_instance_reapets = 100
    
    suite = list(generate_scaled_problems(problem_dim=problem_dim, seed=seed, n_runs=n_instance_reapets))
    random.shuffle(suite)
    
    runs_dir = f'{save_dir}/runs'
    create_directory_if_not_exist(runs_dir)
    
    
    for problem in tqdm(suite):
        file_path = f'{runs_dir}/p_{problem.id_function}__i_{problem.id_instance}__d_{problem_dim}__fe_{n_evals}__scale_{problem.scale}.parquet'

        if os.path.exists(file_path):
            continue

        opt = problem.evaluate(problem.pareto_set())[0, 0]
        pdf = pl.DataFrame(run_algorithms(problem, n_runs=n_runs, n_eval=n_evals)).with_columns([
            pl.lit(problem.id_function).alias("problem"),
            pl.lit(problem.id_instance).alias("instance"),
            pl.lit(problem.scale).alias("scale"),
            pl.lit(opt).alias("optimum"),
        ])
        pdf.write_parquet(file_path)