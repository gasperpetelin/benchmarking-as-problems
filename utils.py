import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.core.callback import Callback
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from pymoo.vendor.vendor_coco import COCOProblem
from pathlib import Path
import polars as pl

def get_ranks(pdf, algorithms=["GA", "PSO", "DE", "CMAES", "ES"]):
    n = pdf.select(algorithms).to_pandas().rank(axis=1)
    n['algorithm_run'] = pdf['algorithm_run']
    n['problem'] = pdf['problem']
    n['instance'] = pdf['instance']
    return pl.DataFrame(n).sort(['problem','instance'])

def get_scale(seed=None, num_values=1):
    log_lower = np.log(0.0001)
    log_upper = np.log(1000)

    if seed is not None:
        np.random.seed(seed)

    log_samples = np.random.uniform(log_lower, log_upper, size=num_values)
    linear_samples = np.exp(log_samples)
    if num_values == 1:
        return linear_samples[0]
    return linear_samples

def get_suit_and_scalars(problem_dim, max_instance):
    suite = generate_problems(problem_dim, max_instance)
    scalars = get_scale(42, len(suite))
    return suite, scalars


def create_directory_if_not_exist(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def generate_problems(problem_dim):
    from pymoo.vendor.vendor_coco import COCOProblem
    for instance_id in range(1, 16):
        for problem_id in range(1,25):
            problem = COCOProblem(f"bbob-f{problem_id}-{instance_id}", n_var=problem_dim)
            problem.id_function=problem_id
            problem.id_instance=instance_id
            yield problem
        
class ScaledCOCOProblem(COCOProblem):
    def __init__(self, ptype, n_var, scale=1.0):
        super().__init__(ptype, n_var)
        self.scale=scale
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] = self.scale*out["F"]
        
def generate_scaled_problems(problem_dim, seed, n_runs=3):
    import itertools
    from pymoo.vendor.vendor_coco import COCOProblem
    runs = range(n_runs)
    instance_ids = range(1, 16)
    problem_ids = range(1,25)
    cross_product = list(itertools.product(runs, instance_ids, problem_ids))
    
    scales = get_scale(seed=seed, num_values=len(cross_product))
    for scale, (run, instance_id, problem_id) in zip(scales, cross_product):
        problem = ScaledCOCOProblem(f"bbob-f{problem_id}-{instance_id}", n_var=problem_dim, scale=scale)
        problem.id_function=problem_id
        problem.id_instance=instance_id
        yield problem
        
        #print(run, instance_id, problem_id, scale)
    #for instance_id in range(1, 16):
    #    for problem_id in range(1,25):
    #        problem = COCOProblem(f"bbob-f{problem_id}-{instance_id}", n_var=problem_dim)
    #        problem.id_function=problem_id
    #        problem.id_instance=instance_id
    #        yield problem

def generate_problems_2(dim, max_instance=999):
    import cocoex
    suite_year_option = f"instances: 1-{max_instance}"
    #suite_year_option = "instances: 1-500"
    suite_filter_options = (""  # without filtering, a suite has instance_indices 1-15
                             f"dimensions: {dim}"

                           )
    suite = cocoex.Suite('bbob', suite_year_option, suite_filter_options)
    return suite


def try_load_file(file):
    try:
        return pd.read_csv(file, index_col=False)
    except Exception as e:
        print(e)
        


def load_runs():
    files = list(glob.glob('runs/*.csv'))
    parquet_file = f'runs/data_{len(files)}.parquet'
    if os.path.exists(parquet_file):
        return pd.read_parquet(parquet_file)
    dfs = Parallel(n_jobs=-1)(delayed(try_load_file)(file=file) for file in files)
    df = pd.concat(dfs)
    df.to_parquet(parquet_file)
    return df

def load_ela():
    files = glob.glob('ela/*.csv')
    parquet_file = f'ela/data_{len(files)}.parquet'
    if os.path.exists(parquet_file):
        return pd.read_parquet(parquet_file)
    dfs = Parallel(n_jobs=-1)(delayed(try_load_file)(file=file) for file in files)
    ela = pd.concat(dfs)
    ela.to_parquet(parquet_file)
    return ela

def get_rank(df, algorithms):
    d = df.copy()
    rdf = df[algorithms].rank(axis=1).copy()
    for column in df.columns.difference(algorithms):
        rdf[column]=df[column].copy()
    return rdf.copy()

def log_performances(df, algorithms):
    ndf = df.copy()
    for algo in algorithms:
        ndf[algo] = np.log(df[algo])
    return ndf

def generate_2d_plot_matrix(problem, n=500):
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, y)
    Z = problem.evaluate(np.vstack([X.ravel(), Y.ravel()]).T).reshape(n, n)
    return Z

def plot(problem, n=500):
    Z = generate_2d_plot_matrix(problem, n=n)
    plt.imshow(Z, cmap='viridis', extent=[-5, 5, -5, 5], origin='lower')
    plt.colorbar()
    #plt.show()
    return Z


def create_directory_if_not_exist(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


class CombinedProblem(Problem):
    def __init__(self, problem1, problem2, alpha):
        assert problem1.n_var==problem1.n_var
        super().__init__(n_var=problem1.n_var, n_obj=1, n_constr=0, xl=-5, xu=5)
        self.problem1 = problem1
        self.problem2 = problem2
        self.alpha = alpha
        self.o1 = problem1.pareto_set()
        self.o2 = problem2.pareto_set()

        
    def _evaluate(self, x, out, *args, **kwargs):
        min_clip = 1e-20
        a = np.log(np.clip(self.problem1.evaluate(x)-self.problem1.evaluate(self.o1), min_clip, 1e12))
        b = np.log(np.clip(self.problem2.evaluate(x + (-self.o1+self.o2))-self.problem2.evaluate(self.o2), min_clip, 1e12))
        out['F'] = np.exp(self.alpha *a + (1-self.alpha)*b )
        
    def pareto_set(self):
        return self.problem1.pareto_set()
        
        
def algorithm_factory(name, population=20):
    """
    Creates an instance of a population-based optimization algorithm with the given name and population size.

    Args:
        name (str): The name of the algorithm.
        population (int, optional): The size of the population. Defaults to 20.

    Returns:
        An instance of the specified algorithm.

    Raises:
        Exception: If the specified name is not one of the supported algorithms.
    """
    switch = {
        'GA': GA(pop_size=population, sampling=LHS()),
        'PSO': PSO(pop_size=population, sampling=LHS()),
        'DE': DE(pop_size=population, sampling=LHS(), CR=0.1),
        'CMAES': CMAES(popsize=population),
        'ES': ES(pop_size=population),
    }
    algorithm = switch.get(name)
    if algorithm is None:
        raise Exception("Unsupported algorithm: {}".format(name))
    return algorithm


def run_algorithms(problem, n_runs=10, n_eval=1000, algorithms=['GA', 'PSO', 'DE', 'CMAES', 'ES']):
    """
    Runs a set of population-based optimization algorithms on the given problem, for a specified number of runs and evaluation budget.

    Args:
        problem (pygmo.problem): The problem to be optimized.
        n_runs (int, optional): The number of runs. Defaults to 10.
        n_eval (int, optional): The evaluation budget per run. Defaults to 1000.
        algorithms (list of str, optional): The names of the algorithms to be used. Defaults to ['GA', 'PSO', 'DE', 'CMAES', 'ES'].

    Returns:
        A dictionary containing the optimization results for each algorithm and run.
    """
    algo_data = {x: [] for x in algorithms}
    algo_data['algorithm_run'] = []
    total_n_evals = problem.n_var*n_eval
    #print(f"Using {total_n_evals} evals")
    for run in range(n_runs):
        algo_data['algorithm_run'].append(run)
        for algo_name in algorithms:
            algorithm = algorithm_factory(algo_name)
            termination = get_termination("n_eval", total_n_evals)
            res = minimize(problem, algorithm, termination=termination, verbose=False, save_history=False)
            algo_data[algo_name].append(res.F[0])
    return algo_data
