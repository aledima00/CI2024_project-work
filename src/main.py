import symgp
from symgp import BaseModel
import numpy as np
import os
from tqdm.auto import tqdm
import beepy

FNAME = "./s314935.py"

# parameters of the model
MAX_DEPTH = 10
POPULATION_SIZE = 2000
FITNESS_GRP_RATIO = 0.16 # rule of thumb for population size = 2000
GENERATIONS = 100
POOL_SIZE = 2
PARSIMONY_FORMAT = "bilinear"
# some parameters can be provided as a tuple that represent the start and end values adjusted by the generation
# the provided tuples must have format (start,end)
MUTATION_RATE = (0.05, 0.1) # more exploration at the beginning, more exploitation at the end
ELITISM_RATE = 0.05 # fixed to avoid premature convergence
PARSIMONY_WEIGHT = (0.001, 0.0001) # allow only simple solutions at the beginning, allow more complex solutions at the end

with open(FNAME,"w") as f:
    f.write("import numpy as np\n\n")

generation_params = {
    "int_constants":True,
    "randc_mean":0,
    "randc_std":5,
    "ctv_prop":0.15,
    "stv_prop":0.2,
    "unary_to_others_prop":0.4
}

for pnum in range(1,9):
    print(f"SOLVING PROBLEM {pnum}")

    file_path = os.path.join(".", f"data/problem_{pnum}.npz")
    problem = np.load(file_path)
    X,Y = problem["x"], problem["y"]

    num_inputs = X.shape[0]
    LEAVES_NAMES = [f"x[{i}]" for i in range(num_inputs)]

    mod =  BaseModel(
        max_depth=MAX_DEPTH,
        population_size=POPULATION_SIZE,
        input_leaves_names=LEAVES_NAMES,
        rand_seed=np.random.randint(10000),
        generation_params=generation_params
    )

    mod.populate()
    mod.evolve(X,Y,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        elitism_rate=ELITISM_RATE,
        pool_size=POOL_SIZE,
        parsimony_weight=PARSIMONY_WEIGHT,
        parsimony_format=PARSIMONY_FORMAT
    )

    best = mod.population()[0]
    print("best fitness:",best.fitness(X,Y,LEAVES_NAMES))
    print(f"best individual:{best.fstr()}")
    print("mse:",best.mse(X,Y,LEAVES_NAMES))

    # create a file with the returned expression
    with open(FNAME,"a") as f:
        f.write(f"def f{pnum}(x:np.ndarray)->np.ndarray:\n\t")
        f.write(f"# estimated function for problem{pnum}\n\t")
        f.write(f"# mse: {best.mse(X,Y,LEAVES_NAMES):.3e}\n\t")
        f.write(f"return {best.getExpr()}\n\n")
