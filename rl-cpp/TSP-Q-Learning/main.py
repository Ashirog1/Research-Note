import time

import numpy as np
from src.model import QLearning
from src.utils import (compute_greedy_route, load_data, route_distance,
                       trace_progress, write_overall_results)
from src.utils import transform_cpp_to_tsp
import data

EPOCHS = 4000
LEARNING_RATE = 0.2
GAMMA = 0.95
EPSILON = 0.1


def run_Q_learning(data_cpp:np.ndarray):

    data_tsp = transform_cpp_to_tsp(data_cpp)
    Q_table = np.zeros((len(data_tsp), len(data_tsp)))
    Q_table, cache_distance_best, cache_distance_comp = QLearning(
        Q_table,
        data_tsp,
        epsilon=EPSILON,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
    )
    # create seed edge list
    # for idx in len(data_tsp):
    greedy_route = compute_greedy_route(Q_table)
    print(greedy_route)
        # greedy_cost = route_distance(greedy_route, data_tsp) - data_cpp[idx][2]
        # print(greedy_route, greedy_cost)

    


    # res[c] = greedy_cost

def main():
    """Q Learning method is ran on each benchmark instance.
    Figures monitoring progress are saved in figures/
    """

    square = [(1,2,1), (2,3,1), (3,4,1), (4,1,1)]
    square = np.array(square)
    run_Q_learning(square)
    # print(f"Time to run : {round(time.time() - start, 3)}")
    # # Overall final results
    # write_overall_results(res, data, "_no_hp_tuning")


if __name__ == "__main__":
    main()
