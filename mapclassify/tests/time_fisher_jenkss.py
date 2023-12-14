import random
import timeit
import functools

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import mapclassify
import mapclassify.classifiers

import numpy as np

try:
    import numba
    raise Exception(
        f"""This test is to compare execution times of two alternatives 
        to the Numba-optimised function (both of which we already 
        know are far slower).  
        
        Please run {__file__} again in a venv, 
        in which Numba is not installed. """
        )
except ImportError:
    pass


number_tests = 1
k = 8


   
def test_fisher_jenks_means(N, HAS_NUMBA):

    data = [random.uniform(1.0, 1000.0) for __ in range(N)]

    if HAS_NUMBA:
        func = mapclassify.classifiers._fisher_jenks_means(np.sort(data).astype("f8"), classes=k)
    else:
        func = mapclassify.classifiers._fjm_without_numpy(sorted(data), classes=k)
   


def test_mapclassify_classify_fisherjenks(N, HAS_NUMBA):

    data = [random.uniform(1.0, 1000.0) for __ in range(N)]


    # This hack avoids changing the interface of the 
    # FisherJenks class, just for this timing code.
    mapclassify.classifiers.HAS_NUMBA = HAS_NUMBA


    mapclassify.classify(y = data, scheme = 'fisherjenks', k=k)


    



data_sizes = [100, 300, 1000, 2800, 8000]




def compare_times(test_runner, descriptions, title):


    print(f'{title}\n')

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8.5, 5), layout='constrained')




    for HAS_NUMBA, description in zip([False, True], descriptions):

        times = []

        for N in data_sizes:
            
            t = timeit.timeit(functools.partial(test_runner, N=N, HAS_NUMBA = HAS_NUMBA), number=number_tests)

            print(f'Time: {t:.3f} seconds, data points: {N} {description}, {number_tests=}')

            times.append(t)


        if HAS_MATPLOTLIB:
            ax.plot(data_sizes, times, 'o-', label=description)


    if HAS_MATPLOTLIB:
        ax.set_xlabel('Number of random data points classified')  
        ax.set_ylabel('Run time (seconds)')  
        ax.set_title(title)  
        ax.legend() 

        plt.show()

compare_times(
    test_fisher_jenks_means,
    title="Run times of the proposed function vs the original (excluding MapClassifier overhead)",
    descriptions = [" _fjm_without_numpy",
                    " _fisher_jenks_means",
                   ]
    )

compare_times(
    test_mapclassify_classify_fisherjenks,
    title="Run times for end user, of the proposed code vs the original (inc MapClassifier overhead)",
    descriptions = ["without Numpy, much less slow, pure python code",
                    'with Numpy, existing "slow pure python" code',
                   ],
    )