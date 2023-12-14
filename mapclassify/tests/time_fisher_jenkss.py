import random
import timeit
import functools

import matplotlib.pyplot as plt

import mapclassify

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


   
def test_fisher_jenks(N):

    data = [random.randint(1, 1000) for __ in range(N)]

    mapclassify.classify(y = data, scheme = 'fisherjenks', k=8)

descriptions = ["without Numpy, proposed less slow Pure Python code",
                'with Numpy, existing "slow pure python" code',
               ]

data_sizes = [100, 300, 900, 1400, 2100, 3800, 10000]


fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')




for HAS_NUMBA, description in zip([False, True], descriptions):

    times = []

    for N in data_sizes:
        
        # This hack avoids changing the FisherJenks 
        # interface, just for this profiling code.
        mapclassify.classifiers.HAS_NUMBA = HAS_NUMBA

        t = timeit.timeit(functools.partial(test_fisher_jenks, N=N), number=number_tests)

        print(f'Time: {t} seconds, data points: {N} {description}, {number_tests=}')

        times.append(t)

    ax.plot(data_sizes, times, label=description)

ax.set_xlabel('Size of data classified')  # Add an x-label to the axes.
ax.set_ylabel('Run time')  # Add a y-label to the axes.
ax.set_title('Comparison of Fisher Jenks implementations. ')  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.show()