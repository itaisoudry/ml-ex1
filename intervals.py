import matplotlib.pyplot as plt
import numpy as np
from numpy import *


def find_best_interval(xs, ys, k):
    assert all(array(xs) == array(sorted(xs))), "xs must be sorted!"

    xs = array(xs)
    ys = array(ys)
    m = len(xs)
    P = [[None for j in range(k + 1)] for i in range(m + 1)]
    E = zeros((m + 1, k + 1), dtype=int)

    # Calculate the cumulative sum of ys, to be used later
    cy = concatenate([[0], cumsum(ys)])

    # Initialize boundaries:
    # The error of no intervals, for the first i points
    E[:m + 1, 0] = cy

    # The minimal error of j intervals on 0 points - always 0. No update needed.

    # Fill middle
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            # The minimal error of j intervals on the first i points:

            # Exhaust all the options for the last interval. Each interval boundary is marked as either
            # 0 (Before first point), 1 (after first point, before second), ..., m (after last point)
            options = []
            for l in range(0, i + 1):
                next_errors = E[l, j - 1] + (cy[i] - cy[l]) + concatenate(
                    [[0], cumsum((-1) ** (ys[arange(l, i)] == 1))])
                min_error = argmin(next_errors)
                options.append((next_errors[min_error], (l, arange(l, i + 1)[min_error])))

            E[i, j], P[i][j] = min(options)

    # Extract best interval set and its error count
    best = []
    cur = P[m][k]
    for i in range(k, 0, -1):
        best.append(cur)
        cur = P[cur[0]][i - 1]
        if cur == None:
            break
    best = sorted(best)
    besterror = E[m, k]

    # Convert interval boundaries to numbers in [0,1]
    exs = concatenate([[0], xs, [1]])
    representatives = (exs[1:] + exs[:-1]) / 2.0
    intervals = [(representatives[l], representatives[u]) for l, u in best]

    return intervals, besterror


def get_points(num_of_points):
    xs = sort(np.random.uniform(0, 1, num_of_points))
    ys = []
    for x in xs:
        if 0 <= x <= 0.25 or 0.5 <= x <= 0.75:
            ys.append(np.random.binomial(1, 0.8))
        if 0.25 <= x <= 0.5 or 0.75 <= x <= 1:
            ys.append(np.random.binomial(1, 0.1))

    return xs, ys


def a():
    xs, ys = get_points(100)
    result = find_best_interval(xs, ys, 2)
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([-0.1, 1.1])
    plt.axvline(0.25)
    plt.axvline(0.5)
    plt.axvline(0.75)
    plt.scatter(xs, ys)
    plt.axhline(0.5, result[0][0][0], result[0][0][1], color='r')
    plt.axhline(0.5, result[0][1][0], result[0][1][1], color='b')
    plt.savefig("a.png")
    plt.clf()


# b.
def smallest_error_hypothesis():
    return [(0, 0.25), (0.5, 0.75)]


def c(intervals):
    k = 2
    m = 10

    error = get_true_error(intervals[0])
    print(error)

    empirical_points = []
    true_error_points = []
    y_points = []
    while m <= 100:
        y_points.append(m)
        empirical_avg = 0
        true_error_avg = 0

        for t in range(0, 100):
            xs, ys = get_points(m)
            result = find_best_interval(xs, ys, k)

            true_error = get_true_error(result[0])
            true_error_avg += true_error
            empirical_avg += result[1] / m

            print("empirical error: " + str(result[1] / m))
            print("true error for new hypothesis: " + str(true_error))

        # calculate average for empirical error and true error
        empirical_avg = empirical_avg / 100
        true_error_avg = true_error_avg / 100
        # save points inside array
        empirical_points.append(empirical_avg)
        true_error_points.append(true_error_avg)

        m += 5

        # plot graph
    plot_empirical_and_true(empirical_points, true_error_points, y_points, 'm')


def d():
    m = 50
    xs, ys = get_points(m)
    empirical_points = []
    true_points = []
    k_points = []
    intervals = []

    for k in range(1, 21):
        result = find_best_interval(xs, ys, k)
        # save result for later use in question e
        intervals.append(result)
        empirical_points.append(result[1] / m)
        true_points.append(get_true_error(result[0]))
        # save array of k's for using in the plot as the y axis
        k_points.append(k)

    plot_empirical_and_true(empirical_points, true_points, k_points, 'k')
    return intervals


def e(d_result):
    xs, ys = get_points(50)
    best_empirical = 100
    best_hypothesis = []

    for k in range(0, 20):
        errors = 0
        hypothesis = d_result[k][0]
        for i in range(0, 50):
            found = False
            x = xs[i]
            y = ys[i]

            for interval in hypothesis:
                # if interval containing x found, break
                if interval[0] <= x <= interval[1]:
                    found = True
                    break
            # check for errors
            if found and y == 0:
                errors += 1
            if not found and y == 1:
                errors += 1

        if errors < best_empirical:
            best_empirical = errors
            best_hypothesis = hypothesis

    print("Best Hypthesis is:" + str(best_hypothesis))
    return best_hypothesis


def get_true_error(intervals):
    overlapping = 0
    not_overlapping = 0

    for interval in intervals:
        if (0 <= interval[0] <= 0.25 and 0 <= interval[1] <= 0.25) or (
                            0.5 <= interval[0] <= 0.75 and 0.5 <= interval[1] <= 0.75):
            overlapping += interval[1] - interval[0]

        if 0 <= interval[0] <= 0.25 and 0.5 <= interval[1] <= 0.75:
            overlapping += interval[0] + (interval[1] - 0.5)
            not_overlapping += 0.25

        if 0 <= interval[0] <= 0.25 <= interval[1] <= 0.5:
            overlapping += 0.25 - interval[0]
            not_overlapping += interval[1] - 0.25

        if 0.25 <= interval[0] <= 0.5 <= interval[1] < 0.75:
            overlapping += interval[1] - 0.5
            not_overlapping += 0.5 - interval[0]

        if 0.5 <= interval[0] <= 0.75 < interval[1]:
            overlapping += 0.75 - interval[0]
            not_overlapping += interval[1] - 0.75

        if (0.25 < interval[0] < 0.5 and 0.25 < interval[1] < 0.5) or (
                            0.75 < interval[0] <= 1 and 0.75 < interval[1] <= 1):
            not_overlapping += interval[1] - interval[0]

    error = 0.2 * overlapping + 0.8 * (0.5 - overlapping) + 0.9 * not_overlapping + 0.1 * (0.5 - not_overlapping)
    return error


def plot_empirical_and_true(empirical, true, y_points, x_label):
    plt.scatter(y_points, empirical, color='r')
    plt.scatter(y_points, true, color='b')
    plt.xlabel(x_label)
    plt.ylabel('errors')

    filename = "d.png"
    if x_label == 'k':
        filename='c.png'

    plt.savefig(filename)
    plt.clf()


a()
xs, ys = get_points(100)
result = find_best_interval(xs, ys, 2)
c(result)
best_hypothesis = d()
e(best_hypothesis)
