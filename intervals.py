import matplotlib.pyplot as plt
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


def plot(xs, ys):
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([-1, 1.1])
    plt.axvline(0.25)
    plt.axvline(0.5)
    plt.axvline(0.75)
    plt.scatter(xs, ys)
    plt.show()


def plot_intervals(result):
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([-1, 1.1])
    plt.axhline(-1, result[0][0][0], result[0][0][1], color='r')
    plt.axhline(-1, result[0][1][0], result[0][1][1], color='b')
    plt.show()


# b. Proof: Since we would like to create the Hypothesis with the smallest true error
# and we know the probability P, we can just calculate the probability of an error.
# But, since we need to return a 0\1 result for x in some intervals such that the number
# of errors will be the smallest, we choose the first two intervals.
# there is a probability of 0.8 to x in the first and third quarter to be labeled as one
# and on the other hand, 0.1 to be 1 if x is in the second or fourth quarter, which is very unlikely.
# so if x is in the first or third quarter, it has the highest probability to be labeled correctly, thus minimizing
# the probability of making a mistake.
def smallest_error_hypothesis():
    return [(0, 0.25), (0.5, 0.75)]


def c(intervals):
    k = 2
    m = 10

    error = get_overlap(intervals[0][0]) + get_overlap(intervals[0][1])
    print(error)

    while m < 100:
        xs, ys = get_points(m)
        result = find_best_interval(xs, ys, k)
        print("empirical error: " + k)
        print("true error for new hypothesis: " + get_overlap(result[0][0] + get_overlap(result[0][1])))
        m += 5


def get_overlap(a):
    error = 0

    # fully overlap
    if a[0] >= 0 and a[1] <= 0.25:
        error += 0.2 * (a[0] - 0) + (0.25 - a[1])
        print(1)
    if a[0] >= 0.5 and a[1] <= 0.75:
        error += 0.2 * (a[0] - 0.5) + (0.75 - a[1])
        print(2)
    # no overlap
    if (a[0] > 0.25 and a[1] < 0.5) or (a[0] > 0.75):
        error += 0.9 * (a[1] - a[0])
        print(3)
    # partial overlap
    if a[1] > 0.25 and a[0] < 0.25:
        error += a[0] * 0.2 + (a[1] - 0.25) * 0.9
        print(4)
    if a[1] > 0.5 and a[0] < 0.5:
        error += (a[1] - 0.5) * 0.2 + (0.5 - a[0]) * 0.9
        print(5)
    if 0.5 < a[0] < 0.75 and a[1] > 0.75:
        error += (a[1] - 0.75) * 0.9 + (0.75 - a[0]) * 0.2
        print(6)
    if a[0] < 0.5 and a[1] > 0.75:
        error += 0.9 * (0.5 - a[0] + a[1] - 0.75)
        print(7)

    return error


xs, ys = get_points(100)
result = find_best_interval(xs, ys, 2)
print(result)
c(result)

# print(smallest_error_hypothesis(2))
# PLOTS
# plot_intervals(result)
# plot(xs, ys)
