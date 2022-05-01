import numpy as np
import time
# Computer exercise 1
from matplotlib import pyplot as plt


def G(row_s: np.ndarray, temp: float):
    return np.exp((1 / temp) * np.sum(row_s[:len(row_s) - 1] * row_s[1:]))


# Computer exercise 2
def F(row_s: np.ndarray, row_t: np.ndarray, temp: float):
    return np.exp((1 / temp) * np.sum(row_s * row_t))


# Computer exercise 3
def exercise3():
    temps = [1, 1.5, 2]
    Z = []
    for temp in temps:
        Ztemp = 0
        for x11 in [-1, 1]:
            for x12 in [-1, 1]:
                for x21 in [-1, 1]:
                    for x22 in [-1, 1]:
                        r = np.array([x11, x11, x21, x12])
                        rhat = np.array([x12, x21, x22, x22])
                        Ztemp += F(r, rhat, temp)
        print("Z" + str(temp) + ":", Ztemp)
        Z.append(Ztemp)


# Computer exercise 4
def exercise4():
    temps = [1, 1.5, 2]
    Z = []
    for temp in temps:
        Ztemp = 0
        for x11 in [-1, 1]:
            for x12 in [-1, 1]:
                for x13 in [-1, 1]:
                    for x21 in [-1, 1]:
                        for x22 in [-1, 1]:
                            for x23 in [-1, 1]:
                                for x31 in [-1, 1]:
                                    for x32 in [-1, 1]:
                                        for x33 in [-1, 1]:
                                            r = np.array([x11, x11, x12, x12, x13, x21, x21, x22, x22, x23, x31, x32])
                                            rhat = np.array(
                                                [x12, x21, x13, x22, x23, x22, x31, x23, x32, x33, x32, x33])
                                            Ztemp += F(r, rhat, temp)
        print("Z" + str(temp) + ":", Ztemp)
        Z.append(Ztemp)


# Computer exercise 5
def y2row(y, width=8):
    """
    y: an integer in (0,...,(2**width)-1)
    """
    if not 0 <= y <= (2 ** width) - 1:
        raise ValueError(y)
    my_str = np.binary_repr(y, width=width)
    # my_list = map(int,my_str) # Python 2
    my_list = list(map(int, my_str))  # Python 3
    my_array = np.asarray(my_list)
    my_array[my_array == 0] = -1
    row = my_array
    return row


def exercise5():
    temps = [1, 1.5, 2]
    Z = []
    for temp in temps:
        Ztemp = 0
        for y1 in range(2 ** 2):
            for y2 in range(2 ** 2):
                row1 = y2row(y1, width=2)
                row2 = y2row(y2, width=2)
                Ztemp += G(row1, temp) * G(row2, temp) * F(row1, row2, temp)
        print("Z" + str(temp) + ":", Ztemp)
        Z.append(Ztemp)


# Computer exercise 6
def exercise6():
    temps = [1, 1.5, 2]
    Z = []
    for temp in temps:
        Ztemp = 0
        for y1 in range(2 ** 3):
            for y2 in range(2 ** 3):
                for y3 in range(2 ** 3):
                    row1 = y2row(y1, width=3)
                    row2 = y2row(y2, width=3)
                    row3 = y2row(y3, width=3)
                    Ztemp += G(row1, temp) * G(row2, temp) * G(row3, temp) * F(row1, row2, temp) * F(row2, row3, temp)
        print("Z" + str(temp) + ":", Ztemp)
        Z.append(Ztemp)


def getTs(size, temp):
    Ts = []
    rangeY = range(2 ** size)
    T1 = [sum([G(y2row(y1, size), temp) * F(y2row(y1, size), y2row(y2, size), temp) \
               for y1 in rangeY]) \
          for y2 in rangeY]
    Ts.append(T1)
    prevT = T1
    for _ in range(1, size - 1):
        nextT = [sum([prevT[y1] * G(y2row(y1, size), temp) * F(y2row(y1, size), y2row(y2, size), temp) \
                      for y1 in rangeY]) \
                 for y2 in rangeY]
        Ts.append(nextT)
        prevT = nextT
    lastT = sum([prevT[y] * G(y2row(y, size), temp) for y in rangeY])
    Ts.append(lastT)
    return Ts


def getPs(size, temp, Ts):
    Ztemp = Ts[-1]
    ps = []
    firstP = lambda y: Ts[-2][y] * G(y2row(y, size), temp) / Ztemp
    ps.append(firstP)
    for i in range(size - 2, 0, -1):
        nextP = lambda y1, y2: Ts[i - 1][y1] * G(y2row(y1, size), temp) * F(y2row(y1, size), y2row(y2, size), temp) \
                               / Ts[i][y2]
        ps.append(nextP)
    lastP = lambda y1, y2: G(y2row(y1, size), temp) * F(y2row(y1, size), y2row(y2, size), temp) / Ts[0][y2]
    ps.append(lastP)
    return ps


def getPsTable(size, ps):
    psTable = []
    y_range = range(2 ** size)
    psTable.append([ps[0](y) for y in y_range])
    for i in range(1, size):
        psTable.append([[ps[i](y1, y2) for y1 in y_range] for y2 in y_range])
    return psTable


def singleSample(size, psTable):
    sample = []
    y_range = range(2 ** size)
    y_first = np.random.choice(y_range, p=psTable[0])
    sample.append(y_first)
    y_prev = y_first
    for i in range(1, size):
        y_next = np.random.choice(y_range, p=psTable[i][y_prev])
        sample.append(y_next)
        y_prev = y_next
    # no need for last
    parsed_sample = np.array([np.array(y2row(y, size)) for y in sample])
    return parsed_sample


def sampleN(n, size, temp):
    Ts = getTs(size, temp)
    ps = getPs(size, temp, Ts)
    psTable = getPsTable(size, ps)
    samples = np.empty((n, size, size))
    for i in range(n):
        samples[i] = singleSample(size, psTable)
    return samples


# Computer exercise 7
def exercise7():
    fig, axes = plt.subplots(nrows=3, ncols=10)
    for i, temp in enumerate([1, 1.5, 2]):
        exact_samples = sampleN(10, 8, temp)
        for j, sample in enumerate(exact_samples):
            axes[i, j].imshow(sample, interpolation='None')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f"Temp = {temp}")
    plt.show()


# Computer exercise 8
def exercise8():
    size = 8
    n = 10000
    print("Exercise 8:")
    for temp in [1, 1.5, 2]:
        Ts = getTs(size, temp)
        ps = getPs(size, temp, Ts)
        psTable = getPsTable(size, ps)
        sum1122 = 0
        sum1188 = 0
        for _ in range(n):
            sample = singleSample(size, psTable)
            sum1122 += sample[0, 0] * sample[1, 1]
            sum1188 += sample[0, 0] * sample[7, 7]
        print(f"For Temp = {temp} -> E(X11X22) = {sum1122 / n}")
        print(f"For Temp = {temp} -> E(X11X88) = {sum1188 / n}")


def randomPaddedLattice(size):
    return np.pad((np.random.randint(low=0, high=2, size=(size, size)) * 2 - 1), 1, padWith)


def padWith(vector, pad_width, _, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def gibbsSampler(size, sample, neighbors, temp):
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            XsXt = np.array(sample[(neighbors[0] + i, neighbors[1] + j)])
            exponent = np.sum(XsXt) / temp
            plusRatio = np.exp(exponent)
            minusRatio = np.exp(-exponent)
            p = [plusRatio, minusRatio] / (plusRatio + minusRatio)
            sample[i, j] = np.random.choice([1, -1], p=p)


# Computer exercise 9
def exercise9Ergodicity():
    neighbors = np.array([
        np.array([1, 0, -1, 0]),
        np.array([0, 1, 0, -1])])
    size = 8
    burnInPeriod = 100
    numSweeps = 25000
    print("Exercise 9 - Ergodicity:")
    for temp in [1, 1.5, 2]:
        sum1122 = 0
        sum1188 = 0
        sample = randomPaddedLattice(size)
        for k in range(numSweeps):
            gibbsSampler(size, sample, neighbors, temp)
            if k > burnInPeriod:
                sum1122 += sample[1, 1] * sample[2, 2]
                sum1188 += sample[1, 1] * sample[size, size]
        print(f"For Temp = {temp} -> E(X11X22) = {sum1122 / (numSweeps - burnInPeriod)}")
        print(f"For Temp = {temp} -> E(X11X88) = {sum1188 / (numSweeps - burnInPeriod)}\n")


def exercise9Independent():
    size = 8
    n = 10000
    numSweeps = 25
    neighbors = np.array([
        np.array([1, 0, -1, 0]),
        np.array([0, 1, 0, -1])])
    print("Exercise 9 - Independent:")
    for temp in [1, 1.5, 2]:
        sum1122 = 0
        sum1188 = 0
        for _ in range(n):
            sample = randomPaddedLattice(size)
            for _ in range(numSweeps):
                gibbsSampler(size, sample, neighbors, temp)
            sum1122 += sample[1, 1] * sample[2, 2]
            sum1188 += sample[1, 1] * sample[size, size]

        print(f"For Temp = {temp} -> E(X11X22) = {sum1122 / n}")
        print(f"For Temp = {temp} -> E(X11X88) = {sum1188 / n}\n")


def gibbsSample(neighbors, size, temp, n_sweeps):
    sample = randomPaddedLattice(size)
    for k in range(n_sweeps):
        for i in range(1, size + 1):
            for j in range(1, size + 1):
                XsXt = np.array(sample[(neighbors[0] + i, neighbors[1] + j)])
                in_exp = np.sum(XsXt) / temp
                plus_ratio = np.exp(in_exp)
                minus_ratio = np.exp(-in_exp)
                p = [plus_ratio, minus_ratio] / (plus_ratio + minus_ratio)
                sample[i, j] = np.random.choice([1, -1], p=p)
    return sample[1:-1, 1:-1]  # padding removed


def posteriorGibbsSample(neighbors, size, Temp, sweeps, y, sigma):
    sample = randomPaddedLattice(size)
    y_padded = np.pad(y, 1, padWith)
    for k in range(sweeps):
        for i in range(1, size + 1):
            for j in range(1, size + 1):
                XsXt = np.array(sample[(neighbors[0] + i, neighbors[1] + j)])
                plusRatio = np.exp(np.sum(XsXt) / Temp \
                                   - np.square(y_padded[i, j] + 1) / (2 * (sigma ** 2)))
                minusRatio = np.exp(-np.sum(XsXt) / Temp \
                                    - np.square(y_padded[i, j] - 1) / (2 * (sigma ** 2)))
                p = [plusRatio, minusRatio] / (plusRatio + minusRatio)
                sample[i, j] = np.random.choice([1, -1], p=p)
    return sample[1:-1, 1:-1]  # padding removed


def maxPosteriorGibbs(neighbors, size, Temp, n_sweeps, y, sigma):
    sample = randomPaddedLattice(size)
    y_padded = np.pad(y, 1, padWith)
    for k in range(n_sweeps):
        for i in range(1, size + 1):
            for j in range(1, size + 1):
                neibs = np.array(sample[(neighbors[0] + i, neighbors[1] + j)])
                plus_ratio = np.exp(np.sum(neibs) / Temp \
                                    - np.square(y_padded[i, j] + 1) / (2 * (sigma ** 2)))
                minus_ratio = np.exp(-np.sum(neibs) / Temp \
                                     - np.square(y_padded[i, j] - 1) / (2 * (sigma ** 2)))
                sample[i, j] = np.argmax([minus_ratio, plus_ratio]) * 2 - 1
    return sample[1:-1, 1:-1]  # padding removed


def exercise10():
    size = 100
    sweeps = 50
    eta = 2 * np.random.standard_normal(size=(size, size))
    temp = [1, 1.5, 2]
    neighbors = np.array([
        np.array([1, 0, -1, 0]),
        np.array([0, 1, 0, -1])])

    fig, axes = plt.subplots(nrows=3, ncols=5)
    fig.suptitle("Temp: [1, 1.5, 2]".format(temp))
    for i, temp in enumerate([1, 1.5, 2]):
        x = gibbsSample(neighbors, size, temp, sweeps)
        y = x + eta
        posteriorSample = posteriorGibbsSample(neighbors, size, temp, sweeps, y, 2)
        posteriorMax = maxPosteriorGibbs(neighbors, size, temp, sweeps, y, 2)
        maxLikelihood = np.sign(y)
        for axe in axes[i]:
            axe.axis('off')
        axes[i, 0].imshow(x, interpolation='None')
        axes[i, 0].set_title('x')
        axes[i, 1].imshow(y, interpolation='None')
        axes[i, 1].set_title('y')
        axes[i, 2].imshow(posteriorSample, interpolation='None')
        axes[i, 2].set_title('posteriorSample')
        axes[i, 3].imshow(posteriorMax, interpolation='None')
        axes[i, 3].set_title('posteriorMax')
        axes[i, 4].imshow(maxLikelihood, interpolation='None')
        axes[i, 4].set_title('maxLikelihood')
    plt.show()

startTime = time.time()
# exercise3()
# print("\n-------------------------------------------------------- \n")
# exercise4()
# print("\n-------------------------------------------------------- \n")
# exercise5()
# print("\n-------------------------------------------------------- \n")
# exercise6()
# print("\n-------------------------------------------------------- \n")
exercise7()
print(f"\nExercise 7 runtime is = {(time.time()- startTime)} seconds")
print("\n-------------------------------------------------------- \n")
exercise8()
print(f"\nExercise 8 runtime is = {(time.time()- startTime)} seconds")
print("\n-------------------------------------------------------- \n")
exercise9Ergodicity()
print("\n-------------------------------------------------------- \n")
exercise9Independent()
print(f"\nExercise 9 runtime is = {(time.time()- startTime)} seconds")
print("\n-------------------------------------------------------- \n")
exercise10()
print(f"\nExercise 10 runtime is = {(time.time()- startTime)} seconds\n")

print(f"Total runtime is = {(time.time()- startTime)} seconds")