import numpy as np

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


# Computer exercise 7
def getTs(size, Temp):
    Ts = []
    rangeY = range(2 ** size)
    T1 = [sum([G(y2row(y1, size), Temp) * F(y2row(y1, size), y2row(y2, size), Temp) \
               for y1 in rangeY]) \
          for y2 in rangeY]
    Ts.append(T1)
    prevT = T1
    for _ in range(1, size - 1):
        nextT = [sum([prevT[y1] * G(y2row(y1, size), Temp) * F(y2row(y1, size), y2row(y2, size), Temp) \
                      for y1 in rangeY]) \
                 for y2 in rangeY]
        Ts.append(nextT)
        prevT = nextT
    lastT = sum([prevT[y] * G(y2row(y, size), Temp) for y in rangeY])
    Ts.append(lastT)
    return Ts


def getPs(size, Temp, Ts):
    Ztemp = Ts[-1]
    ps = []
    firstP = lambda y: Ts[-2][y] * G(y2row(y, size), Temp) / Ztemp
    ps.append(firstP)
    for i in range(size - 2, 0, -1):
        nextP = lambda y1, y2: Ts[i - 1][y1] * G(y2row(y1, size), Temp) * F(y2row(y1, size), y2row(y2, size), Temp) \
                               / Ts[i][y2]
        ps.append(nextP)
    lastP = lambda y1, y2: G(y2row(y1, size), Temp) * F(y2row(y1, size), y2row(y2, size), Temp) / Ts[0][y2]
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


def sampleN(n, size, Temp):
    Ts = getTs(size, Temp)
    ps = getPs(size, Temp, Ts)
    psTable = getPsTable(size, ps)
    samples = np.empty((n, size, size))
    for i in range(n):
        samples[i] = singleSample(size, psTable)
    return samples


def exercise7():
    fig, axes = plt.subplots(nrows=3, ncols=10)
    for i, Temp in enumerate([1, 1.5, 2]):
        exact_samples = sampleN(10, 8, Temp)
        for j, sample in enumerate(exact_samples):
            axes[i, j].imshow(sample, interpolation='None')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f"Temp = {Temp}")
    plt.show()


# exercise3()
# exercise4()
# exercise5()
# exercise6()
exercise7()