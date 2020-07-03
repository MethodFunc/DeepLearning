import numpy as np


def calcul1(x):
    return np.sqrt((x - 1) * ((x - 3) ** 2) * ((x - 5) ** 3))


def calcul2(x, y):
    cal = np.log10(x ** y)
    print(cal, end=' ')
    if cal >= 5 and cal <= 10:
        return True
    else:
        return False


def calcul3(n, m):
    facto = m
    permutation = n
    for i in range(m, 0, -1):
        if i - 1 == 0:
            pass
        else:
            n -= 1
            facto = facto * (i - 1)
            permutation = permutation * n

    return permutation / facto


def calcul4(x):
    for i in range(1, x+1):
        for j in range(i):
            print(i, end='')

print(calcul1(5.6))
print(calcul2(2.0, 7.3))
print(calcul3(45, 6))
calcul4(8)