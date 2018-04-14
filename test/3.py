import sys
n = int(sys.stdin.readline().strip())
res = 0
s, m = 1, 1


def operation1(s, m, res):
    m = s
    s += s
    res += 2
    return s, m, res

def opreation2(s, m, res):
    s += m
    res += 1
    return s, m, res

while s!=n:
    if n-s >= 3*s:
        s, m, res = operation1(s, m, res)
        s, m, res = operation1(s, m, res)
