import sys
line1 = sys.stdin.readline().strip().split(' ')
arry = list(map(int, sys.stdin.readline().strip().split(' ')))
n = int(line1[0])
k = int(line1[1])
arry.sort()
res = 0
dict = {}
for i in range(n-1):
    for j in range(i+1, n):
        if arry[j] == arry[i] and k == 0:
            dict[arry[i]] = 1
            break
        elif arry[j] == arry[i] and k != 0:
            break
        elif arry[j] - arry[i] < k:
            continue
        elif arry[j] - arry[i] == k:
            res += 1
            break
for i in dict.values():
    res += i
print(res)
