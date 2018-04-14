import sys
line1 = sys.stdin.readline().strip().split(' ')
arry = list(map(int, sys.stdin.readline().strip().split(' ')))
arry2 = list(set(arry))
n = int(line1[0])
k = int(line1[1])
arry2.sort()
res = 0
for i in range(len(arry2)-1):
    for j in range(i+1, len(arry2)):
        if arry2[j] - arry2[i] < k:
            continue
        elif arry2[j] - arry2[i] == k:
            res += 1
            break
if k==0:
    arry.sort()
    mark = arry[0]-1
    for i in range(n-1):
        if arry[i+1] == arry[i] and arry[i] != mark:
            res += 1
            mark = arry[i]
print(res)
