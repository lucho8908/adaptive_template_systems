# 1: +train 2: -train 3: +test 4:-test
# http://pongor.itk.ppke.hu/benchmark/#/Benchmark_data_formats
import numpy as np
import os

os.system('mkdir Index')

mat = np.empty([1357,55], int)
infile = open('./CAST.txt')
lines = infile.read().splitlines()
for i in range(len(lines)):
    line = lines[i]
    a = line[7:].split()
    for j in range(55):
        mat[i,j] = int(a[j])

for i in range(55):
    print(i+1)
    TrainIndex = []
    TestIndex = []
    TrainLabel = []
    TestLabel = []
    for j in range(1357):
        if mat[j,i] == 1 or mat[j,i] == 2:
            TrainIndex.append(j)
            if mat[j,i] == 1:
                TrainLabel.append(1)
            elif mat[j,i] == 2:
                TrainLabel.append(-1)
        if mat[j,i] == 3 or mat[j,i] == 4:
            TestIndex.append(j)
            if mat[j,i] == 3:
                TestLabel.append(1)
            elif mat[j,i] == 4:
                TestLabel.append(-1)

    TrainIndex = np.asarray(TrainIndex, int)
    TestIndex = np.asarray(TestIndex, int)
    TrainLabel = np.asarray(TrainLabel, int)
    TestLabel = np.asarray(TestLabel, int)

    print(len(TrainIndex), np.sum(TrainLabel), len(TestIndex), np.sum(TestLabel), len(TrainIndex)+len(TestIndex))

    outfile = open('./Index/TrainIndex'+str(i+1)+'.npy','wb')
    np.save(outfile, TrainIndex)
    outfile.close()
    outfile = open('./Index/TrainLabel'+str(i+1)+'.npy','wb')
    np.save(outfile, TrainLabel)
    outfile.close()
    outfile = open('./Index/TestIndex'+str(i+1)+'.npy','wb')
    np.save(outfile, TestIndex)
    outfile.close()
    outfile = open('./Index/TestLabel'+str(i+1)+'.npy','wb')
    np.save(outfile, TestLabel)
    outfile.close()
