import os
import numpy as np

def geneData(meanval,varval,dataname):
    datasize=270
    batchnum=30
    if not os.path.exists(dataname):
        os.makedirs(os.path.join(dataname))
    for i in range(batchnum):
        f=open(os.path.join(dataname,dataname+'_batch_'+str(i)),'w')
        speed=np.random.normal(meanval,varval**2,datasize)
        intervals=np.random.uniform(0.5,1,datasize)
        curtime=0.0
        for j in range(datasize):
            f.write('%.11f' % curtime)
            f.write('\t')
            if speed[j]>0:
                f.write('%.11f' % speed[j])
            else:
                f.write('%.11f' % 0.0001)
            f.write('\n')
            curtime+=intervals[j]
        f.close()

if __name__ == '__main__':
    geneData(0.5,0.5,'./stable')
    geneData(0.8,1,'./unstable')
