
import paraFilter 
import nibabel as nib

import numpy as np

from multiprocessing import Process, Manager,Pool



img = nib.load('t1_icbm_normal_1mm_pn3_rf20.mnc')
data = img.get_data()


data.shape



# things can be changed:
#number of filter for each batch: num

#stride:stride
#proc number:proc



#num = range(4,6)
num=[4,8,16,32]
stride=6
mycon = False
proc = [2,4,8,16]
import time
sigma = 1.0  # width of kernel
x = np.arange(-3, 4, 1)  # coordinate arrays -- make sure they contain 0!
y = np.arange(-3, 4, 1)
z = np.arange(-3, 4, 1)
xx, yy, zz = np.meshgrid(x, y, z)
kern = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
def getKernels(kern,numOfKern):
    kernels=[]
    # first build the smoothing kernel

    for i in range(numOfKern):
        kernels.append([i,kern])
    return kernels


#performencePara = np.zeros(shape=[len(num),len(stride),len(proc)])
performencePara = np.zeros(shape=[len(num),len(proc)])

performencePara2 = np.zeros(shape=[len(num),len(proc)])


performenceSera = np.zeros(shape=[len(num),1])


for i in range(performencePara.shape[0]):
    for j in range(performencePara.shape[1]):
        
            kernels =  getKernels(kern,num[i])
            start = time.time()
            results = paraFilter.convPara(data,kernels,proc[j],stride, mycon)
            end = time.time()
            lapse = end-start
            performencePara[i,j]=lapse    

mycon = True


for i in range(performencePara2.shape[0]):
    for j in range(performencePara2.shape[1]):
        
            kernels =  getKernels(kern,num[i])
            start = time.time()
            results = paraFilter.convPara(data,kernels,proc[j],stride, mycon)
            end = time.time()
            lapse = end-start
            performencePara2[i,j]=lapse   


import time

for i in range(performenceSera.shape[0]):
        
        kernels =  getKernels(kern,num[i])
        start = time.time()
        results = paraFilter.convSera(data,kernels,stride)
        end = time.time()
        lapse = end-start
        performenceSera[i,0]=lapse
    

numOfKern=[]
numOfProc=[]
TIME=[]

for x in range(performencePara.shape[0]):
    for y in range(performencePara.shape[1]):
        numOfKern.append(pow(2,x+2))
        numOfProc.append(pow(2,y+1))
        TIME.append(performencePara[x,y])
        
numOfKern2=[]
numOfProc2=[]
TIME2=[]

for x in range(performencePara2.shape[0]):
    for y in range(performencePara2.shape[1]):
        numOfKern2.append(pow(2,x+2))
        numOfProc2.append(pow(2,y+1))
        TIME2.append(performencePara2[x,y])

numOfKern_s=[]
numOfProc_s=[]
TIME_s=[]
        
for x in range(performenceSera.shape[0]):
    for y in range(performenceSera.shape[1]):
        numOfKern_s.append(pow(2,x+2))
        numOfProc_s.append(1)
        TIME_s.append(performenceSera[x,y])

# fig = plt.figure()
# scat = fig.add_subplot(111, projection='3d')
#mesh = fig.add_subplot(212, projection='3d')
# scat.scatter(numOfKern, numOfProc, TIME, c="r", label="filter parallel")
# scat.scatter(numOfKern2, numOfProc2, TIME2, c="b", label="(filter + super-voxel) parallel")
# scat.scatter(numOfKern_s, numOfProc_s, TIME_s, c="g", label="serial")
# scat.set_xlabel('number of filters')
# scat.set_ylabel('number of procs')
# scat.set_zlabel('time/s')
#scat.legend(loc="upper left")
#mesh.plot_surface(numOfKern, numOfProc, time, linewidth=0, antialiased=False)



get_ipython().magic(u'pylab inline')
import matplotlib.ticker as ticker

fig = figure(figsize = (10,8))
ax = fig.add_subplot(111)

kerns = [2, 4, 8, 16, 32]
procs = [1, 2, 4, 8, 16]
#cax = plt.imshow(performencePara2)

k_l = [4, 8, 16, 32]
p_l = [2, 4, 8, 16]

x=np.unique(np.log(np.array(kerns)))
y=np.unique(np.log(np.array(procs)))
X,Y = np.meshgrid(x,y)

#Z=z.reshape(len(y),len(x))
Z = performencePara
cax = ax.pcolormesh(X,Y,Z)
ax.set_title("(filter + super-voxel) parallelism\n\n", fontsize=16)

cbt = plt.colorbar(cax)
cbt.ax.set_title('Time (seconds)')
ax.set_xticklabels(k_l)
ax.set_yticklabels(p_l)

ax.set_xlabel("number of kernels")
ax.set_ylabel("number of processes")

ax.xaxis.set_major_locator(ticker.FixedLocator([1,1.7,2.4,3.1]))
ax.yaxis.set_major_locator(ticker.FixedLocator([0.35,1,1.7,2.4]))

#plt.savefig("images/heat_filter+voxel.jpg", dpi=1200)
plt.show()

