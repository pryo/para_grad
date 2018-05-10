import numpy as np
import nibabel as nib
from fractions import gcd
from multiprocessing import Process, Manager, Pool
from functools import partial


def dimdatacorr(core_shape, data_shape):

    idx_core = np.argsort(core_shape)
    idx_data = np.argsort(data_shape)

    corr = np.array([list(x) for x in zip(idx_data, idx_core)])
    corr = corr[corr[:,0].argsort()]

    return corr, idx_core, idx_data

def paddata(data, stride, core_shape, data_shape, idx_core, idx_data):

    # padding to accomodate core divisibility

    Pdata = data

    for i in range(len(data_shape)):
        npad = ((0, 0), (0, 0), (0, 0))
        lcm = (np.sort(core_shape)[i]*stride) / gcd(np.sort(core_shape)[i],stride)
        divisibility = (-np.sort(data_shape)[i] % lcm)
        if divisibility != 0:
            npad = list(npad)
            npad[idx_data[i]] = (0, divisibility)
            npad = tuple(npad)
            Pdata = np.pad(Pdata, pad_width=npad, mode='constant', constant_values=0)
    
    Pdata = np.pad(Pdata, pad_width=((0, 0), (0, 0), (0, 4)), mode='constant', constant_values=0)

    return Pdata

def getsplits(Pdata, stride, corr, core_shape, do_include=True):

    lcm_I = (core_shape[corr[0,1]]*stride) / gcd(core_shape[corr[0,1]],stride)
    lcm_J = (core_shape[corr[1,1]]*stride) / gcd(core_shape[corr[1,1]],stride)
    lcm_K = (core_shape[corr[2,1]]*stride) / gcd(core_shape[corr[2,1]],stride)

    step_I = Pdata.shape[0] / lcm_I
    step_J = Pdata.shape[1] / lcm_J
    step_K = Pdata.shape[2] / lcm_K

    lst_I = np.arange(0,Pdata.shape[0]+1, step_I)
    lst_J = np.arange(0,Pdata.shape[0]+1, step_J)
    lst_K = np.arange(0,Pdata.shape[0]+1, step_K)

    iter_I = list()
    iter_J = list()
    iter_K = list()

    
    for i in range(len(lst_I)-1):
        iter_I.append(list((lst_I[i], lst_I[i+1])))
    iter_I = np.array(iter_I)

    for j in range(len(lst_J)-1):
        iter_J.append(list((lst_J[j], lst_J[j+1])))
    iter_J = np.array(iter_J)

    for k in range(len(lst_K)-1):
        iter_K.append(list((lst_K[k], lst_K[k+1])))
    iter_K = np.array(iter_K)

    if do_include:
 
        for i in iter_I:
            if i[0] == 0:
                i[1] = i[1] + 2*lcm_I
            elif i[1] == Pdata.shape[0]:
                i[0] = i[0] - 2*lcm_I
            else:
                i[0] = i[0] - lcm_I
                i[1] = i[1] + lcm_I
        
        for j in iter_J:
            if j[0] == 0:
                j[1] = j[1] + 2*lcm_J
            elif j[1] == Pdata.shape[1]:
                j[0] = j[0] - 2*lcm_J
            else:
                j[0] = j[0] - lcm_J
                j[1] = j[1] + lcm_J
        
        for k in iter_K:
            if k[0] == 0:
                k[1] = k[1] + 2*lcm_K
            elif k[1] == Pdata.shape[1]:
                k[0] = k[0] - 2*lcm_K
            else:
                k[0] = k[0] - lcm_K
                k[1] = k[1] + lcm_K
            
    return iter_I, iter_J, iter_K

def convolve(supervoxel, kernel, stride):
    
    conv = np.zeros(((supervoxel.shape[0]-kernel.shape[0])/stride + 1, (supervoxel.shape[1]-kernel.shape[1])/stride + 1, (supervoxel.shape[2]-kernel.shape[2])/stride + 1))
    
    for k in range(0,supervoxel.shape[2]-kernel.shape[2]+1,stride):
        for j in range(0,supervoxel.shape[1]-kernel.shape[1]+1,stride):
            for i in range(0,supervoxel.shape[0]-kernel.shape[0]+1,stride):
                
                block = supervoxel[i:i+kernel.shape[0],j:j+kernel.shape[1],k:k+kernel.shape[2]]
                conval = np.sum(np.multiply(block,kernel))
                conv[i/stride,j/stride,k/stride] = conval

    return conv

def runexp(data, kernelT, stride):
    
    kernel = kernelT[1]
    kernelId = kernelT[0]
    print "computing kernel: "+str(kernelId)+'...\n'
    
    core_factors = [4,3,5] # factorization of cores 

    core_shape = np.array(core_factors)
    data_shape = np.array(list(data.shape))


    corr, idx_core, idx_data = dimdatacorr(core_shape, data_shape)
    Pdata = paddata(data, stride, core_shape, data_shape, idx_core, idx_data)
    iter_I, iter_J, iter_K = getsplits(Pdata, stride, corr, core_shape, do_include=True)

    convolved = list()
    for i in iter_I:
        lst2 = list()
        for j in iter_J:
            lst1 = list()
            for k in iter_K:

                voxel = Pdata[i[0]:i[1], j[0]:j[1], k[0]:k[1]]
                lst1.append(voxel)
                
            pool1 = Pool(processes=len(lst1))
            convolved_K_S = partial(convolve, kernel=kernel, stride=stride)
            result1 = pool1.map(convolved_K_S, lst1)
            pool1.close()
            pool1.join()
            convolved1 = np.array(result1)
            
            shape1_dim0, shape1_dim1, shape1_dim2, shape1_dim3 = convolved1.shape
            convolved1 = convolved1.transpose(1, 2, 3, 0).reshape(shape1_dim1, shape1_dim2, -1)
            
            lst2.append(convolved1)
        
        convolved2 = np.array(lst2)
        shape2_dim0, shape2_dim1, shape2_dim2, shape2_dim3 = convolved2.shape
        convolved2 = convolved2.transpose(1, 2, 0, 3).reshape(shape2_dim1, -1, shape2_dim3)
        
        convolved.append(convolved2)
        
    convolved = np.array(convolved)
    shape3_dim0, shape3_dim1, shape3_dim2, shape3_dim3 = convolved.shape
    convolved = convolved.transpose(1, 0, 2, 3).reshape(-1, shape3_dim2, shape3_dim3)
    
    return convolved
   
 
if __name__ == "__main__":
    
    img = nib.load('t1_icbm_normal_1mm_pn3_rf20.mnc')
    data = img.get_data()

    #kernel = np.random.randint(5, size=(3,3))
    sigma = 1.0  # width of kernel
    x = np.arange(-3, 4, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3, 4, 1)
    z = np.arange(-3, 4, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
    
    kernelT = [0, kernel]

    stride = 2 # stride
    
    convolved_3Dimage = runexp(data, kernelT, stride)
    
    print convolved_3Dimage.shape
    