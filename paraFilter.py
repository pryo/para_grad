import nibabel as nib

import numpy as np

from multiprocessing import Process, Manager,Pool

import conv3Dp

def outsize(wsize, ksize, stride):
    return int((wsize - ksize) / stride + 1)


def conv3d(cube, kernelT, stride):
    kernel = kernelT[1]
    kernelId = kernelT[0]
    print "computing kernel: "+str(kernelId)+'...\n'
    outx = outsize(cube.shape[0], kernel.shape[0], stride)
    outy = outsize(cube.shape[1], kernel.shape[1], stride)
    outz = outsize(cube.shape[2], kernel.shape[2], stride)
    outputShape = (outx, outy, outz)

    output = np.zeros(shape=outputShape)
    # print(output.shape)
    for z in range(0, cube.shape[2] - kernel.shape[2] + 1, stride):
        for y in range(0, cube.shape[1] - kernel.shape[1] + 1, stride):
            for x in range(0, cube.shape[0] - kernel.shape[0] + 1, stride):
                cubePatch = cube[x:x + kernel.shape[0], y:y + kernel.shape[1], z:z + kernel.shape[2]]
                # print([x:x+kernel.shape[0],y+kernel.shape[1],z+kernel.shape[2]])
                # print(cubePatch.shape)
                product = np.multiply(cubePatch, kernel)
                outputx = x // stride
                outputy = y // stride
                outputz = z // stride
                # print([outputx,outputy,outputz])
                # print(product.shape)
                output[outputx, outputy, outputz] = np.sum(product)
    return output

def convSera(data,kernels,stride):
    results=[]
    for kernelT in kernels:
        results.append(conv3d(data,kernelT,stride))
    return results

def convPara(data,kernels,proc,stride, mycon):


    mgr = Manager()
    sharedParams = mgr.dict()
    sharedParams['data'] = data
    sharedParams['proc']=proc
    sharedParams['stride']=stride
    pool= Pool(processes=proc)
    if mycon:
        results = [pool.apply_async(conv3Dp.runexp,(sharedParams['data'],kernel,sharedParams['stride'])) for kernel in kernels]
    else:
        results = [pool.apply_async(conv3d,(sharedParams['data'],kernel,sharedParams['stride'])) for kernel in kernels]
    pool.close()
    pool.join()
    return results

if __name__=='__main__':
    img = nib.load('t1_icbm_normal_1mm_pn3_rf20.mnc')
    data = img.get_data()
    stride = 4
    processes=4
    kernelNumber=4
    #coreFactor=[2,2,2]
    from scipy import signal
    kernels=[]
    # first build the smoothing kernel
    sigma = 1.0  # width of kernel
    x = np.arange(-3, 4, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3, 4, 1)
    z = np.arange(-3, 4, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    kern = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
    for i in range(kernelNumber):
        kernels.append([i,kern])

    results = convPara(data,kernels,processes,stride)
    print len(results)