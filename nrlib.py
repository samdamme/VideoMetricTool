"""
import glob, os
import cv2
import csv
import numpy as np
from matplotlib import pyplot as plt
from math import exp
from multiprocessing import Process
import pandas as pd
"""

import cv2
import numpy as np

def Sobel(lum):
    Gx = cv2.Sobel(lum,cv2.CV_64F,0,1,ksize=3)
    Gy = cv2.Sobel(lum,cv2.CV_64F,1,0,ksize=3)
    G = np.sqrt(np.power(Gx,2)+np.power(Gy,2))
    return [Gx, Gy, G]

def calc_si(lum):
    sobel = Sobel(lum)[2]
    return np.std(sobel)

def calc_ti(pvs_lum, lum):
    return np.std(pvs_lum-lum)

def calc_mi(pvs_lum, lum):
    return np.sqrt(np.sum((lum-pvs_lum)**2))

def calculate_D(lum, ax):
    D_p = np.roll(lum, -1, axis=ax)
    D_m = np.roll(lum, 1, axis=ax)
    D = np.abs(D_p - D_m)
    D_mean = np.mean(D)
    return [D, D_mean]

def dim_blur(lum, ax):
    [D, D_mean] = calculate_D(lum, ax)
    C = D
    C[D <= D_mean] = 0
    C_p = np.roll(C, -1, axis=ax)
    C_m = np.roll(C, 1, axis=ax)
    E = np.logical_and(np.greater(C, C_p), np.greater(C, C_m))
    A = D/2
    BR = np.divide(np.abs(lum-A), A, where=A!=0)
    BR[A == 0] = 1.0
    return [E, BR]

def _calc_blur_base(lum, th=0.1):
    [E_h, BR_h] = dim_blur(lum, 0)
    [E_v, BR_v] = dim_blur(lum, 1)
    E = np.logical_or(E_v, E_h)
    BR = np.maximum(BR_h, BR_v)
    B = BR < th
    B_sum = np.sum(B)
    E_sum = np.sum(E)
    return B_sum, BR, B, E_sum, E

def calc_blur_ratio(lum):
    _, _, B, E_sum, E = _calc_blur_base(lum)
    if(E_sum == 0):
        return 0.0
    return np.sum(B)/np.sum(E)

def calc_blur(lum):
    B_sum, BR, B, _, _ = _calc_blur_base(lum)
    if(B_sum == 0):
        return 0.0
    return np.sum(BR)/np.sum(B)

def avg_filter(lum, ksize):
    return cv2.blur(lum, (ksize, ksize))

def _calc_noise_base(lum):
    g = avg_filter(lum, 3)
    [D_h, D_h_mean] = calculate_D(g, 0)
    [D_v, D_v_mean] = calculate_D(g, 1)
    D = np.maximum(D_h, D_v)
    N_c = D
    N_c[np.logical_and(D_h > D_h_mean, D_v > D_v_mean)] = 0
    N_c_mean = np.mean(N_c)
    N = N_c
    mask = N_c > N_c_mean
    N[np.logical_not(mask)] = 0
    N_count = np.sum(mask)
    return N_count, N

def calc_noise_ratio(lum):
    N_count, _ = _calc_noise_base(lum)
    return N_count/(lum.shape[0]*lum.shape[1])

def calc_noise(lum):
    N_count, N = _calc_noise_base(lum)
    if(N_count == 0):
        return 0.0
    return np.sum(N)/N_count

def blocksum(block, bs, mode):
    sets = []
    if(mode==0):
        sets.append(block[:,0])
        sets.append(block[:,bs-1])
    elif(mode==1):
        sets.append(block[0,:])
        sets.append(block[bs-1,:])
    else:
        sets.append(block[1,1:bs-2])
        sets.append(block[bs-2,1:bs-2])
        sets.append(block[2:bs-3,1])
        sets.append(block[2:bs-3,bs-2])
    sums = [np.sum(np.abs(x)) for x in sets]
    return np.sum(sums)

def calc_blockiness(lum, bs=8, k=2.3):
    M = lum.shape[0]
    N = lum.shape[1]
    [Dx, Dy, D] = Sobel(lum)
    max_Dx = np.max(Dx)
    max_Dy = np.max(Dy)
    max_D = np.max(D)
    Ls = []
    blk_range_x = int(M/bs) + 1
    blk_range_y = int(N/bs) + 1
    Bx_div = 2*bs*max_Dx
    By_div = 2*bs*max_Dy
    #I_div = 4*(bs-1)*max_D
    I_div = 4*(bs-3)*max_D

    for sx in range(0, blk_range_x):
        for sy in range(0, blk_range_y):
            x = sx*bs
            y = sy*bs
            if(x<M and y < N):
                block_x = Dx[x:min(x+bs,M),y:min(y+bs,N)]
                block_y = Dy[x:min(x+bs,M),y:min(y+bs,N)]
                block_i = D[x:min(x+bs,M),y:min(y+bs,N)]
                B_x = blocksum(block_x, bs, 0)/Bx_div
                B_y = blocksum(block_y, bs, 1)/By_div
                B = max(B_x, B_y)
                I = blocksum(block_i, bs, 2)/I_div
                if(B==0 and I==0):
                    L = 0.0
                else:
                    L = 2*abs(B**k - I**k)/abs(B**k + I**k)
                Ls.append(L)
    return np.mean(Ls)

"""
def mu(x):
    # px = 5, py = 0.5, q = 0.25
    # a = py/(px**(q*px/py)) = 0.009, b = q*px/py = 2.5, c = 4*q/d = 1, d = 2(1-py) = 1

    return sigmoid(0.009, 2.5, 1.0, 1.0, 5.0, float(x))


def tau(x, res):
    # e = 2,54 for HD and 1.54 for SD
    # px = 0.12/e = 0.047 for HD and 0.078 for SD, py = 0.05, q = 1.5*e = 3.81 for HD and 2.31 for SD
        # SD: a = py/(px**(q*px/py)) = 500, b = q*px/py = 3.6036, c = 4*q/d = 4.863, d = 2(1-py) = 1.9
        # HD: a = py/(px**(q*px/py)) = 2849.165, b = q*px/py = 3.5814, c = 4*q/d = 8.0211, d = 2(1-py) = 1.9

    if(res=="SD"):
        return sigmoid(500.0, 3.6036, 4.863, 1.9, 0.078, float(x))
    else:
        return sigmoid(2849.165, 3.5814, 8.0211, 1.9, 0.047, float(x))

def sigmoid(a,b,c,d,px,x):
    if(x <= px):
        return a*(x**b)
    else:
        return d/(1.0+exp(-c*(x-px))) + 1.0 - d

def calc_jerkiness(mis, T, delta, res):
    return (float(delta)*tau(delta, res)/float(T))*np.sum([mu(x) for x in mis])

def analyse_stream(path, path_pvs=None):
    blurmean = []
    blurrat = []
    noisemean = []
    noiserat = []
    block = []
    #si_stds = []
    lum = np.fromfile(path, dtype=np.uint8, count=1920*1080).reshape((1080,1920))
    pvs_lum = np.fromfile(path_pvs, dtype=np.uint8, count=1920*1080).reshape((1080,1920))

    #[bm, br] = blur(lum, 0.1)
    #[nm, nr] = noise(lum)
    #bl = blockiness(lum, 8, 2.3)
    ti = ti_std(lum, pvs_lum)
    mmi = mi_rmse(lum, pvs_lum)

    #return [bm, br, nm, nr, bl, si]
    return ti, mmi

def proc(n):
    scene = min(int(n/3) + 1,4)
    if(n==12):
        camera = 4
    else:
        camera = n % 3 + 1

    print("PROCESS "+str(n)+": scene "+str(scene)+", camera "+str(camera))


    PATH_EXT = "scene_"+str(scene)+"/camera_"+str(camera)+"/"
    PATH_IN = "./frames/"+PATH_EXT
    PATH_OUT = "./NR_per_frame/scene_"+str(scene)+"_camera_"+str(camera)+"_ti_mmi.dat"
    PATH_SAVE = "./NR_per_frame/SAVE/scene_"+str(scene)+"_camera_"+str(camera)+"_ti_mmi.dat"

    try:
        saved_file = pd.read_csv(PATH_SAVE)
        N = len(saved_file)
    except:
        N = 0

    framerate = 30
    resolution = "HD"
    #nrs = ["BLUR", "BLURRAT", "NOISE", "NOISERAT", "BLOCK", "SI"]
    fs = []
    bms = []
    brs = []
    nms = []
    nrs = []
    bls = []
    tis = []
    mmis = []
    L = len(list(os.scandir(PATH_IN)))

    for i,f in enumerate(os.scandir(PATH_IN)):
        if(i>N):
            print("PROCESS "+str(n)+": frame "+str(i+1)+"/"+str(L))
            #[bm, br, nm, nr, bl, si] = analyse_stream(f.path)
            ti, mmi = analyse_stream(f.path, pvs_path)
            fs.append(f.path)
            bms.append(bm)
            brs.append(br)
            nms.append(nm)
            nrs.append(nr)
            bls.append(bl)
            tis.append(ti)
            mmis.append(mmi)

            if(i%50==0 or i==L-1):
                data = pd.DataFrame()
                data["frame"] = fs
                data["blur"] = bms
                data["blur_ratio"] = brs
                data["noise"] = nms
                data["noise_ratio"] = nrs
                data["blockiness"] = bls
                data["ti"] = tis
                data["mmi"] = mmis
                print("PROCESS "+str(n)+": Saving to csv...")
                data.index = np.arange(N, N + len(data))
                data.to_csv(PATH_OUT)
        elif(i==N):
            tis.append(0)
            mmis.append(0)
            fs.append(f.path)
        pvs_path = f.path

def main():
    processes = []
    p_num = 13


    for n in range(p_num):
        p = Process(target=proc, args=(n,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
"""
