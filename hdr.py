import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import textwrap

NUM_IMG = 10
NUM_PT = 100
L = 1

def progress(cur, total):
    print('%.2f%%'%(100 * cur / total), end='\r')
    
def read_imgs_and_times():
    times = np.array([0.06, 0.13, 0.25, 0.5, 1, 1.6, 3.2, 6, 13, 25], dtype=np.float32)
    log_times = np.log(times)
    filenames = ['img0.jpg',
                 'img1.jpg',
                 'img2.jpg',
                 'img3.jpg',
                 'img4.jpg',
                 'img5.jpg',
                 'img6.jpg',
                 'img7.jpg',
                 'img8.jpg',
                 'img9.jpg']
    imgs = []
    for filename in filenames:
        img = cv2.imread('source images/' + filename)
        imgs.append(img)
    return imgs, log_times

def align(imgs):
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(imgs, imgs)
    return imgs

def is_not_flat(img, i, j, k_size, thres):
    neighbors = img[i - k_size // 2: i + k_size // 2 + 1, j - k_size // 2: j + k_size // 2 + 1]
    diff = sum(sum(abs(neighbors - np.full_like(neighbors, img[i, j]))))
    if diff <= thres:
        return False
    else:
        return True

def is_not_ascending(imgs, i, j):
    for k in range(NUM_IMG - 1):
        if imgs[k][i][j] > imgs[k + 1][i][j]:
            return True
    return False

def pick_points(imgs):
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]
    pts = []
    for i in range(NUM_PT):
        y = random.randint(50, height - 50)
        x = random.randint(50, width - 50)
        while is_not_flat(imgs[6], y, x, 5, 100) or is_not_ascending(imgs, y, x):
            y = random.randint(50, height - 50)
            x = random.randint(50, width - 50)
        pts.append([y, x])
    pts = np.array(pts)
    return pts

def w(z):
    if z == 0 or z == 255:
        return 0.01
    elif z < 30:
        return z / 30
    elif z < 225:
        return 1
    else:
        return 1 + (225 - z) / 30

def reconstruct_g(imgs,log_times,pts):
    A = np.zeros((NUM_IMG * NUM_PT + 255, 256 + NUM_PT), dtype=np.float32)
    B = np.zeros((NUM_IMG * NUM_PT + 255, 1), dtype=np.float32)
    for i in range(NUM_PT):
        for j in range(NUM_IMG):
            weight = w(imgs[j][pts[i][0]][pts[i][1]])
            A[i * NUM_IMG + j][imgs[j][pts[i][0]][pts[i][1]]] = weight
            A[i * NUM_IMG + j][256 + i] = -1 * weight
            B[i * NUM_IMG + j][0] = weight * log_times[j]
    A[NUM_IMG * NUM_PT][127] = 1
    for i in range(1, 255):
        A[NUM_IMG * NUM_PT + i][i - 1] = L * w(i - 1)
        A[NUM_IMG * NUM_PT + i][i] = L * (-2) * w(i)
        A[NUM_IMG * NUM_PT + i][i+1] = L * w(i + 1)
    x = np.linalg.lstsq(A, B, rcond=None)[0]
    g = np.squeeze(x[0: 256])
    return g

def compute_radiance_map(imgs, log_times, g):
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]
    rad_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            progress(i * width + j, width * height)
            sum_w = 0
            for k in range(NUM_IMG):
                sum_w += w(imgs[k][i][j])
                rad_map[i][j] += w(imgs[k][i][j]) * (g[imgs[k][i][j]] - log_times[k])
            if sum_w == 0:
                rad_map[i][j] = 0
            else:
                rad_map[i][j] /= sum_w
    rad_map = np.exp(rad_map)
    return rad_map

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", type=int,
                        help="0: not using saved radiance map\n"
                             "1: using saved radiance map\n")
    parser.add_argument("--algo", type=int,
                        help="0: using Drago algorithm for tone mapping\n"
                             "1: using Mantiuk algorithm for tone mapping\n"
                             "2: using Reinhard algorithm for tone mapping\n")
    args = parser.parse_args()
    mode = args.mode
    algo = args.algo
    
    if mode == 0 :
        print("loading images")
        imgs,log_times = read_imgs_and_times()
        imgs = align(imgs)
        imgs_b = []
        imgs_g = []
        imgs_r = []
        for i in range(NUM_IMG):
            (b,g,r) = cv2.split(imgs[i])
            imgs_b.append(b.astype(np.int32))
            imgs_g.append(g.astype(np.int32))
            imgs_r.append(r.astype(np.int32))

        pts_b = pick_points(imgs_b)
        pts_g = pick_points(imgs_g)
        pts_r = pick_points(imgs_r)

        print('reconstructing g function')
        g_b = reconstruct_g(imgs_b,log_times,pts_b)
        g_g = reconstruct_g(imgs_g,log_times,pts_g)
        g_r = reconstruct_g(imgs_r,log_times,pts_r)

        print('computing radiance map')
        rad_map_b = compute_radiance_map(imgs_b,log_times,g_b)
        rad_map_g = compute_radiance_map(imgs_g,log_times,g_g)
        rad_map_r = compute_radiance_map(imgs_r,log_times,g_r)
        rad_map = cv2.merge([rad_map_b,rad_map_g,rad_map_r])
        cv2.imwrite('rad_map.hdr',rad_map)
    else:
        print('loading saved radiance map')
        rad_map = cv2.imread('rad_map.hdr', -1)

    print('tone mapping')    
    if algo == 0:
        tonemap = cv2.createTonemapDrago(gamma=2.2)
    elif algo == 1:
        tonemap = cv2.createTonemapMantiuk(gamma=2.2)
    elif algo == 2:
        tonemap = cv2.createTonemapReinhard(gamma=1.25)
    res = tonemap.process(rad_map.copy())
    res = np.clip(res*255, 0, 255).astype('uint8')
    if algo == 0:
        cv2.imwrite('ldr_Drago.jpg', res)
    elif algo == 1:
        cv2.imwrite('ldr_Mantiuk.jpg', res)
    elif algo == 2:
        cv2.imwrite('ldr_Reinhard.jpg', res)

if __name__== "__main__":
    main()
