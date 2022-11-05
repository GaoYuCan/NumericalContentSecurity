import random
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    lena: np.ndarray = cv.imread('../../res/lena.bmp', cv.IMREAD_GRAYSCALE)
    bit_layers = [np.zeros_like(lena) for _ in range(8)]
    # 分离出 8 个位平面
    for i in range(8):
        bit_layers[i] = ((lena >> i) & 1) << 7
        cv.imshow(f'L{i + 1}', bit_layers[i])
    # 修改 1 - 3，嵌入随机水印
    W, H = lena.shape
    lena_water = np.array(lena)
    for x in range(W):
        for y in range(H):
            lena_water[x, y] &= 0b11111000
            lena_water[x, y] |= random.randint(0b000, 0b111)
    cv.imshow(f'S', lena)  # 原图
    cv.imshow(f'W', lena_water)  # 嵌入随机水印的图
    cv.waitKey()
