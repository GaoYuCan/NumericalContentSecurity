from math import cos, sin
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    lena: np.ndarray = cv.imread('../../res/lena.bmp', cv.IMREAD_GRAYSCALE)
    lena_f = np.fft.fft2(lena)
    W, H = lena.shape
    lena_f_real = np.zeros((W, H), dtype=np.complex_)
    lena_f_imag = np.zeros((W, H), dtype=np.complex_)

    cv.imshow('Src', lena)

    for x in range(W):
        for y in range(H):
            lena_f_real[x, y] = complex(real=lena_f[x, y].real, imag=0)
            lena_f_imag[x, y] = complex(real=0, imag=lena_f[x, y].imag)
    cv.imshow('Real', np.array(abs(lena_f_real), dtype=np.uint8))
    cv.imshow('Imag', np.array(abs(lena_f_imag), dtype=np.uint8))
    # 幅度谱
    lena_m = np.abs(lena_f)
    cv.imshow('Magnitude', np.array(lena_m, dtype=np.uint8))
    # 相位谱
    lena_p = np.angle(lena_f)
    cv.imshow('Phase', np.array(lena_p, dtype=np.uint8))
    # 幅度谱重构图像
    lena_m_re = np.zeros((W, H), dtype=np.complex_)
    for x in range(W):
        for y in range(H):
            lena_m_re[x, y] = complex(real=cos(150) * lena_m[x, y], imag=sin(150) * lena_m[x, y])
    lena_m_re = np.fft.ifft2(lena_m_re)
    cv.imshow('ReMagnitude', np.array(np.real(lena_m_re), dtype=np.uint8))
    # 相位谱重构图像
    lena_p_re = np.zeros((W, H), dtype=np.complex_)
    for x in range(W):
        for y in range(H):
            lena_p_re[x, y] = complex(real=cos(lena_p[x, y]) * 200, imag=sin(lena_p[x, y]) * 200)
    lena_p_re = np.fft.ifft2(lena_p_re)
    cv.imshow('RePhase', np.array(np.real(lena_p_re), dtype=np.uint8))
    cv.waitKey()
