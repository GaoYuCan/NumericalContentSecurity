import cv2 as cv
import numpy as np


def arnold_3d_encrypt(m: np.ndarray, a: int, b: int, c: int, d: int) -> np.ndarray:
    W, H, _ = m.shape
    mat = np.array([
        [1, 0, a],
        [b * c, 1, a * b * c + c],
        [b * c * d + b, d, a * b * c * d + a * b + c * d + 1]
    ])
    e = np.zeros_like(m)
    for x in range(W):
        for y in range(H):
            e[x, y, 0:3] = np.matmul(mat, m[x, y, 0:3])
    return e


def arnold_3d_decrypt(e: np.ndarray, a: int, b: int, c: int, d: int) -> np.ndarray:
    W, H, RGB = e.shape
    mat = np.array([
        [1 + a * b, a * d, -a],
        [0, 1 + c * d, -c],
        [-b, -d, 1]
    ])
    m = np.zeros_like(e)
    for x in range(W):
        for y in range(H):
            m[x, y, 0:3] = np.matmul(mat, e[x, y, 0:3])
    return m


if __name__ == '__main__':
    a, b, c, d = 1, 1, 1, 1
    R = 4
    mandril: np.ndarray = cv.imread(r'../../res/mandril.png', cv.IMREAD_COLOR)
    cv.imshow("Display Src", mandril)
    mandril_encrypted = mandril
    for i in range(R):
        mandril_encrypted = arnold_3d_encrypt(mandril_encrypted, a, b, c, d)
    cv.imshow("Display Encrypted", mandril_encrypted)

    mandril_decrypted = mandril_encrypted
    for i in range(R):
        mandril_decrypted = arnold_3d_decrypt(mandril_decrypted, a, b, c, d)
    cv.imshow("Display Decrypted", mandril_decrypted)
    cv.waitKey()
