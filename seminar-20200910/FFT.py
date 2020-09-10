import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像読み込み
img_pens = cv2.imread("image/Pens.png", 0)
img_mandrill = cv2.imread("image/mandrill512.bmp", 0)

# フーリエ変換
fft_pens = np.fft.fft2(img_pens)
fft_mandrill = np.fft.fft2(img_mandrill)

# シフト操作
fft_shift_pens = np.fft.fftshift(fft_pens)
fft_shift_mandrill = np.fft.fftshift(fft_mandrill)

# 逆シフト操作
fft_shift_ishift_pens = np.fft.ifftshift(fft_shift_pens)
fft_shift_ishift_mandrill = np.fft.ifftshift(fft_shift_mandrill)

# 逆フーリエ変換
fft_shift_ishift_ifft_pens = np.fft.ifft2(fft_shift_ishift_pens)
fft_shift_ishift_ifft_mandrill = np.fft.ifft2(fft_shift_ishift_mandrill)

mag_pens = 20 * np.log(np.abs(fft_shift_pens))
mag_mandrill = 20 * np.log(np.abs(fft_shift_mandrill))
imag_pens = 20 * np.log(np.abs(fft_shift_ishift_ifft_pens))
imag_mandrill = 20 * np.log(np.abs(fft_shift_ishift_ifft_mandrill))

cv2.imwrite("image/FFT/fft_shift_pens.png", mag_pens)
cv2.imwrite("image/FFT/fft_shift_mandrill.png", mag_mandrill)
cv2.imwrite("image/FFT/fft_shift_ishift_ifft_pens.png", imag_pens)
cv2.imwrite("image/FFT/fft_shift_ishift_ifft_mandrill.png", imag_mandrill)