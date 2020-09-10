import numpy as np
from PIL import Image

# レナの画像読み込み
im_gray = np.array(Image.open('image/lenna512.bmp').convert('L'))

# 閾値設定
threshold1 = 50
threshold2 = 80
threshold3 = 128
threshold4 = 164

# true / falseは 1 / 0で表されるので二値化するときは255掛ければいい
maxval = 255

im_bin1 = (im_gray > threshold1) * maxval
im_bin2 = (im_gray > threshold2) * maxval
im_bin3 = (im_gray > threshold3) * maxval
im_bin4 = (im_gray > threshold4) * maxval

Image.fromarray(np.uint8(im_bin1)).save('image/LennaBinarize/LennaBinarize50.png')
Image.fromarray(np.uint8(im_bin2)).save('image/LennaBinarize/LennaBinarize80.png')
Image.fromarray(np.uint8(im_bin3)).save('image/LennaBinarize/LennaBinarize128.png')
Image.fromarray(np.uint8(im_bin4)).save('image/LennaBinarize/LennaBinarize164.png')