import cv2

#グレースケールで読み込み
gray_img = cv2.imread("image/NumberTile.png", 0)

#エッジ検出
sobel_img = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=5)
laplacian_img = cv2.Laplacian(gray_img, cv2.CV_32F, 1, 5)
canny_img = cv2.Canny(gray_img, 80, 120)

#書き込み
cv2.imwrite("image/EdgeDetect/sobel.png", sobel_img)
cv2.imwrite("image/EdgeDetect/laplacian.png", laplacian_img)
cv2.imwrite("image/EdgeDetect/canny.png", canny_img)