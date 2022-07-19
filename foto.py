import cv2

gambar = cv2.imread('baboon.jpg')
original = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

print(gambar)

cv2.imshow('gambar RGB',gambar)
cv2.imshow('original',original)



cv2.waitKey(0)
cv2.destroyAllWindows()