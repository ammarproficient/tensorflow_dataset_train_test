import cv2

file = 'img.jpg'
img = cv2.imread(file)
cv2.imshow('unilever', img)
cv2.waitKey(0)