import numpy as np
import cv2

img = np.zeros((720, 1280, 3), dtype=np.uint8)

for x in range(1280//32):
    cv2.line(img,(x*32, 0),(x*32, 720),color=(128,128,128))

for y in range(720//32+1):
    cv2.line(img,(0, y*32),(1280, y*32),color=(128,128,128))

bx=13
by=13
for i in range(4,16):
    cv2.rectangle(img, ((bx+i)*32, by*32), ((bx+i)*32+64, by*32+64), thickness=-1, color=(0,0,0))
    cv2.rectangle(img, ((bx+i)*32, by*32), ((bx+i)*32+64, by*32+64), thickness=2, color=(0,0,i*16-1))

cv2.imwrite('grid.png', img)
cv2.imshow('grid',img)
cv2.waitKey()


