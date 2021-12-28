import sys
import numpy as np
import cv2


src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()
#(입력영상, 출력영상데이터타입-1, x방향미분차수, y방향 미분차수, 연산에 추가더할값delta=)
dx = cv2.Scharr(src, -1, 1, 0, delta=128)
dy = cv2.Scharr(src, -1, 0, 1, delta=128)
#dxdy = cv2.Sobel(src, -1, 1, 1, delta=128) #dx는검은색 dy가 흰색인것이 더해져서 희미해져버림

cv2.imshow('src', src)
cv2.imshow('dx', dx)
cv2.imshow('dy', dy)
#cv2.imshow('dxdy', dxdy)
cv2.waitKey()

cv2.destroyAllWindows()
