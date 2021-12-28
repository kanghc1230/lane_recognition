import sys
import numpy as np
import cv2


# 입력 이미지 불러오기
img_path = 'coin_custom.jpg'
# 'coins1.jpg' 'coin_custom.jpg'
src = cv2.imread(img_path)

if src is None:
    print('Image open failed!')
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
blr = cv2.GaussianBlur(gray, (0, 0), 1) # 가우스블러 고주파노이즈 제거
cv2.imshow('gausblr', blr)

# 허프 변환 원 검출
# 입력영상, HOUGH_GRADIENT or HOUGH_GRADIENT_ALT, 배열크기1, 원의 중심점 최소거리,
# 임계값1,2 , 검출할 최소최대 반지름1,2)
circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=150, param2=40, minRadius=20, maxRadius=80)
#print(circles)

# 원 검출 결과 및 동전 금액 출력
sum_of_money = 0
# 깊은 복사
dst = src.copy()
if circles is not None:
    for i in range(circles.shape[1]):
        cx, cy, radius = circles[0][i]
        cv2.circle(dst, (cx, cy), radius, (0, 0, 255), 2, cv2.LINE_AA) # 원그리기

        # 동전 영역 부분 영상 추출
        x1 = int(cx - radius)
        y1 = int(cy - radius)
        x2 = int(cx + radius)
        y2 = int(cy + radius)
        radius = int(radius) # 반지름
        #print("radius : {}".format(radius))

        crop = dst[y1:y2, x1:x2, :]
        ch, cw = crop.shape[:2]

        # 동전 영역에 대한 ROI 마스크 영상 생성
        mask = np.zeros((ch, cw), np.uint8)
        cv2.circle(mask, (cw//2, ch//2), radius, 255, -1)
        #cv2.imshow('mask', mask)

        # hue는 1바퀴 0~179
        # 동전 영역 Hue 색 성분을 +40 시프트하고, Hue 평균을 계산
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, _, _ = cv2.split(hsv)
        hue_shift = (hue + 40) % 180 # hue +40밝게 쉬프트
        #cv2.imshow('hue_shift',hue_shift)
        mean_of_hue = cv2.mean(hue_shift, mask)[0] # meanStdDev() 평균과 표준편차 계산
        # mean 매개변수
        # src	결과를 Scalar_ 에 저장할 수 있도록 1~4개의 채널을 가져야 하는 입력 배열입니다 .
        # 평균	출력 매개변수: 계산된 평균 값.
        # 표준 데브	출력 매개변수: 계산된 표준 편차.
        # 마스크	선택적 작업 마스크.
        
        #print("mean_of_hue : {}".format(mean_of_hue))

        #        hue          반지름
        # 10원 55 53         55.7 56.3
        # 50원 44.8 49.3     51.9,53.6
        # 100원 22.7 34.2    58.9 59.2
        # 500원 63.6 ,56.2   62.5, 62.5

        # radius 반지름크기 50이상은 500원, 사이 100,10원은 색상 hue값 구별, 45.5이하는 50원,
        # 색이 hue 평균40이하(흑백동) 40이상(주황10원)
        # radius > 50
        won = 500  # radius 54 hue 56
        if 45.5 < radius < 50: # 100won,10won
            if mean_of_hue < 40:
                won = 100
            else:
                won = 10
        elif radius <= 45.5: # 50won
            won = 50
        sum_of_money += won

        cv2.putText(crop, str(won), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 2, cv2.LINE_AA)

# sum_of_money 변수에 있는값 글씨 쓰기
cv2.putText(dst, str(sum_of_money) + ' won', (40, 80),
            cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()
