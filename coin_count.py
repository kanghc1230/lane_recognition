import sys
import numpy as np
import cv2


# 입력 이미지 불러오기
img_path = 'coins1.jpg'
# 'coins1.jpg' 'coin_custom.jpg'
src = cv2.imread(img_path)

if src is None:
    print('Image open failed!')
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blr = cv2.GaussianBlur(gray, (0, 0), 1) # 가우스블러 고주파노이즈 제거
# cv2.imshow('blr', blr)

# 허프 변환 원 검출
# 입력영상, HOUGH_GRADIENT or HOUGH_GRADIENT_ALT, 배열크기1, 원의 중심점 최소거리,
# 임계값1,2 , 검출할 최소최대 반지름1,2)
circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=150, param2=40, minRadius=20, maxRadius=80)
print(circles)

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
        radius = int(radius)

        crop = dst[y1:y2, x1:x2, :]
        ch, cw = crop.shape[:2]

        # 동전 영역에 대한 ROI 마스크 영상 생성
        mask = np.zeros((ch, cw), np.uint8) # 동전1개마스크
        cv2.circle(mask, (cw//2, ch//2), radius, 255, -1) # 정수로나눔

        # hue는 1바퀴 0~179
        # 동전 영역 Hue 색 성분을 +40 시프트하고, Hue 평균을 계산
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, _, _ = cv2.split(hsv)
        hue_shift = (hue + 40) % 180 # hue의 평균
        mean_of_hue = cv2.mean(hue_shift, mask)[0] # mean함수는 마스크 내의 부분만 계산

        # Hue 평균이 90보다 작으면 10원, 90보다 크면 100원으로 간주
        won = 100
        if mean_of_hue < 90:
            won = 10

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
