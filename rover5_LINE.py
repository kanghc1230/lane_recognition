# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함
import cv2  # opencv 사용
import numpy as np


# ROI 지정. (원하는 부분 외의 노이즈를 마스크를 씌워서 잘라냄)  # roi 3번째 인자값은 roi부분 색상,
def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    # 255흰색으로 초기화
    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # 다각형밖을 마스크로채움
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)
    # cv2.imshow('mask',mask)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    # cv2.imshow('ROI', ROI_image)
    # 다각형 내부 이미지만 리턴
    return ROI_image

# 흰색 차선 찾기
def mark_img(img, blue_threshold=200, green_threshold=200, red_threshold=200):
    global mark
    # BGR 제한 값
    bgr_threshold = [blue_threshold, green_threshold, red_threshold
                     ]
    # 200 (흰색쪽)보다 작은값은 rhresholds가 1, 다 [0,0,0]검은색
    # BGR 제한 값보다 작으면 검은색으로 다 칠해버리기, 크면 흰색(원본이미지쪽)
    # R이 200보다작거나 or G가 200보다 작거나, B가 200보다 작거나
    thresholds = (image[:, :, 0] < bgr_threshold[0]) \
                 | (image[:, :, 1] < bgr_threshold[1]) \
                 | (image[:, :, 2] < bgr_threshold[2])
    mark[thresholds] = [0, 0, 0] # mark 부분 검은색으로
    # 검은색칠하고 흰색선만남은 mark 리턴
    return mark


image = cv2.imread('rover5_LINE.png')  # 이미지 읽기
height, width = image.shape[:2]  # 이미지 높이, 너비

# 사다리꼴 모형의 Points
vertices = np.array(
    [[(260, height), (width / 2 - 35, height / 2 + 70), (width / 2 + 75, height / 2 + 70), (width - 230, height)]],
    dtype=np.int32)
vertices2 = np.array(
    [[(390, height), (width / 2 - 55, height / 2 + 130), (width / 2 + 95, height / 2 + 130), (width - 360, height)]],
    dtype=np.int32)
cv2.polylines(image, vertices, True, (0, 255, 0), 2)
cv2.polylines(image, vertices2, True, (0, 255, 255), 2)
# ROI 입력영상, roi할 좌표, 씌울색상(default 흰색 (255,255,255)
roi_img = region_of_interest(image, vertices, (0,0,255) )  # vertices에 정한 점들 기준으로 ROI 이미지 생성

mark = np.copy(roi_img)  # roi_img 복사 (mark에 roi_img와 동일한사이즈 메모리 할당(정의))
mark = mark_img(roi_img)  # 흰색 차선 찾기

# 흰색 차선 검출한 부분을 원본 image에 overlap 하기
color_thresholds = (mark[:,:,0] == 0) & (mark[:,:,1] == 0) & (mark[:,:,2] > 200)
image[color_thresholds] = [0,0,255]

cv2.imshow('roi_white', mark)  # 흰색 차선 추출 결과 출력
cv2.imshow('result', image)  # 이미지 출력
cv2.waitKey(0)

cv2.destroyAllWindows()