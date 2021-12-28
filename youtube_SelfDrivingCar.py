# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random
import os, sys
import pafy


fit_result, l_fit_result, r_fit_result, L_lane, R_lane = [], [], [], [], []

# 유튜브 영상 정보 읽어오기 (url)
def video_info(video):
    print("video title : {}".format(video.title))  # 제목
    print("video rating : {}".format(video.rating))  # 평점
    print("video viewcount : {}".format(video.viewcount))  # 조회수
    print("video author : {}".format(video.author))  # 저작권자
    print("video length : {}".format(video.length))  # 길이
    print("video duration : {}".format(video.duration))  # 길이
    print("video likes : {}".format(video.likes))  # 좋아요
    print("video dislikes : {}".format(video.dislikes))  # 싫어요


def grayscale(img):
    """cv 흑백 변환 함수"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """cv 캐니 변환 함수"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """cv 가우스 블러 변환 함수"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# ROI
def region_of_interest(img, vertices):
    """
    roi : 지정한 부분(다각형) 외 부분에 mask를 적용한다.
    지정한 꼭지점(다각형) 4부분 안쪽은 유지되고, 바깥은 검은색으로 덮어진다.
    """
    # 검은 마스크 이미지 생성
    mask = np.zeros_like(img)
    if len(img.shape) > 2: # shape 3개이상, 컬러(3채널)라면
        channel_count = img.shape[2]  # 채널개수 저장 (3 혹은 가끔 4)
        ignore_mask_color = (255,) * channel_count # 다각형내 칠할값 255흰색으로 세팅(255,255,255)
    else: # shpae 2개이하, 흑백(1채널)이라면
        ignore_mask_color = 255

    # 다각형 내부 픽셀 채우기 (mask 흰색진하게 다채워진 다각형으로나옴)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #cv2.imshow("mask", mask)

    # img와 mask를 비트와이즈 연산해 0이아닌 부분만 전달
    masked_image = cv2.bitwise_and(img, mask)
    #cv2.imshow("masked_image",masked_image)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    이 기능은 원할 때 시작점으로 사용할 수 있는 기능입니다.
    전체를 매핑하기 위해 감지한 선분의 평균/외삽
    레인의 범위

    선분을 구분하는 것과 같은 것을 생각해 보십시오.
    기울기((y2-y1)/(x2-x1))는 왼쪽의 일부인 세그먼트를 결정합니다.
    선 대 오른쪽 선. 그런 다음 각 위치의 평균을 구할 수 있습니다.
    라인의 상단과 하단을 외삽합니다.

    이 함수는 '색상'과 '두께'로 '선'을 그립니다.
    이미지의 내부에 선이 그려집니다(이미지 변형).
    선을 반투명하게 만들고 싶다면 결합을 생각해보세요
    """
    # 모든 lines에 x1,y1 -> x2,y2
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_circle(img, lines, color=[0, 0, 255]):
    for line in lines:
        cv2.circle(img, (line[0], line[1]), 2, color, -1)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    캐니 이미지를 받아서 거친 실선만 찾아 반환한다.
    """
    # lines
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # line_arr 공간생성 [[0 0 0] [0 0 0] [0 0 0]]
    #draw_lines(line_arr, lines) # 0 ~ lines
    return lines


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img`는 hough_lines()의 출력이며, 선이 그려진 이미지입니다.
    선이 그려진 빈 이미지(전체 검정색)여야 합니다.

    'initial_img'는 처리 전의 이미지여야 합니다.

    결과 이미지는 다음과 같이 계산됩니다.

    initial_img * α + img * β + λ
    참고: initial_img와 img는 모양이 같아야 합니다!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def Collect_points(lines):
    # reshape [:4] to [:2]
    interp = lines.reshape(lines.shape[0] * 2, 2)
    # interpolation & collecting points for RANSAC
    for line in lines:
        if np.abs(line[3] - line[1]) > 5:
            tmp = np.abs(line[3] - line[1])
            a = line[0];
            b = line[1];
            c = line[2];
            d = line[3]
            slope = (line[2] - line[0]) / (line[3] - line[1])
            for m in range(0, tmp, 5):
                if slope > 0:
                    new_point = np.array([[int(a + m * slope), int(b + m)]])
                    interp = np.concatenate((interp, new_point), axis=0)
                elif slope < 0:
                    new_point = np.array([[int(a - m * slope), int(b - m)]])
                    interp = np.concatenate((interp, new_point), axis=0)
    return interp


def get_random_samples(lines):
    one = random.choice(lines)
    two = random.choice(lines)
    if (two[0] == one[0]):  # extract again if values are overlapped
        while two[0] == one[0]:
            two = random.choice(lines)
    one, two = one.reshape(1, 2), two.reshape(1, 2)
    three = np.concatenate((one, two), axis=1)
    three = three.squeeze()
    return three


def compute_model_parameter(line):
    # y = mx+n
    m = (line[3] - line[1]) / (line[2] - line[0])
    n = line[1] - m * line[0]
    # ax+by+c = 0
    a, b, c = m, -1, n
    par = np.array([a, b, c])
    return par


def compute_distance(par, point):
    # distance between line & point

    return np.abs(par[0] * point[:, 0] + par[1] * point[:, 1] + par[2]) / np.sqrt(par[0] ** 2 + par[1] ** 2)


def model_verification(par, lines):
    # calculate distance
    distance = compute_distance(par, lines)
    # total sum of distance between random line and sample points
    sum_dist = distance.sum(axis=0)
    # average
    avg_dist = sum_dist / len(lines)

    return avg_dist


def draw_extrapolate_line(img, par, color=(0, 0, 255), thickness=2):
    x1, y1 = int(-par[1] / par[0] * img.shape[0] - par[2] / par[0]), int(img.shape[0])
    x2, y2 = int(-par[1] / par[0] * (img.shape[0] / 2 + 100) - par[2] / par[0]), int(img.shape[0] / 2 + 100)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def get_fitline(img, f_lines):
    rows, cols = img.shape[:2]
    output = cv2.fitLine(f_lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
    result = [x1, y1, x2, y2]

    return result


# 정해진 결과에 보라색 선긋기
def draw_fitline(img, result_l, result_r, color=(255, 0, 255), thickness=10):
    # draw fitting line
    lane = np.zeros_like(img)
    cv2.line(lane, (int(result_l[0]), int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
    cv2.line(lane, (int(result_r[0]), int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
    # add original image & extracted lane lines
    final = weighted_img(lane, img, 1, 0.5)
    return final


def erase_outliers(par, lines):
    # distance between best line and sample points
    distance = compute_distance(par, lines)

    # filtered_dist = distance[distance<15]
    filtered_lines = lines[distance < 13, :]
    return filtered_lines


def smoothing(lines, pre_frame):
    # collect frames & print average line
    lines = np.squeeze(lines)
    avg_line = np.array([0, 0, 0, 0])

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line

# 써클이 범위를벗어나는 값 지우는 함수
def ransac_line_fitting(img, lines, min=100):
    global fit_result, l_fit_result, r_fit_result
    best_line = np.array([0, 0, 0])
    if (len(lines) != 0):
        for i in range(30):
            sample = get_random_samples(lines)
            parameter = compute_model_parameter(sample)
            cost = model_verification(parameter, lines)
            if cost < min:  # update best_line
                min = cost
                best_line = parameter
            if min < 3: break
        # erase outliers based on best line
        filtered_lines = erase_outliers(best_line, lines)
        fit_result = get_fitline(img, filtered_lines)
    else:
        if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0:
            l_fit_result = fit_result
            return l_fit_result
        else:
            r_fit_result = fit_result
            return r_fit_result

    if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0:
        l_fit_result = fit_result
        return l_fit_result
    else:
        r_fit_result = fit_result
        return r_fit_result


def detect_lanes_img(img):
    height, width = img.shape[:2]

    # ROI할 부분 vertices 지정
    # vertices = np.array(
    #     [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
    #     dtype=np.int32)
    vertices = np.array(
        [[(170, height), (width / 2 - 45, height / 2 + 20), (width / 2 + 85, height / 2 + 20), (width - 20, height)]],
        dtype=np.int32)
    # ROI 적용
    ROI_img = region_of_interest(img, vertices) # ROI_img 필요부분외 roi마스크로 처리
    cv2.polylines(img, vertices, True, (0, 255, 0), 2)
    
    # 그레이 스케일 적용
    # g_img = grayscale(img)

    # 가우시안 블러 적용 (노이즈제거)
    blur_img = gaussian_blur(ROI_img, 3) 

    # 캐니 엣지 적용 (제외,노이즈제거된것)
    canny_img = canny(blur_img, 70, 210)
    # vertices2 = np.array(
    #     [[(52, height), (width / 2 - 43, height / 2 + 62), (width / 2 + 43, height / 2 + 62), (width - 52, height)]],
    #     dtype=np.int32)
    vertices2 = np.array(
        [[(170, height), (width / 2 - 45, height / 2 + 20), (width / 2 + 85, height / 2 + 20), (width - 20, height)]],
        dtype=np.int32)
    # 캐니 이미지에 roi 적용
    canny_img = region_of_interest(canny_img, vertices2)

    # cv2.polylines(img, vertices2, True, (0, 0, 255), 1)

    # 허프 변환으로 직선 찾기 (입력영상(edge된), 1직선array, 1픽셀해상도, 계산할(선회전) 각도, 라인교차수, 검출 직선길이, 검출 점사이거리)
    line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 10, 20)

    # 라인에 선긋기
    #draw_lines(img, line_arr, thickness=2)

    # squeeze() 불필요한
    line_arr = np.squeeze(line_arr)
    # 기울기 분리. arctan2
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # 수평경사선 무시
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    # 수직경사선 무시
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    # print(line_arr.shape,'  ',L_lines.shape,'  ',R_lines.shape)

    # interpolation & collecting points for RANSAC 랜덤표본합의. 표본끼리 라인을 따는것
    L_interp = Collect_points(L_lines)
    R_interp = Collect_points(R_lines)

    draw_circle(img,L_interp,(255,255,0)) 
    draw_circle(img,R_interp,(0,255,255))

    # 지저분한 라인을 하나의 직선으로 만들기 fitline
    left_fit_line = ransac_line_fitting(img, L_interp)
    right_fit_line = ransac_line_fitting(img, R_interp)

    # 이전프레임을 사용해 smoothing (빈도가 0이 되지않게하는 기법)
    L_lane.append(left_fit_line), R_lane.append(right_fit_line)
    if len(L_lane) > 10:
        left_fit_line = smoothing(L_lane, 10)
    if len(R_lane) > 10:
        right_fit_line = smoothing(R_lane, 10)

    # 이미지final = (입력영상img, line 왼쪽,오른쪽에 보라색 선그어줌)
    final = draw_fitline(img, left_fit_line, right_fit_line)

    return final


# main #
# 유튜브 가져오기
youtube_url = 'https://www.youtube.com/watch?v=ipyzW38sHg0'
Video = pafy.new(youtube_url)
# 유튜브 정보 읽어오기
video_info(Video)
best = Video.getbest(preftype="mp4")
print("best resolution : {}".format(best.resolution)) # 사이즈 1280x720

# 동영상 가져오기
cap = cv2.VideoCapture(best.url)

# 동영상 크기(frame정보)를 읽어옴
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frameWidth, frameHeight)
frameRate = int(cap.get(cv2.CAP_PROP_FPS))
delay = round(1000/frameRate)
print('frame_size={}'.format(frame_size))
print('fps={}'.format(frameRate))

# 저장영상포맷
fourcc = cv2.VideoWriter_fourcc(*'XVID') # *'XVID' *'DIVX' *'MPEG' *'X264'
# 저장위치, Writer
outPath = 'youtube_SelfDrivingCar_result.mp4'
out = cv2.VideoWriter(outPath, fourcc, frameRate, frame_size)

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # # 영상 크기 변환. 작업에 용이하게
    # if frame.shape[0] != 540:  # resizing for challenge video
    #     frame = cv2.resize(frame, None, fx=3 / 4, fy=3 / 4, interpolation=cv2.INTER_AREA)

    cv2.imshow('frame', frame)
    # 라인 검색 함수 실행
    result = detect_lanes_img(frame)

    # 동영상 파일에 쓰기
    out.write(result)

    cv2.imshow('result', result)

    keyValue = cv2.waitKey(delay)
    if keyValue == 27:
        break

# cap이 열려있으면
if cap.isOpened():
    out.release()
    cap.release()

cv2.destroyAllWindows()