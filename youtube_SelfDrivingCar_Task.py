# -*- coding: utf-8 -*- # 한글 주석
import cv2  # opencv 사용
import numpy as np
import sys
import pafy

def region_of_interest(img, vertices, vertices2, color3=(255, 255, 255), color1=255, color3_b=(0,0,0), color1_b = 0):
    # mask = img와 같은 크기의 0으로 가득찬 빈 이미지(배열)을 생성 <-> 1로채운것은 np.ones_like()
    mask = np.zeros_like(img)
    # 255흰색으로 초기화
    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
        color_b = color3_b
    else:  # 흑백 이미지(1채널)라면 :
        color = color1
        color_b = color1_b

    # 다각형밖을 마스크로채움
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 하얀색 color로 채움
    cv2.fillPoly(mask, vertices, color)
    # vertices2에 정한 점들로 이뤄진 다각형 (ROI 내부 도로글씨 부분)을 검정색 color_b로 마스크 채움
    cv2.fillPoly(mask, vertices2, color_b)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    #cv2.imshow('mask', mask)
    cv2.imshow('ROI', ROI_image)
    # 다각형 내부 이미지만 리턴
    return ROI_image

# 기울기가 적용된 라인 그리기
def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            #print(line)

# 차선 여러개를 하나의 fitline 중심 선으로 합치기
def get_fitline(img, lines):
    lines = lines.reshape(lines.shape[0] * 2, 2)    # 4개씩 3차원 배열을 직선인 2개씩 2차원배열로
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01) # (입력영상,알고리즘지정.)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1 # 선분의 시작
    x2, y2 = int(((img.shape[0] / 2 + 50) - y) / vy * vx + x), int(img.shape[0] / 2 + 50) # 선분의끝
    fit_result = [x1, y1, x2, y2]

    return fit_result

# fitline한 선 안쪽을 색칠하기
def drawing_inside(img, vertices, color3=(200, 255, 200), color1=100):
    mask = np.full(img.shape, (255, 255, 255), dtype=np.uint8) # 하얀보드 생성
    cv2.fillPoly(mask, [vertices], color3)
    drawing_image = cv2.bitwise_and(img, mask)
    cv2.imshow('drawing_image', drawing_image)
    return drawing_image

## Main ##
# main #
# 유튜브 가져오기
youtube_url = 'https://www.youtube.com/watch?v=ipyzW38sHg0'
Video = pafy.new(youtube_url)
# 유튜브 정보 읽어오기
best = Video.getbest(preftype="mp4")
print("best resolution : {}".format(best.resolution)) # 사이즈 1280x720

# 동영상 가져오기
#image = cv2.imread('slope_test.jpg')
#cap = cv2.VideoCapture('solidWhiteRight.mp4')
cap = cv2.VideoCapture(best.url)
if cap is None:
    print("Cant read video")
    sys.exit()

# 동영상 크기(frame정보)를 읽어옴
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frameWidth, frameHeight)
frameRate = int(cap.get(cv2.CAP_PROP_FPS))
delay = round(1000/frameRate)
print('frame_size={}'.format(frame_size))
print('fps={}'.format(frameRate))

# 저장영상포맷
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # *'XVID' *'DIVX' *'MPEG' *'X264'
# 저장위치, Writer
outPath = 'youtube_SelfDrivingCar_Task.mp4'
out = cv2.VideoWriter(outPath, fourcc, frameRate, frame_size)

height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while (cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break

    # 그레이
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우스
    kernel_size = 3 #blur정도
    Gablur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # 캐니
    Lowth, Highth = 50, 255
    canny = cv2.Canny(Gablur, Lowth,Highth)

    # 바깥쪽 ROI 인식범위잡기
    vertices = np.array(
        [[(200, height), (width / 2 + 5, height / 2 + 10), (width / 2 + 65, height / 2 + 10), (width - 20, height)]],
        dtype=np.int32)
    cv2.polylines(image, vertices, True, (255, 0, 0), 2) # 선 그리기

    # 안쪽 ROI 인식범위 잡기
    vertices2 = np.array(
        [[(360, height), (width / 2 - 25, height / 2 + 55), (width / 2 + 85, height / 2 + 55), (width - 200, height)]],
        dtype=np.int32)
    cv2.polylines(image, vertices2, True, (255, 0, 0), 2) # 선 그리기

    # ROI 범위로 ROI함수 영역 생성
    ROI_img = region_of_interest(canny, vertices = vertices, vertices2 = vertices2)

    # HoughlinesP 를 통해 거친 직선 찾기
    rho, theta = 1, 1*np.pi/180
    threshold = 30
    min_line_len , max_line_gap = 10, 20
    # HoughLinesP는 HoughLines의 직선과 다르게, 선분을 두 조건 min,max으로 2번 걸러준다. 연산이 빠르다
    lines = cv2.HoughLinesP(ROI_img, rho, theta, threshold, np.array([]),min_line_len,max_line_gap)

    # DEBUG 허프라인 선 lines 그려주기
    # 차선그리기
    # line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    # cv2.imshow('line_img',line_img)


    # 허프라인 에서 기울기가 낮은값을 제거하기
    # np.expand_dims(arr, axis=0) 차원생성 각각 분리, np.squeeze(lines) 차원합치기
    line_arr = np.squeeze(lines) #3차원배열[[608 430 677 432]], [[]] 을 2차원배열[608 430 677 432]로

    # 선들의 기울기 구하기 arctan2 * 180 / 파이
    # fx : line_arr[:, 1] - line_arr[:, 3]
    # fy : line_arr[:, 0] - line_arr[:, 2]
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # 라인배열에서의 직선 중에 160도 미만인것만 라인배열에 남김 # 절대값
    line_arr = line_arr[np.abs(slope_degree) < 160]
    # slope_degree 리스트도 160도 미만인것만 slope에 남김
    slope_degree = slope_degree[np.abs(slope_degree) < 160]

    # 라인배열에서의 직선 중에 95도 이상인것만 라인배열에 남김
    line_arr = line_arr[np.abs(slope_degree) > 95]
    # slope_degree 리스트도 95도 이상인것만 slope에 남김
    slope_degree = slope_degree[np.abs(slope_degree) > 95]

    # 차선 그리기
    # line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # line_arr_exp = np.expand_dims(line_arr, axis=1)
    # draw_lines(line_img, line_arr_exp)
    # cv2.imshow('line_img',line_img)

    # Left, Right line을 분리 후
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]

    # 원본이미지와 차선 lines 합치기(그리기)
    draw_lines(image, L_lines) # 왼쪽라인 기울기정리된 선 그리기
    draw_lines(image, R_lines) # 오른쪽라인 기울기정리된 선 그리기



    # 지저분한 라인을 하나의 직선으로 만들기 fitline
    left_fit_line = get_fitline(image, L_lines)
    right_fit_line = get_fitline(image, R_lines)

    # 화면에 합치기
    cv2.line(image, (left_fit_line[0], left_fit_line[1]), (left_fit_line[2], left_fit_line[3]), (0,255,0), 2)
    cv2.line(image, (right_fit_line[0], right_fit_line[1]), (right_fit_line[2], right_fit_line[3]), (0,255,0), 2)

    # fitline해온 직선 안쪽을 bitwise로 색칠하기
    pts = np.array([[left_fit_line[0], left_fit_line[1]], [right_fit_line[0], right_fit_line[1]], \
                    [right_fit_line[2], right_fit_line[3]], [left_fit_line[2], left_fit_line[3]] ])
    drawing_image = drawing_inside(image, pts)

    # 파일에 frame 저장
    out.write(drawing_image)
    #
    cv2.imshow('drawing_image',drawing_image)
    #cv2.imshow('image', image)

    keyvalue = cv2.waitKey(delay)
    if keyvalue == ord('s'):
        cv2.waitKey(0)
    elif keyvalue == 27:
        break


# cap이 열려있으면
if cap.isOpened():
    out.release()
    cap.release()
cv2.destroyAllWindows()