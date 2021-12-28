# bird eyesview image -> video-> cap.read-> crop > perspective
# 1.정지이미지-> 동영상
#   imread("road.jpg")->cv2.VidoeCaputure('project_video.mp4')
# 2. 동영상 체크
# 3. cap.read -> frame
# 4. crop.
#   frame을 필요한부분만 crop numpy영역지정(이미지 crop)
# 5. perspect 좌표지정
#   마우스로 perspective transform 4모서리 좌표 지정, 빨간색 선긋기
# 6. PerspectiveTransform
#   pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
#   pers_frame = cv2.warpPerspective(src, pers, (w, h))
# 7. 'bird eyesview'창 생성해 pers_frame이미지 출력
# * docuscan에서 drawROI

import sys
import cv2
import numpy as np


# perspective를 위한 4개 좌표값 초기화
pt1 = [0,0]
pt2 = [0,0]
pt3 = [0,0]
pt4 = [0,0]
counter = 0


# persepective 마우스 좌표 제어 함수
def on_mouse(event, x,y, flags, param):
    global counter, pt1,pt2,pt3,pt4
    if flags & cv2.EVENT_FLAG_LBUTTON:
        if counter==0:
            pt1 = [x, y]
            print("pt1, x:{}, y:{}".format(x, y))
        elif counter==1:
            pt2 = [x, y]
            print("pt2, x:{}, y:{}".format(x, y))
        elif counter==2:
            pt3 = [x,y]
            print("pt3, x:{}, y:{}".format(x, y))
        elif counter==3:
            pt4 = [x,y]
            print("pt4, x:{}, y:{}".format(x, y))
        counter += 1
    # 취소
    elif flags & cv2.EVENT_FLAG_RBUTTON:
        if counter > 0:
            counter -= 1
            print(counter)
    if event == cv2.EVENT_MOUSEMOVE:
        print("x:{}, y:{}".format(x,y))


# 비디오 파일 읽어오기
video_path = 'project_video.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to load VideoCapture")
    sys.exit()

# 비디오 사이즈, fps와 프레임당 딜레이설정
video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = round(1000 / fps)
print("video_width : {0} , video_height : {1}".format(video_width, video_height)) # 1280, 720

# 비디오 출력 창 생성
cv2.namedWindow("Video")
# birdeyeView 창 사이즈 w,h
w, h = 200, 380

# 비디오 좌표처리 시작<
# 비디오 1장 읽어오고
ret, frame = cap.read()
if not ret:
    sys.exit()

# 마우스 제어함수 persepective
cv2.setMouseCallback('Video', on_mouse, frame)

# 이미지 띄우고 x,y축 크기입력후 키보드 대기 (onmouse 마우스 4번클릭 끝날때까지)
cv2.imshow("Video", frame)
cv2.waitKey()

# persepective
srcQuad = np.array([pt1, pt2, pt3, pt4], np.float32) # np.array 변환될 원근 좌표위치
dstQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32) # np.array 결과 도착 좌표위치 [10시] [2시] [4시] [8시]

# pers는 변환행렬 (3x3 matrix) [[-2.88000000e+01 -9.60000000e+00  1.05600000e+04]
#                             [-4.44089210e-15 -2.15400000e+01  4.30800000e+03]
#                             [-1.77809156e-17 -2.00000000e-02  1.00000000e+00]]
pers = cv2.getPerspectiveTransform(srcQuad, dstQuad) # getPers...(원근좌표A -> 변환좌표B)

# > 비디오 좌표 처리 끝

# 비디오 프레임처리 시작
while True:
    # 프레임 읽어오기
    ret, frame = cap.read()
    if not ret:
        print("cant read Video anymore")
        break

    # frame 가져와서 원근 변환 함수(cv2.warpPerspective)는 원근 맵 행렬에 대한 기하학적 변환을 실시
    # pers (입력 이미지(frame)에 원근 맵 행렬Matrix(pers), (표현될 이미지크기))
    pers_frame = cv2.warpPerspective(frame, pers, (w, h))

    # 프레임 출력
    cv2.imshow("Video", frame)
    cv2.imshow("birdeyeView_Video", pers_frame)

    # 입력시 종료
    if cv2.waitKey(delay) == 27:
        break
# 비디오 프레임 처리 끝

# 연결분리
cap.release()
cv2.destroyAllWindows()




