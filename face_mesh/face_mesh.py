import cv2
import mediapipe as mp
import time

wCam, hCam = 1280, 720
kamera = cv2.VideoCapture(0)
kamera.set(3,1280)
kamera.set(4,720)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(circle_radius = 2)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

while True:
    source, img = kamera.read() #Source gambar atau webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    hasil = hands.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)

    if hasil.multi_hand_landmarks:
        for handlns in hasil.multi_hand_landmarks:
            for id, ln in enumerate(handlns.landmark):
                h,w,c = img.shape
                cx,cy = int(ln.x*w), int(ln.y*h)
                # if id == 4:
                cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlns, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # Memberi text FPS di layar
    cv2.putText(img, "FPS: " +str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3 ,(0,255,255),3)
    # Show image pop up
    cv2.imshow("videos", img)
    key = cv2.waitKey(5)
    if key == 27 or key == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()