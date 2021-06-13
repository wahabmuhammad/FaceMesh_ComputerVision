import cv2
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence


def main():
    pTime = 0
    cTime = 0
    cam = cv2.VideoCapture(0)
    detector = faceDetector()
    while True:
        source, img = cam.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        # if len(lmlist) != 0:
        #     print(lmlist[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3 ,(255,0,255),3)

        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()