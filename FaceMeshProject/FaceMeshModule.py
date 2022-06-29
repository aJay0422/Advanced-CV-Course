import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=2,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5
                 ):
        self.staticMode = static_image_mode
        self.maxFaces = max_num_faces
        self.refineLm = refine_landmarks
        self.minDetectCon = min_detection_confidence
        self.minTrackCon = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,
                                                 self.maxFaces,
                                                 self.refineLm,
                                                 self.minDetectCon,
                                                 self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0),
                                                thickness=1,
                                                circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #             0.5, (0, 255, 0), 1)
                faces.append(face)

        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))

        # Add fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()