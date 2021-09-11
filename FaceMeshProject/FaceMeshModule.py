import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self,
               static_image_mode=False,
               max_num_faces=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence


        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces,
                                                 self.min_detection_confidence, self.min_tracking_confidence)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)


    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACE_CONNECTIONS,
                                          landmark_drawing_spec=self.drawSpecs,
                                          connection_drawing_spec=self.drawSpecs)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x, y =int(lm.x*iw), int(lm.y*ih)

                    #cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN,
                    #            0.5, (0, 255, 0), 1)
                    #print(id, x, y)
                    face.append([id,x,y])
                    #cv2.circle(img,(x,y),1,(0,0,255),1)
                faces.append(face)
        return img, faces





def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces)!=0:
            print(faces[0][8])
            cv2.circle(img, (faces[0][8][1], faces[0][8][2]), 5, (0, 0, 255), 5)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()