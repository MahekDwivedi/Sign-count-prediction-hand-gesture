!pip install pyforest
#first install pyforest
!pip install mediapipe --user
# customizable ML solutions for live and streaming media.

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands()
# # First step is to initialize the Hands class an store it in a variable
mpDraw = mp.solutions.drawing_utils
finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_Coord = (4,2)

while True:
    success, image = cap.read()
    #Converting the input image to ‘RGB’ image (When the image file is read with OpenCV imread(),
    #the order of colors is BGR but the order of colors is assumed to be RGB of img)
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(RGB_image)
    #Drawing the landmarks present in the hand
    multiLandMarks = results.multi_hand_landmarks  #Collection of detected/tracked hands, where each hand is represented as a 
    #list of 21 hand landmarks and each landmark is composed of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
    if multiLandMarks:
        handList = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(image, handLms, mp_Hands.HAND_CONNECTIONS)
            
            for idx, lm in enumerate(handLms.landmark):
                
                #Changing the hand points coordinates into image pixels
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handList.append((cx, cy))
        #now circle each hand point we have identified
        for point in handList:
            cv2.circle(image, point, 10, (255, 255, 0), cv2.FILLED)
        upCount = 0
        for coordinate in finger_Coord:
            if handList[coordinate[0]][1] < handList[coordinate[1]][1]:
                upCount += 1
        if handList[thumb_Coord[0]][0] > handList[thumb_Coord[1]][0]:
            upCount += 1
        cv2.putText(image, str(upCount), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)

    cv2.imshow("Counting number of fingers", image)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cap.release()

cv2.destroyAllWindows()
    
