import cv2
import numpy as np
from ultralytics import YOLO

import sys

from PIL import Image, ImageTk
from tkinter import Tk, Canvas, PhotoImage, NW
import qimage2ndarray

import random

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel



# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')


starImg = cv2.imread("./star.png", cv2.IMREAD_UNCHANGED)
starImg = cv2.resize(starImg, [128,128])
starImg_mask = starImg.copy()
starImg_mask[starImg_mask[:, :, -1] > 0] = 255

# # 이미지를 윈도우에 표시합니다.
# cv2.imshow('Image', image)

# # 사용자가 키보드의 아무 키나 누를 때까지 대기합니다.
# cv2.waitKey(0)

# # 모든 윈도우를 닫습니다.
# cv2.destroyAllWindows()

class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Catch Star")
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.windowWidth = 1800
        self.windowHeight = 1080
        self.setFixedSize(self.windowWidth, self.windowHeight)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)  # Center align the label
        self.label.setGeometry(0, 0, self.windowWidth, self.windowHeight)

        # set contour line color / thickness
        self.lineColor = [255, 255, 255]
        self.thickness = 20  # Adjust the thickness as needed

        # Open the webcam
        self.cap_web = cv2.VideoCapture(1)

        # Setup a timer to update the window with webcam frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 milliseconds (approximately 30 FPS)

        # Catching star boolean
        self.catchedStar = False
        self.initialized = True
        self.star_w = 0
        self.star_h = 0


    def getRandomPosition(self, humanMask):
        while True:
            w = random.randint(130, self.windowWidth-130)
            h = random.randint(130, self.windowHeight-130)
            # Calculate the distance transform of the inverted mask
            inverted_mask = cv2.bitwise_not(humanMask[:, :, 0])
            dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            distance = dist_transform[h, w]
            # print(distance, w, h)
            if distance > 100:
                break

        return w, h

    def update_frame(self):

        # Read a frame from the webcam

        success, frame = self.cap_web.read()

        if success:

            # segmented result from yolo v8
            results = model.predict(frame, classes = 0)
        
            output_mask_all = np.zeros_like(frame)
            # iterate detection results 
            for r in results:
                img = np.copy(r.orig_img)

                # iterate each object contour 
                for ci,c in enumerate(r):

                    b_mask = np.zeros(img.shape[:2], np.uint8)

                    # Create contour mask 
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                    # Choose one:

                    # OPTION-1: Isolate object with black background
                    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                    isolated = cv2.bitwise_and(mask3ch, img)

                    # add each object's mask to the entire mask of this frame
                    output_mask_all[(output_mask_all > 0) | (isolated > 0)] = 255

        

            # # Display the annotated frame
            human = cv2.bitwise_and(frame, output_mask_all)

            # convert horizontally so that it can be seen as a mirror
            human = human[:, ::-1]


            # Create RGBA array with the same dimensions 
            rgba_array = np.zeros((human.shape[0], human.shape[1], 4), dtype=np.uint8)

            # Copy RGB values from the original array to the new array
            rgba_array[:, :, :3] = human[:, :, [2,1,0]]
            # Set alpha channel to 255 (fully opaque) for the entire array
            output_mask_all_converted = output_mask_all[:, ::-1]


            # Find contours in the mask image
            contours, _ = cv2.findContours(cv2.cvtColor(output_mask_all_converted, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an empty image to draw contours on
            contour_image = np.zeros_like(output_mask_all_converted)

            # Draw contours on the empty image with thickness
            cv2.drawContours(contour_image, contours, -1, self.lineColor, thickness=self.thickness)

            # add alpha value to be visible on mask
            output_mask_all_converted += contour_image


            # draw contour lines on rgba array
            rgba_array[:, :, :3] += contour_image

            output_mask_humanOnly = output_mask_all_converted.copy()

            output_mask_starOnly = np.zeros_like(output_mask_humanOnly)
            ################################################## Below: Make star in random position


            # if initialized, put star
            if self.initialized:
                self.star_w, self.star_h = self.getRandomPosition(output_mask_humanOnly)
                
                # Get rgba box                
                boxImg = rgba_array[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :].copy()
                # get alpha mask of star image
                alpha_mask = (starImg[:, :, 3] == 255)

                # put star image where there is a star
                boxImg[alpha_mask, :3] = starImg[alpha_mask, :3]
                rgba_array[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :] = boxImg

                # get updated mask in the box
                output_mask_box = boxImg.copy()
                output_mask_box[output_mask_box >0] = 255

                output_mask_starOnly[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :] = output_mask_box[:, :, :3]

                output_mask_all_converted[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :] = output_mask_box[:, :, :3]
                self.initialized = False

            else:

                # didn't catched the star? then display it
                if not self.catchedStar:
                    # Get rgba box                
                    boxImg = rgba_array[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :].copy()
                    # get alpha mask of star image
                    alpha_mask = (starImg[:, :, 3] == 255)

                    # put star image where there is a star
                    boxImg[alpha_mask, :3] = starImg[alpha_mask, :3]
                    rgba_array[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :] = boxImg

                    # get updated mask in the box
                    output_mask_box = boxImg.copy()
                    output_mask_box[output_mask_box >0] = 255

                    output_mask_starOnly[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :] = output_mask_box[:, :, :3]

                    output_mask_all_converted[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :] = output_mask_box[:, :, :3]
                    

                if self.catchedStar:
                    self.star_w, self.star_h = self.getRandomPosition(output_mask_humanOnly)
                    # Get rgba box                
                    boxImg = rgba_array[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :].copy()
                    # get alpha mask of star image
                    alpha_mask = (starImg[:, :, 3] == 255)

                    # put star image where there is a star
                    boxImg[alpha_mask, :3] = starImg[alpha_mask, :3]
                    rgba_array[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :] = boxImg

                    # get updated mask in the box
                    output_mask_box = boxImg.copy()
                    output_mask_box[output_mask_box >0] = 255


                    output_mask_all_converted[self.star_h:self.star_h+128, self.star_w:self.star_w+128, :] = output_mask_box[:, :, :3]
                    self.catchedStar = False

                
            
            
            
            
            
            
            
            # take alpha values with mask. (Human will be visible only, and background's alpha will be 0)
            rgba_array[:, :, 3] = cv2.cvtColor(output_mask_all_converted, cv2.COLOR_BGR2GRAY)
            

            
            
            rgba_array = qimage2ndarray.array2qimage(rgba_array, normalize=False)

  

            pixmap = QPixmap.fromImage(rgba_array)

            self.label.setPixmap(pixmap)
            self.label.resize(pixmap.width(), pixmap.height())



            ######################### Check if star is catched by distance
            if np.count_nonzero(output_mask_starOnly & output_mask_humanOnly) > 0:
                print("Star is catched!!")
            
                self.catchedStar = True



    def closeEvent(self, event):
        # Release the webcam when the window is closed
        self.cap_web.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamWindow()
    window.show()
    sys.exit(app.exec_())