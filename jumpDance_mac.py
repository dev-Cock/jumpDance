import cv2
import numpy as np
from ultralytics import YOLO

import sys

from PIL import Image, ImageTk
from tkinter import Tk, Canvas, PhotoImage, NW
import qimage2ndarray


import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel

import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel



# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')

class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Jump Dance")
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



            # take alpha values with mask. (Human will be visible only, and background's alpha will be 0)
            rgba_array[:, :, 3] = cv2.cvtColor(output_mask_all_converted, cv2.COLOR_BGR2GRAY)
            

            # draw contour lines on rgba array
            rgba_array[:, :, :3] += contour_image

            rgba_array = qimage2ndarray.array2qimage(rgba_array, normalize=False)
  

            pixmap = QPixmap.fromImage(rgba_array)

            self.label.setPixmap(pixmap)
            self.label.resize(pixmap.width(), pixmap.height())



    def closeEvent(self, event):
        # Release the webcam when the window is closed
        self.cap_web.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamWindow()
    window.show()
    sys.exit(app.exec_())