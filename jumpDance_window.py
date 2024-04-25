import cv2
import numpy as np
from ultralytics import YOLO

import sys

from PIL import Image, ImageTk
from tkinter import Tk, Canvas, PhotoImage, NW

root = Tk()

root.attributes('-transparent',True)
# root.attributes('-alpha', 0.5)
# root.config(bg='systemTransparent')
# root.config(bg="black", bd=0, highlightthickness=0)
# root.wm_attributes("-transparentcolor", "white")

# Canvas
canvas = Canvas(root, width=1800, height=1080)
canvas.pack()


# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')

# Open the video file
cap_web = cv2.VideoCapture(1)




# Loop through the video frames
while cap_web.isOpened():


    # Read a frame from the video
    success, frame = cap_web.read()

    if success:

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

       
                # TODO your actions go here 
                # add each object's mask to the entire mask of this frame
                output_mask_all[(output_mask_all > 0) | (isolated > 0)] = 255

     

        # # Display the annotated frame
        human = cv2.bitwise_and(frame, output_mask_all)
        human = human[:, ::-1]



        # Create RGBA array with the same dimensions
        rgba_array = np.zeros((human.shape[0], human.shape[1], 4), dtype=np.uint8)

        # Copy RGB values from the original array to the new array
        rgba_array[:, :, :3] = human[:, :, [2,1,0]]
        # Set alpha channel to 255 (fully opaque) for the entire array
        rgba_array[:, :, 3] = 255

        # print(rgba_array.shape)
        # Image
        img = ImageTk.PhotoImage(image=Image.fromarray(rgba_array))
        
        # Positioning the Image inside the canvas
        canvas.create_image(0, 0, anchor=NW, image=img)
        
        # Starts the GUI
        # root.update_idletasks()
        root.update()
        # cv2.imshow("YOLOv8 Inference", human)



        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap_web.release()
cv2.destroyAllWindows()

