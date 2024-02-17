import cv2
import numpy as np
from threading import Thread

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

offsetx = 0
offsety = 0

startW = 1440
startH = 1080

previous_x = 0
previous_y = 0
previous_w = startW
previous_h = startH
offset = 50

counter = 0
while True:
    check, frame = video.read()
    previous_x -= offset
    previous_y -= offset
    if previous_x + offsetx < 0:
        previous_x = 0
    if previous_y + offsety < 0:
        previous_y = 0
    crop_img = frame[(previous_y + offsety):(previous_y + previous_h + 2 * offset + offsety), (previous_x + offsetx):(previous_x + previous_w + 2 * offset + offsetx)]
    frame = crop_img

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if counter % 5 == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
        offsetx += previous_x
        offsety += previous_y

    for x, y, w, h in faces:
        previous_x = x
        previous_y = y
        previous_w = w
        previous_h = h
        frame = cv2.rectangle(frame, (x, y), (x+60, y-25), (0, 0, 0), -1)
        frame = cv2.putText(frame, "Face", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 1)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

    for x, y, w, h in eyes:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        frame_w = frame.shape[1]
        frame_h = frame.shape[0]
        if (x + w) > frame_w:
            continue
        if (y + h) > frame_h:
            continue
        image = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        average = np.mean(blurred)
        #print(average)
        # Increase brightness
        brightness_factor = .8
        if average < 50:
            brightness_factor = 2
        elif average < 100:
            brightness_factor = 1.5
        elif average < 150:
            brightness_factor = 1.1
        elif average < 200:
            brightness_factor = .8

        brightened_image = cv2.convertScaleAbs(blurred, alpha=brightness_factor, beta=0)

        # Apply sharpening using a kernel
        sharpening_kernel = np.array([[-1, -1, -1],
                                    [-1, 10, -1],
                                    [-1, -1, -1]])
        sharpened_image = cv2.filter2D(brightened_image, -1, sharpening_kernel)


        # Use HoughCircles to detect circles (pupils) in the image
        circles = cv2.HoughCircles(
            sharpened_image, cv2.HOUGH_GRADIENT_ALT, dp=.8, minDist=20, param1=20, param2=.5, minRadius=5, maxRadius=30
        )

    if len(faces) == 0:
        previous_x = 0
        previous_y = 0
        previous_w = startW
        previous_h = startH

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    counter = counter + 1
