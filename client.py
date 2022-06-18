import socket
import time
from imutils.video import VideoStream
import imagezmq
import cv2
import json
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("--server_IP", type=str, help="IP address of server")
args = ap.parse_args()

IP_ADDRESS = args.server_IP
sender = imagezmq.ImageSender(connect_to=IP_ADDRESS)
rpi_name = socket.gethostname()
picam = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

avgFPS = 0
frameCount = 0

while True:
    frameCount += 1
    start = time.time()
    image = picam.read()
    image = imutils.resize(image, width=500)
    image_h, image_w, _ = image.shape
    reply = sender.send_image(rpi_name, image)
    
    results = json.loads(reply.decode("utf-8"))
    
    for result in results:
        label = result["label"]
        xMin = result["xMin"]
        xMax = result["xMax"]
        yMin = result["yMin"]
        yMax = result["yMax"]
        percentage = result["confidence"]

        cv2.putText(image, f"Mask: {label} ({percentage * 100:.2f}%)", (xMin - 40, yMin - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0) if label == "correct" else (0, 0, 255), 2)
        cv2.rectangle(image, (xMin, yMin),
                        (xMax, yMax), (0, 255, 0) if label == "correct" else (0, 0, 255), 2)
    end = time.time()
    currentFPS = 1 / (end - start)
    avgFPS = (1 - 1 / frameCount) * avgFPS + 1 / frameCount * currentFPS
    cv2.putText(image, "Current FPS: " + str(round(currentFPS, 1)), (30, image_h - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    cv2.putText(image, "Average FPS: " + str(round(avgFPS, 1)), (30, image_h - 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    cv2.namedWindow("Frame", cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
picam.stop()
