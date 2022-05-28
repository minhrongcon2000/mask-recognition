from imutils.video import VideoStream
import imutils
import time
import cv2
from mask_recognition import make_inference_from_frame



# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    results = make_inference_from_frame(frame)

    if results is not None:
        face_locs = results

        for (startX, startY, endX, endY) in face_locs:
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
