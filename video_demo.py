import cv2
from visualize_cv2 import model, display_instances, class_names
import sys
import time

args = sys.argv

if(len(args) < 2):
    print("run command: python video_demo.py 0 or video file name")
    sys.exit(0)

if(args[1].isdigit()):
    name = int(args[1])
else:
    name = args[1]

stream = cv2.VideoCapture(name)

started = time.time()
thisLoop = time.time()

while True:
    thisLoop = time.time()
    ret, frame = stream.read()
    if not ret:
        print("Unable to fetch frame")
        break
    results = model.detect([frame], verbose=1)

    # visualize results
    r = results[0]
    masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    cv2.imshow("masked_image", masked_image)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    timeperloop = (time.time() - thisLoop)
    print("time per loop {} ".format(timeperloop))

stream.release()
cv2.destroyWindow("masked_image") 
