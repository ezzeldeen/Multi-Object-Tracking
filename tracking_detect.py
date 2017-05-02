from darkflow.net.build import TFNet
import cv2
import numpy as np
import dlib

options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.1, "demo":"videofile.avi"}
tfnet = TFNet(options)

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture('data/Town.avi')

cv2.namedWindow('frame',cv2.WINDOW_AUTOSIZE)

ret, frame = cap.read()
result = tfnet.return_predict(frame)
arr = np.zeros((len(result), 4))
points = []
for i in range(0, len(result)):
    #print result[i], '\n'
    arr[i][0] = result[i]['topleft']['x']   # x1
    arr[i][1] = result[i]['topleft']['y']   # y1
    arr[i][2] = result[i]['bottomright']['x']   # x2
    arr[i][3] = result[i]['bottomright']['y']   # y2

    points.append((int(arr[i][0]),int(arr[i][1]),int(arr[i][2]),int(arr[i][3])))

tracker = [dlib.correlation_tracker() for _ in xrange(len(points))]
[tracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(points)]
rect = tracker[i].get_position()

while(cap.isOpened()):
    ret, frame = cap.read()

    for i in xrange(len(tracker)):
        #update tracker
        tracker[i].update(frame)

        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))

        #cv2.putText(frame, str(i+1), (int((arr[i][2] + arr[i][0])/2), int((arr[i][1]+ arr[i][3])/2)), font, 1, (255, 0, 0), 2)
        if (int(rect.left()) > 0 & int(rect.left()) < 1920) & (int(rect.top()) > 0 & int(rect.top()) < 500) & (int(rect.right()) > 0 & int(rect.right()) < 1920) & (int(rect.bottom()) > 0 & int(rect.bottom()) < 500):
            cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 3)
            cv2.putText(frame, str(i + 1), (int((rect.left() + int(rect.right())) / 2), int((rect.top() + rect.bottom()) / 2)), font, 1, (255, 255, 255), 6)
        rect = tracker[i].get_position()

        #print "Object tracked at [{}, {}] \r".format(pt1, pt2),

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
