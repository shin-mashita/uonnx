import serial
import numpy
import cv2

ser = serial.Serial('/dev/ttyUSB0', 115200)

vals = []
while(True):
    start = ser.read(1)
    if(start == b'X'):
        s= ser.read(3500)
        s = s.decode()
        pred_idx = s.find("DEMO")-1
        pred = s[pred_idx:pred_idx+40]
        s = s.lstrip("X")
        s = s[0:s.find("O")]
        try:
            frame = [int(num) for num in s.split(" ") if num != '']
        except ValueError:
            continue

        if(len(frame)==784):
            frame = numpy.array(frame, dtype=numpy.uint8)
            frame = numpy.reshape(frame, [28,28,1])
            frame = cv2.resize(frame, [280, 280])
            cv2.imshow("image", frame)
            print(pred)
            cv2.waitKey(10)
        else:
            continue
# s = list(s)
# print(s)
# print(len(s))



print(frame, len(frame))

# cv2.imwrite('color_img.jpg', img)

ser.close()
