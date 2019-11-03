import numpy
import cv2

try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract

s = ''
n = 0

cap = cv2.VideoCapture(0)

while True:
    ret, image_np = cap.read()
#cv2.resize(image_np, (800, 600))
    cv2.imshow('Testing', image_np)
    print (pytesseract.image_to_string(image_np))

    if not(s == pytesseract.image_to_string(image_np)):
        s = pytesseract.image_to_string(image_np)
        n = 0

    else:
        if not(s == ''):
            n = n + 1

    if (n == 3):
        print ('Done! ' + s)
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
