import cv2
import os

def yo_make_the_conversion(img_data, countSum):
    img = img_data
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    path_png='test_Image'
    filename = os.path.join(path_png,'image{}.png'.format(countSum))
    countSum += 1
#    yo = tester.process_image_for_ocr(file)
    cv2.imwrite(filename, gray)
    #cv2.imshow('Conversion', gray)
    return filename, countSum