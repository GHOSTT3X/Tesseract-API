from flask import Flask, jsonify
from flask import request
from cv2 import dilate
import os
from werkzeug.utils import secure_filename
import sys
from io import BytesIO
import pytesseract
import cv2
import numpy as np
import imutils
from PIL import Image, ImageEnhance
from collections import namedtuple
import pytesseract
import numpy as np
import imutils
from PIL import Image, ImageEnhance
from PyPDF4 import PdfFileReader
from PyPDF4 import PdfFileWriter
from pdf2image import convert_from_bytes
import io


app = Flask(__name__)
app.secret_key = 'SecretKey420'

ALLOWED_EXTENSIONS = set(["PNG", "JPG", "JPEG", "RAW", "CR2", "PDF", "GIF"])

app.config["ALLOWED_IMAGE_EXTENTIONS"] = ALLOWED_EXTENSIONS


def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENTIONS"]:
        return True
    else:
        return False


custom_config_eng = r'-l eng --psm 6'
custom_config_ara = r'-l ara --psm 6'
custom_config_mix = r'-l ara+eng --psm 6'
custom_config_date = r'-l ara+eng --psm 12'
custom_config_digits = r'--oem 3 --psm 6 outputbase digits'
custom_config_whitelist = r'-c tessedit_char_whitelist=0123456789/ --psm 6'


@app.route("/upload-image", methods=["POST", "GET"])
def upload_and_proc_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            if image.filename == "":
                return '* File Not Selected *'

            if not allowed_image(image.filename):
                return '* File Extension NOT ALLOWED! *'

            else:

                filename = secure_filename(image.filename)

                if (filename.rsplit(".", 1)[1].upper() == "PDF"):

                    src_pdf = PdfFileReader(image.stream, strict=False)

                    dst_pdf = PdfFileWriter()

                    dst_pdf.addPage(src_pdf.getPage(0))

                    pdf_bytes = io.BytesIO()

                    dst_pdf.write(pdf_bytes)

                    pdf_bytes.seek(0)

                    bytes = pdf_bytes.getvalue()

                    pages = convert_from_bytes(bytes, last_page=1, dpi=200)

                    for page in pages:
                        page.save(pdf_bytes, 'PNG')
                    pdf_bytes.seek(0)

                    img = cv2.imdecode(np.frombuffer(
                        pdf_bytes.read(), np.uint8), 1)

                else:
                    # Bytes of Image
                    img_bytes = request.files['image'].read()

                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)

                heightImg = 2000
                widthImg = 4000

                img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE

                def reorder(myPoints):
                    myPoints = myPoints.reshape((4, 2))
                    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
                    add = myPoints.sum(1)
                    myPointsNew[0] = myPoints[np.argmin(add)]
                    myPointsNew[3] = myPoints[np.argmax(add)]
                    diff = np.diff(myPoints, axis=1)
                    myPointsNew[1] = myPoints[np.argmin(diff)]
                    myPointsNew[2] = myPoints[np.argmax(diff)]
                    return myPointsNew

                def biggestContour(contours):
                    biggest = np.array([])
                    max_area = 0
                    for i in contours:
                        area = cv2.contourArea(i)
                        x, y, w, h = cv2.boundingRect(i)
                        peri = cv2.arcLength(i, True)
                        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                        if area > max_area and len(approx) == 4:
                            biggest = approx
                            max_area = area
                    return biggest, max_area

                def drawRectangle(img, biggest, thickness):
                    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (
                        biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
                    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (
                        biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
                    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (
                        biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
                    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (
                        biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
                    return img

                def nothing(x):
                    pass

                # CONVERT IMAGE TO GRAY SCALE
                imgBlur = cv2.GaussianBlur(img, (5, 5), 1)  # ADD GAUSSIAN BLUR
                imgThreshold = cv2.Canny(imgBlur, 30, 150)  # APPLY CANNY BLUR
                kernel = np.ones((5, 5))
                imgDial = cv2.dilate(imgThreshold, kernel,
                                     iterations=2)  # APPLY DILATION
                imgThreshold = cv2.erode(
                    imgDial, kernel, iterations=1)  # APPLY EROSION

                # FIND ALL COUNTOURS
                imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
                imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
                contours, hierarchy = cv2.findContours(
                    imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
                # DRAW ALL DETECTED CONTOURS
                cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

                # FIND THE BIGGEST COUNTOUR
                biggest, maxArea = biggestContour(contours)
                if biggest.size != 0:
                    biggest = reorder(biggest)
                    # DRAW THE BIGGEST CONTOUR
                    cv2.drawContours(
                        imgBigContour, biggest, -1, (0, 255, 0), 20)
                    imgBigContour = drawRectangle(imgBigContour, biggest, 2)
                    pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
                    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [
                                      widthImg, heightImg]])  # PREPARE POINTS FOR WARP
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    imgWarpColored = cv2.warpPerspective(
                        img, matrix, (widthImg, heightImg))
                # REMOVE THE PIXELS FORM EACH SIDE
                    imgWarpColored = imgWarpColored[30:imgWarpColored.shape[0] -
                                                    20, 10:imgWarpColored.shape[1] - 800]
                    imgWarpColored = cv2.resize(
                        imgWarpColored, (widthImg, heightImg))
                # Deskewing:
                    # img_edges = cv2.Canny(
                    # imgWarpColored, 100, 100, apertureSize=3)
                    # lines = cv2.HoughLinesP(
                    # img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
                    #angles = []
                    # for x1, y1, x2, y2 in lines[0]:
                    #angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    # angles.append(angle)

                    #median_angle = np.median(angles)
                    #final = ndimage.rotate(imgWarpColored, median_angle)

                    # Enhacing the image:
                    imgWarpColored = cv2.cvtColor(
                        imgWarpColored, cv2.COLOR_BGR2GRAY)

                    kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])
                    imgWarpColored = cv2.filter2D(
                        src=imgWarpColored, ddepth=-1, kernel=kernel)
                    # I NEED TO CHECK THIS ONE!
                    img = Image.fromarray(np.uint8(img))
                    cv_img = ImageEnhance.Contrast(img).enhance(1.1)
                    img = np.array(cv_img)

                    # Tesseract: Third Block

                    third_block = pytesseract.image_to_string(
                        imgWarpColored, config=custom_config_mix)

                    def split(first_block):
                        return [char for char in first_block]

                    third_list = split(third_block)

                    symbols = set([',', '.', '#', '%', '!', '|', "'", '=', '??', '~', '`', '+', '_', '[', ']',
                                  '{', '}', '^', '&', '*', '(', ')', '@', '$', '?', '<', '>', '??', '\u200e', '\u200f', '', ' '])

                    res = symbols.intersection(third_list)

                    for x in third_list:
                        if x in symbols:
                            third_list.remove(x)

                    third_clean = ''.join([str(item) for item in third_list])

                    third_clean = third_block.split(sep='\n')

                    third_clean = [s.replace('\u200e', '')
                                   for s in third_clean]
                    third_clean = [s.replace('\u200f', '')
                                   for s in third_clean]

                    ########################

                    # First Block:
                    img_copy = img.copy()
                    img_copy2 = img.copy()

                    cnts = cv2.findContours(
                        imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                    for c in cnts:
                        rect = cv2.boundingRect(c)
                        x, y, w, h = rect
                        if (h > 196 and h < 203) and (w > 3627 and w < 3638):
                            cv2.rectangle(img_copy, (x, y),
                                          (x + w, y + h), (36, 255, 12), 1)

                            roi = img[y:y+h, x:x+w]

                            roi = roi[10:roi.shape[0] -
                                      70, 10:roi.shape[1] - 800]

                            roi = img = cv2.resize(
                                roi, (3000, 600))

                            # Tesseract:
                            roi = cv2.cvtColor(
                                roi, cv2.COLOR_BGR2GRAY)

                            kernel = np.array([[0, -1, 0],
                                               [-1, 5, -1],
                                               [0, -1, 0]])
                            roi = cv2.filter2D(
                                src=roi, ddepth=-1, kernel=kernel)

                            img = Image.fromarray(np.uint8(roi))
                            cv_img = ImageEnhance.Contrast(img).enhance(1.1)
                            img = np.array(cv_img)

                            first_block = pytesseract.image_to_string(
                                roi, config=custom_config_date)

                            first_list = split(first_block)

                            symbols = set([',', '.', '#', '%', '!', '|', "'", '=', '??', '~', '`', '+', '_', '[', ']',
                                          '{', '}', '^', '&', '*', '(', ')', '@', '$', '?', '<', '>', '??', '\u200e', '\u200f', '', ' '])

                            res = symbols.intersection(first_list)

                            for x in first_list:
                                if x in symbols:
                                    first_list.remove(x)

                            first_clean = ''.join(
                                [str(item) for item in first_list])

                            first_clean = first_block.split(sep='\n')

                            for item in first_clean:
                                if '/' in item:
                                    tarikh = item

                            for i in range(len(first_clean)):
                                if len(first_clean[i]) == 8:
                                    mouaref = first_clean[i]
                                elif len(first_clean[i]) in range(9, 13):
                                    if '/' not in first_clean[i]:
                                        adad_sejel = first_clean[i]

                    ########################

                    # forth Block:

                    cnts = cv2.findContours(
                        imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                    for c in cnts:
                        rect = cv2.boundingRect(c)
                        x, y, w, h = rect
                        if (h > 180 and h < 187) and (w > 3627 and w < 3638):
                            cv2.rectangle(img_copy2, (x, y),
                                          (x + w, y + h), (36, 255, 12), 1)

                            roi = img_copy2[y:y+h, x:x+w]

                            roi = roi[10:roi.shape[0] -
                                      10, 10:roi.shape[1] - 700]

                            roi = img = cv2.resize(
                                roi, (3200, 400))

                            # Tesseract:
                            roi = cv2.cvtColor(
                                roi, cv2.COLOR_BGR2GRAY)

                            kernel = np.array([[0, -1, 0],
                                               [-1, 5, -1],
                                               [0, -1, 0]])
                            roi = cv2.filter2D(
                                src=roi, ddepth=-1, kernel=kernel)

                            img = Image.fromarray(np.uint8(roi))
                            cv_img = ImageEnhance.Contrast(img).enhance(1.1)
                            img = np.array(cv_img)

                            forth_block = pytesseract.image_to_string(
                                roi, config=custom_config_date)

                            forth_list = split(forth_block)

                            symbols = set([',', '.', '#', '%', '!', '|', "'", '=', '??', '~', '`', '+', '_', '[', ']',
                                          '{', '}', '^', '&', '*', '(', ')', '@', '$', '?', '<', '>', '??', '\u200e', '\u200f', ''])

                            res = symbols.intersection(forth_list)

                            for x in forth_list:
                                if x in symbols:
                                    forth_list.remove(x)

                            forth_clean = ''.join(
                                [str(item) for item in forth_list])

                            forth_clean = forth_block.split(sep='\n')

                            # Extraction of interested Data:

                            for i in range(len(forth_clean)):
                                if len(forth_clean[i]) == 2:
                                    modatCharika = forth_clean[i]

                            date = []
                            for item in forth_clean:
                                if '/' in item:
                                    date.append(item)
                                elif '-' in item:
                                    date.append(item)

                            date1 = date[0]
                            date2 = date[1]

                            if date1 < date2:
                                tarikhBideyetNachat = date1
                                tarikhEchhar = date2
                            else:
                                tarikhBideyetNachat = date2
                                tarikhEchhar = date1

                            for i in range(len(forth_clean)):
                                if len(forth_clean[i]) > 9:
                                    if '/' not in forth_clean[i]:
                                        if '-' not in forth_clean[i]:
                                            if ':' not in forth_clean[i]:
                                                nachatRaisi = forth_clean[i]
                                                
                dict1 = {'mouaref': mouaref, 'tarikh': tarikh, 'adad_sejel': adad_sejel, 'nachatRaisi': nachatRaisi,
                         'tarikhBideyetNachat': tarikhBideyetNachat, 'tarikhEchhar': tarikhEchhar, "modatCharika": modatCharika, "tasmiya": third_clean[0], "tasmiyaLatin": third_clean[1],
                         "esmTijari": third_clean[2], "esmTijariLatin": third_clean[3], "makarEjtima": third_clean[4], "makarNachat": third_clean[5], "nithamKanouni": third_clean[6], "rasMal": third_clean[7], "adadFar": third_clean[8]}

                return dict1


if __name__ == "__main__":
    app.run()