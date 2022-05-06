from flask import Flask
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
import json
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

@app.route("/flask", methods=["POST", "GET"])
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
 
                    img = cv2.imdecode(np.frombuffer(pdf_bytes.read(), np.uint8), 1)

                
                   
                else:
                    # Bytes of Image
                    img_bytes = request.files['image'].read()
                    
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)

            

                heightImg = 2000
                widthImg  = 4000

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
                    
                    
                # Third Block
               
                
                



                

                    
                
         
                    
                
         
                    
    return 'Working so far!'
               
                                    

if __name__ == "__main__":
    app.run(debug=True)



from flask import Flask
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
import json
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

@app.route("/flask", methods=["POST", "GET"])
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
 
                    img = cv2.imdecode(np.frombuffer(pdf_bytes.read(), np.uint8), 1)

                
                   
                else:
                    # Bytes of Image
                    img_bytes = request.files['image'].read()
                    
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)

                third_block = pytesseract.image_to_string(img, config=custom_config_mix)

                

            
         
                    
    return 'Working so far!'
               
                                    

if __name__ == "__main__":
    app.run(debug=True)



