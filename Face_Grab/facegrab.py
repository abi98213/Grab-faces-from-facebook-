"""
The MIT License (MIT)

Copyright (c) 2014 Ankit Aggarwal <ankitaggarwal011@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import print_function
import sys
import os
import requests
from datetime import datetime
from random import randint
import cv2
import sys
import time
import matplotlib.pyplot as plt

class Facegrab:
    def __init__(self):
        self.base_url = "http://graph.facebook.com/picture?id={}&width=800"
        self.sess = requests.Session()
        self.classifier =cv2.CascadeClassifier("/home/abdullah/Documents/Comapny work/Dataset/FaceGrab-master/classifier.xml")
        self.sess.headers.update({
            "User-Agent": "Facegrab v2"
        })
        self.folder_to_save = ""

    @staticmethod
    def create_dir(prefix):
        dir_c = os.path.join(
            os.getcwd(),
            prefix,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        try:
            os.makedirs(dir_c)
        except OSError as e:
            if e.errno != 17:
                pass
            else:
                print("Cannot create a folder.")
                exit
        return dir_c

    def filter_Image(self, img_path, _id):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = image.shape
        if (size[0] > 400 and size[1] > 400 ):


            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.classifier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30, 30))#,flags = cv2.cv.CV_HAAR_IMAGE_SCALE)

            for (x, y, w, h) in faces:
                if (w != 0 or h != 0):

                    plt.imsave(self.folder_to_save  + "/" + {}.jpg".format(_id), image)


    def getProfile(self, photoUrl, saveUrl):
        print(f"Downloading {photoUrl}.")
        response = self.sess.get(photoUrl)
        if response.headers["Content-Type"] == "image/gif":
            return
        with open(saveUrl, "wb") as f:
            f.write(response.content)

        return True

    def getImages(self, sizeDataset):
        _id = randint(10000, int(1e4))
        photoCount = 0
        folder = self.create_dir("facegrab")
        while photoCount < sizeDataset:
            profile = self.getProfile(
                self.base_url.format(_id),
                f"{folder}/{_id}.jpg"
            )


            if profile:
                self.filter_Image(f"{folder}/{_id}.jpg", _id)
                photoCount += 1
                _id += 1
            else:
                _id += 10    # Cannot understand the logic behind this.
        print(
            "\nFace Dataset created in facegrab folder."
            f"\nSize: {photoCount}"
        )
        print()
        return

if __name__ == "__main__":
    checks = [
        len(sys.argv) == 2,
        sys.argv[1].isdigit(),
        int(sys.argv[1]) < int(1e7)
    ]
    if all(checks):
        grabby = Facegrab()
        grabby.getImages(int(sys.argv[1]))
    else:
        print("\nIncorrect arguments.")
        print(
            "Usage: python facegrab.py <dataset size (integer < 10,000,000)>"
        )
