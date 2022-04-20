from matplotlib import image
import requests
import string
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import cv2
import os
from PIL import Image
from skimage.exposure import rescale_intensity

app = Flask(__name__)
app.config['DEBUG'] = True
# prevent cached responses
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

@app.route('/')
def index_get():
    # Remove the new image on refresh on main page
    if os.path.exists('static/newImage.jpg'):
        os.remove('static/newImage.jpg')
    return render_template('main.html')

@app.route('/newImage/<name>')
def add_newImg(name):
    return render_template('output.html', type=name)

@app.route('/convolutions', methods=["POST"])
def apply_convolution():
    # Remove the new image from directory on submit
    if os.path.exists('static/newImage.jpg'):
        os.remove('static/newImage.jpg')

    # Read original image in greyscale
    originalImage = cv2.imread('static/iu.jpg',cv2.IMREAD_GRAYSCALE)

    # getting the name from the POST request
    name = request.form['convolution_type']
    kernel = pick_convolution_type(name)
    newImage = convolve(originalImage, kernel)
    cv2.imwrite('static/newImage.jpg'.format(kernel),newImage)
    return redirect(url_for('add_newImg', name=(name)))





@app.route('/contrast')
def contrast():
    # Remove the new image from directory on submit
    if os.path.exists('static/newImage.jpg'):
        os.remove('static/newImage.jpg')

    # Read original image in greyscale
    originalImage = cv2.imread('static/iu.jpg',cv2.IMREAD_GRAYSCALE)

    # getting the name from the POST request
    # name = request.form['convolution_type']
    # kernel = pick_convolution_type(name)
    # newImage = convolve(originalImage, kernel)
    newImage = originalImage
    mmin = np.array(originalImage).min()
    mmax = np.array(originalImage).max()
    lmin = 0
    lmax = 255
    delL = 255
    delM = 255
    for i in range(len(originalImage)):
        for j in range(len(originalImage[i])):
            newImage[i][j] = ((delL*(originalImage[i][j] - mmin))/delM) + lmin

    cv2.imwrite('static/newImage.jpg',newImage)
    return redirect(url_for('add_newImg', name="contrast"))



# KERNELS
def Identity():
    identity = np.array([[0,0,0],[0,1,0],[0,0,0]],dtype='int')
    return identity
def Emboss():
    emboss = np.array([[-2,-1,0],[-1,1,1],[0,1,2]],dtype='int')
    return emboss
def Sharpen():
    sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype='int')
    return sharpen
def EdgeDetection():
    edgedetect = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype='int')
    return edgedetect
def BoxBlur():
    boxblur = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype='int') / 9 
    return boxblur
def GaussianBlur():
    gblur = np.array([[1,2,1],[2,4,2],[1,2,1]],dtype='int') / 16 
    return gblur
def LeftSobel():
    leftsobel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype='int')
    return leftsobel
def RightSobel():
    rightsobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype='int')
    return rightsobel
def TopSobel():
    topsobel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype='int')
    return topsobel
def BottomSobel():
    bottomsobel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype='int')
    return bottomsobel

kernelDict = {
    "Identity": Identity,
    "Sharpen": Sharpen,
    "EdgeDetection": EdgeDetection,
    "BoxBlur": BoxBlur,
    "GaussianBlur": GaussianBlur,
    "LeftSobel": LeftSobel,
    "RightSobel": RightSobel,
    "TopSobel": TopSobel,
    "BottomSobel": BottomSobel,
    "Emboss": Emboss,
}
# Function to access kernel dictionary for corresponding kernel
def pick_convolution_type(name):
    return kernelDict[name]()

# Convolution function
def convolve(img,kernel):
  # establish variables for the image height and length
  (imgHeight,imgLength) = img.shape[:2]

  # establish variables for the kernel height and length
  (kHeight,kLength) = kernel.shape[:2]

  # A padded border is needed to deal with edge cases so
  # we first calculate the thickness of the pad
  pad = (kLength-1)//2

  # Make a border with thickness of the aforementioned pad
  img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
  
  # establish numpy 2D array the size of the original image
  final = np.zeros((imgHeight,imgLength),dtype='float32')

  # apply convolutions from left to right, then up to down
  for i in range(pad,imgHeight+pad):
    for j in range(pad,imgLength+pad):
      # define region of interest
      region = img[i-pad:i+pad+1,j-pad:j+pad+1]
      # sum up the new value of the center element
      newVal = (region*kernel).sum()
      final[i-pad,j-pad] = newVal

  # normalize pixel value to lie in range 0-255    
  final = rescale_intensity(final,in_range=(0,255))
  final = (final * 255).astype("uint8")
  # return manipulated image
  return final