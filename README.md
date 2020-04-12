# Getting started with OpenCV 

In this beginner tutorial we learn how to work with the basics
of opencv using python and pycharm. Lets get started.

# Installation

We need to have python installed on our computers. We also need to have pip installed
on our computer.  Once you do this we can get started.

## Installing OpenCv

To install OpenCV we can run the command

```bash
pip install opencv-python
```

Doing this should have opencv install in your current working environment.

# Reading in an image

Working with opencv we need to read in our image. To do this we just called `cv2.imread()` function.
It returns the image in a numpy array format.

```python
image = cv2.imread('dog.png')
```

So we have a simple image called `dog.png`. If we run print on the `image` we could see the results.

```python
image = cv2.imread('dog.png')
print(image)
```

A sample of what a numpy array looks lie is shown below.

[numpyarray_sample.png]

## Viewing an image

After we read in and image we can display this image again using `cv2.imshow`. This takes two arguments
The window name and the numpy array.

```python
import cv2

image = cv2.imread('dog.png')
cv2.imshow("dog", image)
cv2.waitKey(0)
```

Using `cv2.waitKey(0)` stops the dialog from closing before we get to see the image.