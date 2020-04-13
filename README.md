# Getting started with OpenCV 

In this beginner tutorial we learn how to work with the basics
of opencv using python and pycharm. Lets get started.

# Installation

We need to have python installed on our computers. We also need to have pip installed
on our computer.  Once you do this we can get started. 

> You can also use anaconda or any other package manager to load your environment. This may be the
> hardest part. Once you get your environment setup. You are good to go.

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

## Getting image shape

We can get the image shape by calling `image.shape`. Where `image` is the image we read in using
`cv2.imread()` function.

Let see this in action.

```python
import cv2

image = cv2.imread('dog.png')
print(image.shape)
```

The results you will see is

```python
(547, 512, 3)
```

Where 547 is the height, 512 is the width and 3 is the number of channels.

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
Show we can see the image in a dialog with the title we put in the first argument.

[showing_image.png]

## Alternative ways to view image

We can use `matplot` to view images  as well lets see how.

First install matplotlib

```bash
python3 pip install matplotlib
```

Now we import it

```python
from matplotlib import pyplot as plt
```

Next we add the code to view the image


```python
image = cv2.imread('dog.png')
plt.imshow(image)
plt.show()
```

The results is shown below. The colors will be off but don't worry about that.
You can always change the color scale afterwards.

[image_using_matplot.png]

# Changing the color scale

You can change the color scale of an image quit easily. Lets see how.
Lets change our image to gray scale. To do this we call the 
`cv2.cvtColor` function and pass in the image and `cv2.COLOR_BGR2GRAY` option.
The code is shown below.

```python
image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("dog", gray_image)
cv2.waitKey(0)
```

The results can be seen here.

[grey_dog_image.png]

## From RGB to BGR

Remember when we use `matplotlib`. The image showed in the scale `GBR`.
Lets try to get the same results. Below we add the option `cv2.COLOR_RGB2BGR`

```python
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
```

The results is shown below.

[bgr_dog_image.png]

## From BGR to RGB

So we can always convert our matplot image using

```python
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

Play with these options to see the different results you can get.

## Converting to HSV

A color space to note is `HSV`. We can create comparsions between image with it.
Lets convert our image to this

```python
import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("dog", gray_image)
cv2.waitKey(0)
```

The results is shown below.

[dog_hsv_space.png]

# Resizing images

Lets see how we can resize image. To do this we need to run the `cv2.resize()` function.

```python
import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300)) # resize image
cv2.imshow("dog", small_image)
cv2.waitKey(0)
```

Above we call the `cv2.resize` function. We pass the image we want to resize then we pass
a tuple `(300, 300)` with the width and height as parameters.

The results is shown below.

[resized_image_300.png]

# Saving images

We can save an image by call the `cv2.imwrite()` function.
It takes the name of the image and the object to save. Lets see the code below

```python
import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300))
cv2.imwrite("dogtest.png", small_image)
```

So we called `imwrite` and named our file `dogtest.png`. Lets see the results.

[save_image.png]

# Thresholding

This is an important part of working with images in Opencv. You can learn more about it
[here on wikipedia](https://en.wikipedia.org/wiki/Thresholding_(image_processing)) or
[here on opencv docs](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)

But write now let me show you what it does.

```python
import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300))
ret, thresh_image = cv2.threshold(small_image, 127, 255, cv2.THRESH_BINARY) # thresholding
cv2.imshow("dog", thresh_image)
cv2.waitKey(0)
```

In the above code. We did things we already showed you. We read the image;
We turned it gray. Then we resized it. **THEN lets get into**

We ran the `cv2.threshold` function. We passed first the `image`. The we passed `127`
which is the **threshold** we can change the value from 0 - 255. Try it out. Next
we have `255` leave this. And finally we have the option `cv2.THRESH_BINARY`. We can 
modify this last value.

The results is shown below.

**At threshold 127**

[threshold_127.png]

**At threshold 170**

[threshold_170.png]

**At threshold 100**

[threshold_100.png]

## Thresholding with text

You tend to see thresholding value better if our image as text in it. Lets
try with this image.

[car_plate.png]

Now focus on the registration plate. We will run thresholding on it. And you can see the results
more clearly.

```python
import cv2

image = cv2.imread('car.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300))
ret, thresh_image = cv2.threshold(small_image, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("reg plate", thresh_image)
cv2.waitKey(0)
```

Using threshold value at **100**

[carplate_thres_100.png]

We raise the threshold to **150** and the text in our image gets grainy

[grainy_thress_lplate.png]

## Other types of thresholding

Lets try out different versions/methods of thresholding. They give different results.
You can learn more about the technical details [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html).

Lets get into it

### The inverse threshold

Using `cv2.THRESH_BINARY_INV` we get the reverse threshold. Black goes to white. White goes to black.

```python
ret, thresh_image = cv2.threshold(small_image, 100, 255, cv2.THRESH_BINARY_INV)
```

The results is shown below.

[reverse_threshold.png]

### The elephant truck

Lol. WE can use `cv2.THRESH_TRUNC` to truncate values that reach our limit.

```python
ret, thresh_image = cv2.threshold(small_image, 100, 255, cv2.THRESH_TRUNC)
```

The results is shown below.

[car_truncate_thres.png]

There really are alot of options. So I can go through them.

### Using ADAPTIVE_THRESH_GAUSSIAN_C

This last option I am fond off. Because it has been effective for me. Use in the similar way.

```python
ret, thresh_image = cv2.threshold(small_image, 100, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
```

The results is shown below. This gave me the best results when running ocr.

[adaptive_thres_car.png]

# Blurring your images

Blurring smooths your image and makes it easier to read for the computer.
We can call the `cv2.GaussianBlur` function to blur our images.
The code is shown below

```python
import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300))
blurimg = cv2.GaussianBlur(small_image, (7, 7), 0) # blur image
cv2.imshow("dog", small_image)
cv2.imshow("blurred dog", blurimg)
cv2.waitKey(0)
```

The results is shown below.

[bluring_images.png]

Using the `cv2.GaussianBlur` function we pass the image and a tuple you can change these values

```python
blurimg = cv2.GaussianBlur(small_image, (3, 3), 0) # blur image
blurimg = cv2.GaussianBlur(small_image, (5, 5), 0) # blur image
blurimg = cv2.GaussianBlur(small_image, (7, 7), 0) # blur image
```

Play with these values to get different results. You are passing in a kernel that modifies how
the function blurs your image. Learn more about it
[here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html).

# Drawing rectangles

Let see how we can draw rectangles on your image.
To do this we use the `cv2.rectangle` function.
An example is shown below

```python
cv2.rectangle(image, (50, 100), (100, 200), (0, 255, 0))
```

So above we have the source image called `image`. The we have a tuple for 
the start point (x, y) coordinates `(50, 100)`. Then the end point with 
(x, y) coordinates `(100, 200)`. Finally we have set the color of the via
tuple rgb. So green is `(0, 255, 0)`.

The results is shown below.

[draw_rectangle.png]

Lets try to get the nose inside the box. Or the box over the nose.

```python
import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
cv2.rectangle(small_image, (150, 80), (230, 150), (0, 255, 0))
cv2.imshow("dog", small_image)
cv2.waitKey(0)
```

The results is shown below. If you play with this long enough. To move your box horizontally play with the 
x values in the starting and ending points. To move your box vertically play with the y values in the starting and
ending points.

[bounding_box_nose.png]

# Cropping Images

So you can now draw rectangles. Lets see how we can crop an image.
To crop an image we only need to do this as shown below

```python
cropped = orig[startY:endY, startX:endX]
```

Where `cropped` is the finally image. `orig` is the original image. `startY` and `endY` are y coordinates.
`startX` and `endX` are x coordinates.

The full code can be seen here

```python
import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
cv2.rectangle(small_image, (150, 80), (230, 150), (0, 255, 0))
cropped = small_image[80:150, 150:230]
cv2.imshow("dog", small_image)
cv2.imshow("cropped image", cropped)
cv2.waitKey(0)
```

The results is shown below.

[cropping_images.png]

# Rotating your images

Lets see how we can rotate images with opencv. We can use `cv2.warpAffine()`
to rotate our images. You can learn more [here](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html)

Lets get started.

```python
image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
(h, w) = small_image.shape[:2] # get height and width
center = (w / 2, h / 2) # divide by 2
M = cv2.getRotationMatrix2D(center, 30, 1.0) # get matrix
rotated = cv2.warpAffine(small_image, M, (h, w)) # rotate image
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
```

So we get our matrix using `cv2.getRotationMatrix2D(center, 30, 1.0)`.
The `30` is the angle. We can change this to rotate the image to different degrees.

The results is shown below.

**For 30 **

[rotate_image_30.png]

**For 90**

[roate_image_90.png]


# Conclusion

Alrite. We just completed a basic introduction into opencv. You have the basics to get you started with a 
whole world of computer vision.

Below I play with some more concepts that you might have heard of.


# Edges in images

You can using `cv2.Canny` to view edges in images.
The example is shown below

```python
template = cv2.Canny(small_image, 150, 200)
```

You can change the `150` and `200`. Play with these values to see the variations.
The full code is below

```python
import cv2

image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
template = cv2.Canny(small_image, 150, 200)
cv2.imshow("template", template)
cv2.imshow("dog", small_image)
cv2.waitKey(0)
```

The results is shown below

[egde_detection.png]

## Playing with Edges

Lets get a new image. A sample crossword image.

[simple_crossword.png]

Now lets add edge detection on it.

```python
import cv2

image = cv2.imread('crossword.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
template = cv2.Canny(small_image, 175, 200)
cv2.imshow("template", template)
cv2.imshow("normal", small_image)
cv2.waitKey(0)
```

The results is shown below.

[edges_in_crossword.png]

## Drawing contours

We can draw the contours we find using edge detection.

```python
import cv2

image = cv2.imread('crossword.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
template = cv2.Canny(small_image, 175, 200)
contours, hierarchy = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("template", template)
cv2.drawContours(small_image, contours, -1, (0, 255, 0), 3)
cv2.imshow("draw contours", small_image)
cv2.waitKey(0)
```

The results is shown below.

[drawing_contours.png]


