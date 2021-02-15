# Convert Mapillary Vistas Dataset to Coco format

> This python script helps you convert your mapillary vistas dataset to coco format.

# Brief Introduction

Here is a given `instance` image. Label info is embeded into each pixel value.

```python
pixel / 256 # the value represents this pixel belongs to which label.
pixel % 256 # the value represents this pixel is the i-th instance of its label.
```

# Download Link

You can download the training and validation set in CoCo format (in JSON annotations) from the following links.

[1] [Training Set](https://drive.google.com/file/d/1JFJGM1fMB07fvdgRXQqGKqXaoEDh4DL9/view?usp=sharing)

[2] [Validation Set](https://drive.google.com/file/d/1ZJWNb-u9JQgBjKknT2RtNurtlnqvPFgf/view?usp=sharing)
