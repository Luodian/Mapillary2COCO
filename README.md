# Transfer Mapillary Vistas Dataset to Coco format

> This python script can help you transfer your MVD to coco format.

Currently we are working at MVD & ECCV 2018 competition, one of our plans is to transfer the dataset to coco format. And use currently prevalent frame implementation(mask RCNN) to train our model.

## Brief Introduction

`SplitTools.py` finds `label x instances` tuple in each given `instance` image.

Here is a given `instance` image. Label info is embeded into each pixel value.

```python
pixel / 256 # the value represents this pixel belongs to which label.
pixel % 256 # the value represents this pixel is the i-th instance of its label.
```

## Examples

Here we have a `instance` image.

![](https://ws2.sinaimg.cn/large/006tKfTcly1ft2l171m14j31kw16o0tr.jpg)

we can split it according to `label x instance` relations.

`1_41_0_Manhole` is the location about the 0-th Manhole(Manhole's label_id is 41).

![](https://ws2.sinaimg.cn/large/006tKfTcly1ft2l70ay7bj31je144760.jpg)

And then we use pycococreator tools to transfer them into json format, which is fit for coco data format.

I put the tools in `Image2Json` directory, and also you can download the tools here: [Pycococreator tools](https://github.com/waspinator/pycococreator)





