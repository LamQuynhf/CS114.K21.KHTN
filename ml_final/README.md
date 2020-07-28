# Environment

## Platform: python3

## Libraries:pygame,skimage,numpy

### Hãy chắc chắn rằng bạn đã cài đặt thư viện scikit-image và numpy, nếu chưa hãy cài đặt:


```
python3 -m pip install scikit-image==0.14.2 
```


```
python3 -m pip install numpy==1.15
```

# Demo video

## File: video_demo.ogv

# Chạy chương trình


```
python3 run_demo.py
```

# Quy trình

## Chụp ảnh gốc

### Quay video từ màn hình và chọn vùng quan tâm


```
frame=frame[100:400,400:700]
```

### Đưa bàn tay nằm trọn vào trong khung hình hiện trên screen



<figure>
<center>
<img src='https://i.postimg.cc/13sGXSTj/Screenshot-from-2020-07-19-00-17-19.png' />
<figcaption>Original image</figcaption></center>
</figure>



## Background subtraction

### Sử dụng background subtraction method được gọi là Gaussian Mixture-based Background/Foreground Segmentation Algorithm để trừ nền.


### Ở đây tôi sử dụng  function BackgroundSubtractorMOG2 của opencv để trừ nền


```
bgModel = cv2.BackgroundSubtractorMOG2(0, bgSubThreshold)
```

### Áp dụng vào từng frame


```
fgmask = bgModel.apply(frame,learningRate=learningRate)
```

### Lấy foreground(hand) image


```
res = cv2.bitwise_and(frame, frame, mask=fgmask)
```

#### Ảnh lấy được


<figure>
<center>
<img src='https://i.postimg.cc/TYVDbSv5/Screenshot-from-2020-07-19-00-17-23.png' />
<figcaption>get foreground</figcaption></center>
</figure>


## Gaussian blur & Threshold

### Convert gray scale


```
gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
```

### Dùng Gaussian blur 


```
blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
```

<figure>
<center>
<img src='https://i.postimg.cc/pL1LHSWF/Screenshot-from-2020-07-28-21-38-57.png' />
<figcaption>blur</figcaption></center>
</figure>


### Chuyển về ảnh nhị phân


```
ret, thresh = cv2.threshold(blur,threshold, 255, cv2.THRESH_BINARY)
```


<figure>
<center>
<img src='https://i.postimg.cc/C5Z4cXxs/Screenshot-from-2020-07-28-21-44-21.png' />
<figcaption>binary image</figcaption></center>
</figure>



<figure>
<center>
<img src='https://i.postimg.cc/XJMbQSLn/Screenshot-from-2020-07-28-21-58-36.png' />
<figcaption>image</figcaption></center>
</figure>



```

```
