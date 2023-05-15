import cv2,numpy
imgs = cv2.imread(r'D:\software\Code\code-file\final\test\RawImagePart.png')
zeros = numpy.ones(imgs.shape[:2],dtype=imgs.dtype)*255
result = cv2.merge([imgs,zeros])
cv2.imwrite(r'D:\software\Code\code-file\final\test\RawImagePart2.png',result)
a =cv2.imread(r'D:\software\Code\code-file\final\test\RawImagePart2.png',cv2.IMREAD_UNCHANGED)
print(a.shape)