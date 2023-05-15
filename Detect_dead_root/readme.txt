findPixDis文件夹为从原始图像中截取的部分,用于判断活根和死根的差异.
    1.png为活,2.png和3.png为死

test文件夹为测试活根和死根的一个图像,RawImagePart.png为原始的测试数据,其上部代表活,下部代表4.

forTest.py文件用于测试test/RawImagePart.png的死根去除的效果. 其效果参见 test/RawImagePartb5.png文件


baseImg/rawImg.png是所发送的基础图像,运行forRaw.py文件,会得到去去掉死根的图像.由于rawImg.png文件的维度较大,其运行较慢,最终运行结果保存的文件为baseImg/rawImgb5.png