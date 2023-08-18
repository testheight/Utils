from base64 import  b64encode
from json import  dumps

image_path = r"D:\31890\Desktop\root\16-20220704-1200-0201.jpg"

# 读取二进制图片，获得原始字节码
with open(image_path, 'rb') as jpg_file:
    byte_content = jpg_file.read()

# 把原始字节码编码成base64字节码
base64_bytes = b64encode(byte_content)

# 把base64字节码解码成utf-8格式的字符串
base64_string = base64_bytes.decode('utf-8')

# 用字典的形式保存数据
dict_data = {}
dict_data['name'] = image_path
dict_data['imageData'] = base64_string

# 将字典变成json格式，缩进为2个空格
json_data = dumps(dict_data, indent=2)

# 将json格式的数据保存到文件中
with open('test.json', 'w') as json_file:
    json_file.write(json_data)
