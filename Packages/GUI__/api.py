from tkinter import *
from tkinter import messagebox


root = Tk()
root.title("FIRST")         # 程序标题
root.geometry("500x300+100+200")        # 程序窗口大小设置500x300  位置+100+200

bt1 = Button(root)          # 添加按键
bt1["text"] = "测试"        # 按键名称
bt1.pack()

def fun(e):
    messagebox.showinfo('Message','正在测试')
bt1.bind("<Button-1>",fun)  # 按键链接

root.mainloop()