import json,os,re
import matplotlib.pyplot as plt

json_path = r'D:\31890\Desktop\codefile\Utils\Packages\Read_log__\log\info.log'
with open(json_path, 'r',encoding='utf-8') as log_file:
    mean_l=[]
    list =[]
    [list.append([]) for i in range(100)]
    # print(mean_l)
    # print(list)
    for line in log_file:
        if '---Trian---' in line:
            match1 = re.search(r'epoch: (\d+)', line)
            match2 = re.search(r'loss: (\d+(\.\d+)?)', line)
            epoch = int(match1.group(1))
            num = float(match2.group(1))
            list[epoch-1].append(num)
    for i in range(len(list)):
        mean = sum(list[i])/len(list[i])
        mean_l.append(mean)



    x  = [i for i in range(1, 101)]
    plt.figure()
    plt.plot(x, mean_l)

    # plt.savefig(r'D:\software\Code\code-file\T\acc.jpg',dpi = 200, bbox_inches = 'tight')
    plt.show()
            