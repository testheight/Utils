import json,os,re,openpyxl
import pandas as pd
import matplotlib.pyplot as plt

def read_log(json_path):
    with open(json_path, 'r',encoding='utf-8') as log_file:
        mean_loss=[]
        loss_list =[]
        [loss_list.append([]) for i in range(100)]

        for line in log_file:
            if '---Trian---' in line:
                match1 = re.search(r'epoch: (\d+)', line)
                match2 = re.search(r'loss: (\d+(\.\d+)?)', line)

                epoch = int(match1.group(1))
                num = float(match2.group(1))
                loss_list[epoch-1].append(num)
        
        for i in range(len(loss_list)):
            mean = sum(loss_list[i])/len(loss_list[i])
            mean_loss.append(mean)
        
    return mean_loss

if __name__ =="__main__":

    file = r'D:\31890\Desktop\codefile\Utils\Packages\Read_log__\log'
    ditA = {}
    for file_name in os.listdir(file):
        list = read_log(os.path.join(file,file_name))
        file_name = file_name.split('.')[0]
        ditA[file_name] = list
    data = pd.DataFrame(ditA)

    data.to_excel('Packages/Read_log__/loss.xlsx', index=False)
            