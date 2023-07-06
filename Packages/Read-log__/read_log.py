import json
import os
import matplotlib.pyplot as plt

json_path = r'D:\31890\Desktop\codefile\Utils\Read-log\log\info.log'
with open(json_path, 'r') as log_file:
    for line in log_file:
        log = json.loads(line.strip())
        if 'epoch' not in log:
            continue
        for k,v in log.items():
            if k == 'iter' and ('time' in log) and ('iter' in log) and ('loss' in log):
                iter.append(v)
            if k == loss_p and ('time' in log) and ('loss' in log) and ('iter' in log):
                loss.append(v)





loss_p = 'decode.acc_seg'


log_dicts = [dict() for _ in os.listdir(json_path)]
log_name = []
for json_log,log_dict in zip(os.listdir(json_path),log_dicts):
    log_name.append(json_log.split('.')[0])
    loss=[]
    iter = []
    with open(os.path.join(json_path,json_log), 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            if 'epoch' not in log:
                continue
            for k,v in log.items():
                if k == 'iter' and ('time' in log) and ('iter' in log) and ('loss' in log):
                    iter.append(v)
                if k == loss_p and ('time' in log) and ('loss' in log) and ('iter' in log):
                    loss.append(v)
    log_dict['iter']=iter
    log_dict[loss_p]=loss

print(log_name)


x = log_dicts[0]['iter']
loss1 = log_dicts[0][loss_p]
loss2 = log_dicts[1][loss_p]
loss3 = log_dicts[2][loss_p]
loss4 = log_dicts[3][loss_p]
loss5 = log_dicts[4][loss_p]
loss6 = log_dicts[5][loss_p]
x2 = log_dicts[5]['iter']


print(len(loss5))
fig, ax = plt.subplots(figsize=(18,9))
ax.plot(x, loss1, c='blue', linestyle=':',linewidth=1, label=log_name[0])
ax.plot(x, loss2, c='red', linestyle='-', linewidth=1,label=log_name[1])
ax.plot(x, loss3, c='green', linestyle='-', linewidth=1,label=log_name[2])
ax.plot(x, loss4, c='yellow', linestyle='-',linewidth=1, label=log_name[3])
ax.plot(x, loss5, c='magenta', linestyle='-',linewidth=1, label=log_name[4])
ax.plot(x2, loss6, c='cyan', linestyle='-',linewidth=1, label=log_name[5])


# #设置图例并且设置图例的字体及大小
ax.set(ylabel='acc', xlabel='iter', title='acc')

ax.legend()
plt.savefig(r'D:\software\Code\code-file\T\acc.jpg',dpi = 200, bbox_inches = 'tight')
plt.show()
