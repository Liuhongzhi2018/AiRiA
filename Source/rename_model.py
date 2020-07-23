from collections import OrderedDict
import time
import torch
from torch import nn

# rename dict
# adict=torch.load('./ckpt_max.pth',map_location='cpu')
# print(adict['state_dicts']['module.backbone.conv1.weight'].shape)
# new_state_dict = OrderedDict()
# for k, v in adict['state_dicts'].items():
#     head = k[:7]
#     if head == 'module.':
#         name = k[7:] # remove `module.`
#     else:
#         name = k
#     new_state_dict[name] = v
# model.load_state_dict(new_state_dict)
# torch.save(net.state_dict(),'./model-dict.pth')

# adict = torch.load('/home/zhuminchen/liuhongzhi/pretrained/mobilenetv2_final_7027.pth',map_location='cpu')
adict = torch.load('/home/zhuminchen/liuhongzhi/pretrained/mobilenetv2_0723.pth')
new_state_dict = OrderedDict()
relist = ['00',
        '10','11',
        '20','21','22',
        '30','31','32','33',
        '40','41','42',
        '50','51','52',
        '60',]
n = 0
sign = 0
for k, v in adict.items():
    print(k)
    if 'mobilenet.conv1' in k:
        #print('*'*3," modify conv1")
        leg = len('mobilenet.conv1')
        name = 'backbone.features.0.0' + k[leg:]
        print('*'*5," new name: ",name)
        time.sleep(1)
    elif 'mobilenet.bn1' in k:
        #print('*'*3," modify bn1")
        leg = len('mobilenet.bn1')
        name = 'backbone.features.0.1' + k[leg:]
        print('*'*5," new name: ",name)
        time.sleep(1)
    elif 'mobilenet.bottlenecks.Bottlenecks_' in k:
        s1 = k[len('mobilenet.bottlenecks.Bottlenecks_')]
        s2 = k[len('mobilenet.bottlenecks.Bottlenecks_0.LinearBottleneck0_')]
        tail = k.split('.')[-1]
        #print("tail: ", tail)
        loc = s1+s2
        # print("combine: ",loc)
        c = relist.index(loc)
        curr = c + 1
        #print("loc in relist: ",curr)
        if curr != sign and tail == 'weight':
            sign = curr
            n = 0
        # if curr == sign and tail == 'weight':
        #     n = n + 1
        # if tail == 'num_batches_tracked':
        #     n = n + 1
            name = 'backbone.features.'  + str(curr) + '.conv.' + str(n) + '.' + tail
        elif curr == sign and tail == 'weight':
            n = n + 1
            name = 'backbone.features.'  + str(curr) + '.conv.' + str(n) + '.' + tail
        else:
            name = 'backbone.features.'  + str(curr) + '.conv.' + str(n) + '.' + tail
        if tail == 'num_batches_tracked':
            n = n + 1
        print('*'*5," new name: ",name)
        time.sleep(1)
    elif 'mobilenet.conv_last' in k:
        name = 'backbone.conv.0.' + k.split('.')[-1]
        print('*'*5," new name: ",name)
    elif 'mobilenet.bn_last' in k:
        name = 'backbone.conv.1.' + k.split('.')[-1]
        print('*'*5," new name: ",name)
    elif 'extras' in k:
        s2, s3 = k.split('.')[1], k.split('.')[2]
        name = 'backbone.extra_convs.' + s2 + '.conv.' + s3 + '.' + k.split('.')[-1]
        print('*'*5," new name: ",name)
    # elif 'loc' in k:
    #     s2, s3 = k.split('.')[1], k.split('.')[2]
    #     name = 'bbox_head.reg_convs.' + s2 + '.' + s3
    #     print('*'*5," new name: ",name)
    # elif 'conf' in k:
    #     s2, s3 = k.split('.')[1], k.split('.')[2]
    #     name = 'bbox_head.cls_convs.' + s2 + '.' + s3
    #     print('*'*5," new name: ",name)
    else:
        name = k
    new_state_dict[name] = v

# newmodel= torch.load('/home/zhuminchen/liuhongzhi/pretrained/mobilenetv2_0723.pth')
# newmodel.load_state_dict(new_state_dict) # 接着就可以将模型参数load进模型。
# torch.save(net.state_dict(),'./model-dict.pth')
torch.save(new_state_dict,'/home/zhuminchen/liuhongzhi/pretrained/mobilenetv2_0723.pth')


# visual conv layers
# x = torch.rand((1, 1280, 10, 10))
# a=nn.Conv2d(1280,6,3,padding=6,dilation=6)
# x=a(x)
# print(x.shape)

# load_dict = torch.load('/home/zhuminchen/liuhongzhi/pretrained/mobilenetv2_final_7027.pth')
# f = open('/home/zhuminchen/liuhongzhi/pretrained/mv2_model.txt','w')
# print(load_dict.keys(),file=f)
# f.close()

print('*'*6,'Load checkpoint','*'*6)
# checkpoint = torch.load('/home/zhuminchen/liuhongzhi/Slim_Airiacvlib_0715/work_ljq_20200722_2/epoch_88.pth')
# checkpoint = torch.load('/home/zhuminchen/liuhongzhi/pretrained/mobilenetv2_final_7027.pth')
checkpoint = torch.load('/home/zhuminchen/liuhongzhi/pretrained/mobilenetv2_0723.pth')
# checkpoint_dict = checkpoint.eval()
# f = open('/home/zhuminchen/liuhongzhi/pretrained/mv2_model_dict.txt','a')
f = open('/home/zhuminchen/liuhongzhi/pretrained/new_model_dict.txt','a')
# for k, v in checkpoint['state_dict'].items():
for k, v in checkpoint.items():
    print(k,' ',v.shape,file=f)
    # print('*'*6)
f.close()
print('*'*6,'Finish checkpoint','*'*6)

