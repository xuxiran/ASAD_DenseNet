import torch
from model_2D import DenseNet_2D
from model_3D import DenseNet_3D
import torch.nn as nn
import config as cfg
# 用固定随机种子生成(1,10,11)的torch数据
# 随机数固定
torch.manual_seed(0)
X = torch.rand((8, 10, 11))
X_3D = X.unsqueeze(1)
# 将X在第1维度重复128次，从而变成(8,128,10,11)
X_3D = X_3D.repeat(1,128, 1, 1)
# 在第0维度增加一个维度，变成(1,128,10,11)



saveckpt = './fold0.ckpt'
model_2D = DenseNet_2D().to(cfg.device)

# 加载模型到cpu上
model_2D.load_state_dict(torch.load(saveckpt, map_location='cpu'))

model_3D = DenseNet_3D().to(cfg.device)

import torch

# 假设您已经加载了2D网络的权重，并且已经构建了一个3D网络
weights_2D = model_2D.state_dict()# torch.load('./fold0.ckpt', map_location='cpu')  # 加载2D网络的权重
# weights_3D为模型model_3D的state_dict
weights_3D = model_3D.state_dict()  # 获取3D网络的权重

# 将2D网络的权重复制到3D网络上
# 将2D网络的权重复制到3D网络上
for name, param in weights_2D.items():
    # 获取2D网络权重的形状
    shape_2D = param.shape
    # 获取3D网络权重的形状
    shape_3D = weights_3D[name].shape
    # 如果shape_2D和shape_3D大小相同，则直接复制
    if shape_2D == shape_3D:
        weights_3D[name] = param
    # 如果shape_2D和shape_3D大小不同，则生成param_3D,是将param倒数第三个维度重复shape_3D倒数第三个维度大小次，并将值也变成倒数第三个维度大小分之一
    else:
        param_3D = torch.zeros(shape_3D)
        for i in range(shape_3D[-3]):
            param_3D[:,:,i, :, :] = param / shape_3D[-3]
        weights_3D[name] = param_3D


# 将复制好的权重加载到3D网络上
model_3D.load_state_dict(weights_3D)


pred_2D = model_2D(X)
pred_3D = model_3D(X_3D)

diff_pred = pred_3D- pred_2D

print(diff_pred)

