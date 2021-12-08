import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn

from models.Autoencoder import Autoencoder
from torchvision import datasets, transforms
from scipy.stats import norm

bs = 4
test_ds = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
test_dl = t.utils.data.DataLoader(dataset=test_ds, batch_size=bs, shuffle=False, drop_last=True)

device = t.device('cuda') if t.cuda.is_available() else 'cpu'
model = Autoencoder(test_ds[0][0][None], in_c=1, enc_out_c=[32, 64, 64, 64],
                    enc_ks=[3, 3, 3, 3], enc_pads=[1, 1, 0, 1], enc_strides=[1, 2, 2, 1],
                    dec_out_c=[64, 64, 32, 1], dec_ks=[3, 3, 3, 3], dec_strides=[1, 2, 2, 1],
                    dec_pads=[1, 0, 1, 1], dec_op_pads=[0, 1, 1, 0], z_dim=2)
# model.cuda(device)
model.load_state_dict(t.load('models/state_dicts/03_01.pth', map_location=t.device('cpu')))
model.eval()

t.set_grad_enabled(False)

examples = next(iter(test_dl))
x, y = examples[0], examples[1]
encoder = model.enc_conv_layers
decoder = model.dec_conv_layers

z_points = encoder(examples[0].to(device))
reconst_imgs = decoder(z_points)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(bs):
    img = x[i].squeeze()  # x=[4, 1, 28, 28]->[4, 28, 28] x[i]=[1,28,28]->[28,28]
    ax = fig.add_subplot(2, bs, i+1)  # 2*4的子图 add_subplot(nrows, ncols, index, **kwargs)
    ax.axis('off')
    ax.text(0.5, -0.35, str(np.round(z_points[i].cpu(), 1)), fontsize=10, ha='center', transform=ax.transAxes)  # 0.5，-0.35控制文字位置
    ax.imshow(img, cmap='gray_r')

for i in range(bs):
    img = reconst_imgs[i].cpu().squeeze()
    ax = fig.add_subplot(2, bs, i + bs + 1)
    ax.axis('off')
    ax.imshow(img, cmap='gray_r')

plt.show()