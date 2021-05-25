import torch.nn as nn
from torch.autograd import Variable as V
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
from root import ROOT

dataset = MAG240MMINIDataset(ROOT)
device = 'cuda:5'
###### 读入数据
x = np.load(f'{dataset.dir}/full_weighted_feat.npy')[:dataset.num_papers]

###### 对输入进行归一化，因为autoencoder只用到了input
MMScaler = MinMaxScaler()
x = MMScaler.fit_transform(x)

###### 输入数据转换成神经网络接受的dataset类型，batch设定为10
tensor_x = torch.from_numpy(x.astype(np.float32))
tensor_y = torch.from_numpy(np.zeros(x.shape[0]))
my_dataset = TensorDataset(tensor_x, tensor_y)
my_dataset_loader = DataLoader(my_dataset, batch_size=10000, shuffle=False)


###### 定义一个autoencoder模型
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 768),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


model = autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 如果采用SGD的话，收敛不下降

for epoch in range(300):
    total_loss = 0
    for i, (x, y) in enumerate(my_dataset_loader):
        _, pred = model(V(x).to(device))
        loss = criterion(pred, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    if epoch % 100 == 0:
        print(total_loss.data.numpy())

###### 基于训练好的model做降维并可视化

x_ = []
y_ = []
for i, (x, y) in enumerate(my_dataset):
    _, pred = model(V(x))
    # loss = criterion(pred, x)
    dimension = _.data.numpy()
np.save(f'{dataset.dir}/paper_autoencoder_feat.npy', dimension)

