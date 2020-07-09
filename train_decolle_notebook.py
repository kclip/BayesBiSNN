import torch
from utils.loss import VRDistance, decolle_loss
from model.LIF_MLP import LIFMLP
from utils import decolle_utils

T = 300
gen_train = decolle_utils.get_mnist_loader(100, Nparts=100, train=True)
gen_test = decolle_utils.get_mnist_loader(100, Nparts=100, train=False)
data, target = next(decolle_utils.iter_mnist(gen_train, T=T))

print(data.shape)

model = LIFMLP(input_shape=data.shape[2:],
               output_shape=10,
               n_neurons=[150, 120],
               num_layers=2)


# specify loss function (categorical cross-entropy)
criterion = torch.nn.SmoothL1Loss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.Adamax(model.get_trainable_parameters(), lr=1e-8, betas=[0., .95])

model.init_parameters()

for e in range(20):
    error = []
    readout_hist = [torch.Tensor() for _ in range(len(model.readout_layers))]

    for data, label in decolle_utils.iter_mnist(gen_train, T=T):
        model.train()
        loss_hist = 0
        model.init(data, burnin=100)
        readout = 0
        for n in range(T):
            st, rt, ut = model.forward(data[n])

            loss_tv = decolle_loss(rt, label[n], criterion)
            loss_tv.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist += loss_tv
            readout += rt[-1]


        error += list((readout.argmax(dim=1) != label[-1].argmax(dim=1)).float())
    print('Training Error', torch.mean(torch.Tensor(error)).data)

    print('Epoch', e, 'Loss', loss_hist.data)
