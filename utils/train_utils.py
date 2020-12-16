import torch

def train_on_example(binary_model, optimizer, decolle_loss, inputs, labels, burnin, T):
    optimizer.update_binary_weights()
    binary_model.init(inputs, burnin=burnin)

    readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

    for t in range(burnin, T):
        # forward pass: compute new pseudo-binary weights
        optimizer.update_binary_weights()

        # forward pass: compute predicted outputs by passing inputs to the model
        s, r, u = binary_model(inputs[t])

        for l, ro_h in enumerate(readout_hist):
            readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

        # calculate the loss
        loss = decolle_loss(s, r, u, target=labels[:, :, t])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return readout_hist
