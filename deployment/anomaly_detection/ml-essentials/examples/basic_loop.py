import mltk
import numpy as np
import torch
from torch import nn, Tensor, optim
from torch.nn import functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.LogSoftmax(-1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


if __name__ == '__main__':
    # prepare for the data
    (train_x, train_y), (test_x, test_y) = \
        mltk.data.load_mnist(x_shape=[784], x_dtype=np.float32, y_dtype=np.int64)
    train_x /= 255.
    test_x /= 255.

    (train_x, train_y), (valid_x, valid_y) = \
        mltk.utils.split_numpy_arrays([train_x, train_y], portion=0.2,
                                      shuffle=True)

    train_stream = mltk.DataStream.arrays(
            [train_x, train_y],
            batch_size=32,
            shuffle=True,
            skip_incomplete=True). \
        to_torch_tensors().threaded(5)

    val_stream = mltk.DataStream.arrays(
            [valid_x, valid_y], batch_size=128). \
        to_torch_tensors().threaded(5)

    test_stream = mltk.DataStream.arrays(
            [test_x, test_y], batch_size=128). \
        to_torch_tensors().threaded(5)

    predict_stream = mltk.DataStream.arrays(
            [test_x], batch_size=128). \
        to_torch_tensors().threaded(5)

    # construct the model
    model: nn.Module = Net()

    # construct the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # prepare for the train loop
    train_loop = mltk.TrainLoop(max_epoch=10)

    # set the model to `train` mode at the beginning of each epoch,
    # required by PyTorch
    train_loop.on_epoch_begin.do(model.train)

    def train_step(x, y):
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        acc = (torch.argmax(out, dim=-1) == y).to(torch.float32).mean()
        return {'loss': loss, 'acc': acc}

    def test_step(x, y):
        pred = predict_step(x)['y']
        acc = (pred == y).to(torch.float32).mean()
        return {'acc': acc}

    def predict_step(x):
        return {'y': torch.argmax(model(x).detach(), dim=-1)}

    @train_loop.on_epoch_end.do
    def run_validation():
        # set the model to `eval` mode at the beginning of each validation,
        # required by PyTorch
        model.eval()
        train_loop.validation().run(test_step, val_stream)

    # train the model
    train_loop.run(train_step, train_stream)

    # do final test
    model.eval()
    result = mltk.TestLoop().run(test_step, test_stream)
    print(f'test acc: {result["acc"]:.6g}')

    # run predict loop
    model.eval()
    result = mltk.PredictLoop().run(predict_step, predict_stream)
    print(f'predicted y: {result["y"].shape}, {result["y"]}')
