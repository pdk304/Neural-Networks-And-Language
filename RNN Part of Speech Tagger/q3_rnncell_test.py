import numpy as np
import torch
from q3_model import RNNPOSCell

def test_rnn_cell():
    cell = RNNPOSCell(3,2)
    weights = cell.hidden.weight.detach().numpy()
    print('weights', weights)

    bias = cell.hidden.bias.detach().numpy()[:,np.newaxis]
    print('bias', bias)

    x = np.array([[0.4, 0.5, 0.6],
                  [0.6, 0.7, -0.2],
                   [.2,8,0],
                 [1,2,3]], dtype=np.float32)
    h = np.array([[0.5,.5],
                  [-0.3,.6],
                  [-2,.7],
                 [.5,.8]], dtype=np.float32)

    result = 1./(1.+np.exp(-1*(np.matmul(weights, np.hstack((x,h)).T) + bias)))
    cell_result = (cell(torch.tensor(x),torch.tensor(h))).detach().numpy().T
    print("numpy result = ", result)
    print("cell result = ", cell_result)

    assert np.allclose(result, cell_result), "output and state are not equal."
    print("RNN Cell Check passed!")

if __name__ == "__main__":
    test_rnn_cell()