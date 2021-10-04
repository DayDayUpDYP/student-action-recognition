from torch.nn import *
import torch


class GATLayer(Module):
    def __init__(self, input_features, output_features):
        super(GATLayer, self).__init__()
        self.W = Parameter(torch.randn(input_features, output_features), requires_grad=True)

        self.mlp = [
            Linear(output_features, 1),
        ]
        self.mlp = Sequential(*self.mlp)

        self.softmax = Softmax(dim=2)
        self.activation = LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        h_hat = torch.matmul(x, self.W)

        h_sum = h_hat + h_hat.permute([0, 2, 1, 3])
        h_sum = self.softmax(self.activation(self.mlp(h_sum)))

        h_hat = h_hat * h_sum
        h_hat = self.activation(torch.sum(h_hat, dim=1))

        return torch.reshape(h_hat, shape=(h_hat.shape[0], 1, h_hat.shape[1], h_hat.shape[2]))


class GAT(Module):
    def __init__(self, input_features, output_features, multi_num):
        super(GAT, self).__init__()
        self.gat_layers = ModuleList()
        for i in range(multi_num):
            self.gat_layers.append(GATLayer(input_features, output_features))

    def forward(self, x):
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        result = torch.zeros(size=x.shape, device=dev)
        for layer in self.gat_layers:
            result += layer(x)
        return result


if __name__ == '__main__':
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # layer = GATLayer(26, 26)
    h = torch.randn(size=(32, 1, 26, 26), device=dev)
    # out = layer(h)
    # print(out.size())

    gat = GAT(26, 26, 3).to(dev)
    out = gat(h)
    print(out.size())
