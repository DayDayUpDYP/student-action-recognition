from torch.nn import *
import torch


class GATLayer(Module):
    def __init__(self, input_features, output_features):
        super(GATLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.W = Parameter(torch.randn(input_features, output_features), requires_grad=True)

        self.mlp = [
            Linear(output_features, 1),
        ]
        self.mlp = Sequential(*self.mlp)

        self.softmax = Softmax(dim=2)
        self.activation = LeakyReLU(negative_slope=0.2)

    def forward(self, p, x):
        h_hat = torch.matmul(x, self.W)

        # h_sum = h_hat.permute([0, 2, 1, 3]) + h_hat * p
        h_sum = h_hat.permute([0, 2, 1, 3])
        h_sum = self.softmax(self.activation(self.mlp(h_sum)))
        h_sum = self.softmax(h_sum)

        h_hat = h_hat * h_sum
        h_hat = torch.sum(h_hat, dim=2)

        return torch.reshape(h_hat, shape=(h_hat.shape[0], 1, h_hat.shape[1], h_hat.shape[2]))


class GAT(Module):
    def __init__(self, input_features, output_features, multi_num):
        super(GAT, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.multi_num = multi_num
        self.activation = LeakyReLU(negative_slope=0.2)
        self.gat_layers = ModuleList()
        for i in range(multi_num):
            self.gat_layers.append(GATLayer(input_features, output_features))
        self.norm = BatchNorm2d(1)

    def forward(self, p, x):
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        result = torch.zeros(size=(x.shape[0], 1, x.shape[2], self.output_features), device=dev)
        for layer in self.gat_layers:
            result += layer(p, x)
        result = result / self.multi_num
        result = self.activation(result)
        return result


if __name__ == '__main__':
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # layer = GATLayer(26, 26)
    h = torch.randn(size=(1, 1, 26, 26), device=dev)
    # out = layer(h)
    # print(out.size())

    gat = GAT(26, 3, 3).to(dev)
    out = gat(h)
    print(out.size())
