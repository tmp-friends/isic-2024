import torch
import torch.nn as nn

sigmoid = nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)

        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class SwishModule(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
