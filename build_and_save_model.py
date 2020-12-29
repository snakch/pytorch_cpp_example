import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input:torch.Tensor)->torch.Tensor:
        if input.sum() > 0:
          output = torch.matmul(self.weight ,(input))
        else:
          output = torch.add(self.weight,input)
        return output

my_module = MyModule(10,5)
my_module(torch.ones(5))
my_module(torch.zeros(5))

sm = torch.jit.script(my_module)
sm.save("traced_resnet_model.pt")