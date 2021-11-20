import torch

x = torch.randn(4,4)
print(x.size())

y = x.view(16)
print(y.size())

z = x.view(-1,8) # the size -1 is inferred from other dimensions
print(z.size())

a = torch.randn(1,2,3,4)
print(a.size())

b = a.transpose(1,2) # Swaps 2nd and 3rd dimension
print(b.size())

c = a.view(1,3,2,4) # Does not change tensor layout in memory
print(c.size())

print(torch.equal(b, a.transpose(1,2)))
print(torch.equal(b,c))