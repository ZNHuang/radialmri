import torch

device = torch.device('cuda')
#device = torch.device('cpu')

R = torch.tensor([0], device = device)
I = torch.tensor([1], device = device)

def multiplication(a, b, dim = 0):
    """
    Perform complex multiplication with torch tensor that has the real and
    imaginary part in 2 channels

    (a_real + a_imag * i) * (b_real + b_imag * i)
     = (a_real * b_real - a_imag * b_imag)
     + (a_real * b_imag + a_imag * b_real)* i

    Keyword Arguments:
        a -- torch tensor
        b -- torch tensor, same shape as a
        dim -- the complex dimension in a and b, where there the shape
        is 2
    """
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2

    return torch.cat((torch.index_select(a, dim = dim, index = R) *\
                      torch.index_select(b, dim = dim, index = R) -\
                      torch.index_select(a, dim = dim, index = I) *\
                      torch.index_select(b, dim = dim, index = I), \
                      torch.index_select(a, dim = dim, index = R) *\
                      torch.index_select(b, dim = dim, index = I) +\
                      torch.index_select(a, dim = dim, index = I) *\
                      torch.index_select(b, dim = dim, index = R)), dim)

def multiplication_conjugate(a, b, dim = 1):
    """
    complex multiplication with conjugate a * conjugate(b)

    Keyword Arguments:
        a -- torch tensor
        b -- torch tensor, same shape as a
        dim -- the complex dimension in a and b, where there the shape
        is 2
    """
    return multiplication(a, conjugate(b, dim = dim), dim = dim)

def modulus_square(a, dim = 0):
    assert a.shape[dim] == 2
    shape = torch.index_select(a, dim = dim, index = I).shape
    device = a.device
    dtype = a.dtype
    return torch.cat((torch.index_select(a, dim = dim, index = R)**2 +\
                      torch.index_select(a, dim = dim, index = I)**2,\
                      torch.zeros(shape).to(device, dtype)),
                      dim = dim)

    #return multiplication(a, conjugate(a, dim = dim), dim = dim)

def modulus(a, dim = 0):
    return torch.sqrt(modulus_square(a, dim = dim))

def conjugate(a, dim = 0):
    assert a.shape[dim] == 2
    return torch.cat((torch.index_select(a, dim = dim, index = R),
                      torch.index_select(a, dim = dim, index = I) * (-1)),
                      dim = dim)

def reciprocal(a, dim = 0):
    assert a.shape[dim] == 2
    temp = torch.index_select(modulus_square(a, dim = dim),\
                              dim = dim,\
                              index = R)
    return conjugate(a, dim = dim) / temp #?

def division(a, b, dim = 0):
    return multiplication(a, reciprocal(b, dim = dim), dim = dim)


