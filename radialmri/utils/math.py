import torch


def complex_mult(a, b, dim=0, index=[0, 1]):
    """Complex multiplication, real/imag are in dimension dim."""
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2
    real_a = a.select(dim, index[0]).unsqueeze(dim)
    imag_a = a.select(dim, index[1]).unsqueeze(dim)
    real_b = b.select(dim, index[0]).unsqueeze(dim)
    imag_b = b.select(dim, index[1]).unsqueeze(dim)

    #print('real_a.shape, real_b.shape', real_a.shape, real_b.shape)
    #print('real_a.dtype, real_b.dtype', real_a.dtype, real_b.dtype)
    c = torch.cat(
        (real_a * real_b - imag_a * imag_b, imag_a * real_b + real_a * imag_b),
        dim
    )

    return c


def conj_complex_mult(a, b, dim=0, index=[0, 1]):
    """Complex multiplication, real/imag are in dimension dim.

    This script applies the complex conjugate to b for multiplication.
    """
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2

    real_a = a.select(dim, index[0]).unsqueeze(dim)
    imag_a = a.select(dim, index[1]).unsqueeze(dim)
    real_b = b.select(dim, index[0]).unsqueeze(dim)
    imag_b = b.select(dim, index[1]).unsqueeze(dim)

    c = torch.cat(
        (real_a * real_b + imag_a * imag_b, imag_a * real_b - real_a * imag_b),
        dim
    )

    return c


def imag_exp(a, dim=0):
    """Imaginary exponential, exp(ia), returns real/imag separate in dim."""
    c = torch.stack((torch.cos(a), torch.sin(a)), dim)

    return c


def inner_product(a, b, complex_dim=0):
    inprod = conj_complex_mult(b, a, dim=complex_dim)

    real_inprod = inprod.select(complex_dim, 0).sum()
    imag_inprod = inprod.select(complex_dim, 1).sum()

    return torch.cat((real_inprod.view(1), imag_inprod.view(1)))


def absolute(t, complex_dim=0):
    abst = torch.sqrt(
        t.select(complex_dim, 0) ** 2 +
        t.select(complex_dim, 1) ** 2
    ).unsqueeze(complex_dim)

    return abst
