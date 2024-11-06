import numpy
import torch

# copied from https://github.com/scipy/scipy/blob/ae34ce4835949a8310d7c3d7bcb4a55aafd11f4f/scipy/ndimage/filters.py#L135
def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = numpy.arange(order + 1)
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius+1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = numpy.zeros(order + 1)
        q[0] = 1
        D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = numpy.diag(numpy.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x
    
# Copied from https://github.com/scipy/scipy/blob/ae34ce4835949a8310d7c3d7bcb4a55aafd11f4f/scipy/ndimage/filters.py#L167C1-L216C68
def gaussian_filter1d(sigma, order=0, truncate=4.0):
    """1-D Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    gaussian_filter1d : ndarray

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter1d
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(280490)
    >>> x = np.random.randn(101).cumsum()
    >>> y3 = gaussian_filter1d(x, 3)
    >>> y6 = gaussian_filter1d(x, 6)
    >>> plt.plot(x, 'k', label='original data')
    >>> plt.plot(y3, '--', label='filtered, sigma=3')
    >>> plt.plot(y6, ':', label='filtered, sigma=6')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.show()
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return weights
    # return correlate1d(input, weights, axis, output, mode, cval, 0)

def gaussian_filter1d_torch(sigma, order=0, truncate=4.0):
    # [1, 1, kernel_size]
    return torch.from_numpy(gaussian_filter1d(sigma, order, truncate).copy()).float()[None, None, :]


def apply_filter1d_torch(input, kernel):
    # apply the kernel for all channels
    # kernel [1, 1, kernel_size] (created by gaussian_filter1d_torch)
    bs, seqlen, ch = input.shape
    radius = seqlen // 2
    mid_idx = kernel.shape[-1] // 2
    if mid_idx-radius > 0:
        truncated_kernel = kernel[:, :, mid_idx-radius:mid_idx+radius+1].clone() # [1, 1, seqlen+1]
    else:
        truncated_kernel = kernel.clone()
    truncated_kernel /= truncated_kernel.sum()  # normalize
    padded_input = torch.nn.functional.pad(input.permute(0, 2, 1).reshape(bs*ch, 1, seqlen), pad=(radius, radius) , mode='reflect')  # [bs, ch, seqlen*2]
    filtered_result = torch.conv1d(padded_input, truncated_kernel).reshape(bs, ch, -1).permute(0, 2, 1)  # -1 = len() - len(truncated_kernel) + 1
    assert filtered_result.shape == input.shape
    return filtered_result


def apply_one_side_filter1d_torch(input, kernel):
    # apply the kernel for all channels - but only for the last entry - so just one side of the kernel is used
    # kernel [1, 1, kernel_size] (created by gaussian_filter1d_torch)
    # input [bs, seqlen, ch]
    # out [bs, 1, ch]
    bs, seqlen, ch = input.shape
    radius = seqlen
    mid_idx = kernel.shape[-1] // 2
    truncated_kernel = kernel[:, :, max(0, mid_idx-radius+1):mid_idx+1].clone() # [1, 1, radius]
    truncated_kernel /= truncated_kernel.sum()  # normalize
    truncated_input = input.permute(0, 2, 1).reshape(bs*ch, 1, seqlen)[:, :, -truncated_kernel.shape[-1]:] # [bs*ch, 1, radius]
    filtered_last_entry = torch.sum(truncated_input * truncated_kernel, dim=-1)  # [bs*ch, 1]  # dot product
    return filtered_last_entry.reshape(bs, 1, ch)

