from tinygrad import Tensor
import numpy as np

def conv2d_explicit_broadcast(x, K, pad):
    B, C, H, W = x.shape
    O, Cw, KH, KW = K.shape
    assert C == Cw, "in channels must match"
    assert KH == 3 and KW == 3, "this assumes 3x3"

    x_pad = x.pad(((0,0), (0,0), (pad,pad), (pad,pad)))
    patches = [] 
    for kh in range(KH):
        row = [] 
        for kw in range(KW):
            row.append(x_pad[:, :, kh:kh+H, kw:kw+W])
        row = Tensor.stack(row, dim=0)
        patches.append(row)
    patches = Tensor.stack(patches, dim=0)
    patches = patches.permute(2,0,1,3,4,5)

    x_exp = patches.permute(0,4,5,3,1,2)
    x_exp = x_exp.reshape(B, 1, H, W, C, KH, KW)

    K_exp = K.reshape(1, O, 1, 1, C, KH, KW)

    return (x_exp * K_exp).sum(axis=(4,5,6))

def main():
    B, C, H, W = (1,3,8,8)
    O, KH, KW = (4,3,3)
    pad = 1

    x = Tensor.ones(B, C, H , W)
    K = Tensor.ones(O, C, KH, KW)

    y_builtin = x.conv2d(K, stride=1, padding=pad)
    print(f"{y_builtin.uop.sink()=}")

    y_explicit = conv2d_explicit_broadcast(x, K, pad=pad)
    print(f"{y_explicit.uop.sink()=}")

    yb = y_builtin.realize().numpy()
    ye = y_explicit.realize().numpy() 
    print(f"max diff:{np.abs(yb - ye).max()}")

if __name__ == '__main__':
    main()



