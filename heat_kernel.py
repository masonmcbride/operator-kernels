"""
uv run --python 3.14 \
  --with numpy --with scipy --with matplotlib --with tinygrad \
  heat_kernel.py
"""
import sys, threading
from itertools import repeat, accumulate, count
from queue import Queue
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tinygrad import Tensor, nn, dtypes

def make_animation(M=120, N=120, steps=600, interval=30, cmap="turbo", alpha=0.20):
    X, Y = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')

    # Initial State
    u0 = (
        np.exp(-((X-M*0.30)**2 + (Y-N*0.35)**2)/(2*4.5**2)) * 80.0
      + np.exp(-((X-M*0.70)**2 + (Y-N*0.65)**2)/(2*6.0**2)) * 60.0
      + 90.0 * ((np.abs(X-M//2) < 2) & (np.abs(Y-N//2) < 2)).astype(float)
    ).astype(float)

    # Convolution Kernel 
    K = np.array([[0, alpha, 0],
                  [alpha, 1 - 4*alpha, alpha],
                  [0, alpha, 0]], dtype=float)

    # Scipy Solution
    U = list(accumulate(repeat(None, steps),
                        lambda a, _ : convolve(a, K, mode='constant',cval=0.0),
                        initial=u0))

    # Tinygrad Solution
    conv = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
    conv.weight.assign(Tensor(K, dtype=dtypes.float).reshape(1,1, *K.shape))
    R = lambda t: (t.realize(), t)[1]  # realize and return
    Up = list(accumulate(
        repeat(None, steps),
        lambda a, _: R(conv(a)),
        initial=Tensor(u0, dtype=dtypes.float).reshape(1, 1, *u0.shape)
    ))
    #sink = conv(Tensor(u0, dtype=dtypes.float).reshape(1, 1, *u0.shape)).uop.sink()
    #print("printing sink")
    #print(sink)

    t_queue = Queue()
    def stdin_reader():
        print("\nType a time t (>= 0) and press Enter.\n"
                "Right panel shows analytic u(t) = G_t * u0.\n"
                "(Non-numeric/negative/inf/NaN inputs are ignored.)\n",
            file=sys.stderr, flush=True)
        for line in sys.stdin:
            try:
                t = float(line)
            except ValueError:
                continue
            if t >= 0 and np.isfinite(t):
                t_queue.put(t)
    threading.Thread(target=stdin_reader, daemon=True).start()

    # figure
    vmin, vmax = float(u0.min()), float(u0.max())
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    for ax in (axA, axB, axC): ax.set_axis_off()

    imA = axA.imshow(U[0],  cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    imB = axB.imshow(u0,   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    imC = axC.imshow(Up[0].numpy()[0,0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

    axA.set_title("A: Precomputed (SciPy, constant-0)")
    axB.set_title("B: Analytic u(t) = G_t * u0 (t = 0, σ = 0, constant-0)")
    axC.set_title("C: Precomputed (Tinygrad, constant-0)")

    cbar = fig.colorbar(imA, ax=[axA, axB, axC], fraction=0.046, pad=0.04)
    cbar.set_label("Temperature")

    def animate(frame_idx,
                steps=steps, U=U, Uprime=Up,
                imA=imA, axA=axA,
                imB=imB, axB=axB,
                imC=imC, axC=axC,
                u0=u0, alpha=alpha, t_queue=t_queue):

        k = frame_idx if frame_idx <= steps else steps
        # Image A -- Scipy convolution
        imA.set_array(U[k])
        axA.set_title(f"Live diffusion via Scipy (step {k}/{steps})" if k < steps
                      else f"Live diffusion via Scipy (completed {steps} steps)")

        # Image B -- Heat Kernel solution 
        while not t_queue.empty():
            t = t_queue.get_nowait()
            sigma = float(np.sqrt(2.0 * alpha * t))
            imB.set_array(gaussian_filter(u0, sigma=sigma, mode='constant'))
            axB.set_title(f"Analytic u(t) = G_t * u0 (t = {t:g}, σ = {sigma:.3f})")

        # Image C -- Tinygrad convolution 
        imC.set_array(Uprime[k].numpy()[0,0])
        axC.set_title(f"Live diffusion via Tinygrad (step {k}/{steps})" if k < steps 
                      else f"Live diffusion via Tinygrad (completed {steps} steps)")


        return [imA, imB, imC]

    anim = FuncAnimation(fig, animate, frames=count(),
                         interval=interval, blit=False,
                         cache_frame_data=False, repeat=False)
    return fig, anim

def main():
    fig, _ = make_animation()
    plt.show()

if __name__ == "__main__":
    main()