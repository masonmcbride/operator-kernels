"""
uv run --python 3.14 \
  --with numpy --with scipy --with matplotlib --with tinygrad --with pillow \
  heat_kernel_demo.py
"""
import sys, threading
from itertools import repeat, accumulate, count
from queue import Queue
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tinygrad import Tensor, nn, dtypes

# ---------- core animation + recording ----------

def make_animation(M=120, N=120, steps=600, interval=30, cmap="turbo", alpha=0.20):
    X, Y = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')

    # Initial State
    u0 = (
        np.exp(-((X-M*0.30)**2 + (Y-N*0.35)**2)/(2*4.5**2)) * 80.0
      + np.exp(-((X-M*0.70)**2 + (Y-N*0.65)**2)/(2*6.0**2)) * 60.0
      + 90.0 * ((np.abs(X-M//2) < 2) & (np.abs(Y-N//2) < 2)).astype(float)
    ).astype(float)

    # 3x3 diffusion kernel
    K = np.array([[0, alpha, 0],
                  [alpha, 1 - 4*alpha, alpha],
                  [0, alpha, 0]], dtype=float)

    # Precompute discrete diffusion (SciPy)
    U = list(accumulate(repeat(None, steps),
                        lambda a, _ : convolve(a, K, mode='constant', cval=0.0),
                        initial=u0))

    # Precompute discrete diffusion (tinygrad)
    conv = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
    conv.weight.assign(Tensor(K, dtype=dtypes.float).reshape(1,1, *K.shape))
    realize = lambda t: (t.realize(), t)[1]
    Up = list(accumulate(
        repeat(None, steps),
        lambda a, _: realize(conv(a)),
        initial=Tensor(u0, dtype=dtypes.float).reshape(1, 1, *u0.shape)
    ))

    # --- interactive time input (for analytic panel) ---
    t_queue = Queue()
    def stdin_reader():
        print("\nType a time t (>= 0) and press Enter.\n"
              "Middle panel shows analytic u(t) = G_t * u0.\n"
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

    # Figure
    vmin, vmax = float(u0.min()), float(u0.max())
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    for ax in (axA, axB, axC): ax.set_axis_off()

    imA = axA.imshow(U[0],             cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    imB = axB.imshow(u0,               cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    imC = axC.imshow(Up[0].numpy()[0,0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

    axA.set_title("A: Precomputed (SciPy, constant-0)")
    axB.set_title("B: Analytic u(t) = G_t * u0 (t = 0, σ = 0)")
    axC.set_title("C: Precomputed (Tinygrad, constant-0)")

    cbar = fig.colorbar(imA, ax=[axA, axB, axC], fraction=0.046, pad=0.04)
    cbar.set_label("Temperature")

    # --- RECORDING SCHEDULE ---
    # analytic_t_schedule[k] = the t that should be used on frame k (or None)
    analytic_t_schedule = [None]*(steps+1)

    def animate(frame_idx,
                steps=steps, U=U, Uprime=Up,
                imA=imA, axA=axA,
                imB=imB, axB=axB,
                imC=imC, axC=axC,
                u0=u0, alpha=alpha, t_queue=t_queue,
                analytic_t_schedule=analytic_t_schedule):

        k = min(frame_idx, steps)

        # A -- SciPy convolution
        imA.set_array(U[k])
        axA.set_title(f"Live diffusion via SciPy (step {k}/{steps})" if k < steps
                      else f"Live diffusion via SciPy (completed {steps} steps)")

        # B -- Analytic heat kernel:
        # Drain any times typed since the last frame. Keep the *latest* one for this frame.
        latest_t = None
        while not t_queue.empty():
            latest_t = t_queue.get_nowait()
        if latest_t is not None:
            analytic_t_schedule[k] = latest_t

        # Compute analytic panel using the most recent t applicable up to this frame
        # (carry forward the last known t if none typed exactly at this frame)
        t_use = analytic_t_schedule[k]
        if t_use is None and k > 0:
            t_use = analytic_t_schedule[k-1]
            analytic_t_schedule[k] = t_use

        if t_use is None:
            # still initial state
            imB.set_array(u0)
            axB.set_title("Analytic u(t) = G_t * u0 (t = 0)")
        else:
            sigma = float(np.sqrt(2.0 * alpha * t_use))
            imB.set_array(gaussian_filter(u0, sigma=sigma, mode='constant'))
            axB.set_title(f"Analytic u(t) = G_t * u0 (t = {t_use:g}, σ = {sigma:.3f})")

        # C -- Tinygrad convolution
        imC.set_array(Uprime[k].numpy()[0,0])
        axC.set_title(f"Live diffusion via Tinygrad (step {k}/{steps})" if k < steps
                      else f"Live diffusion via Tinygrad (completed {steps} steps)")

        return [imA, imB, imC]

    anim = FuncAnimation(fig, animate, frames=count(),
                         interval=interval, blit=False,
                         cache_frame_data=False, repeat=False)

    # We return all data needed to REPLAY (for saving)
    replay_state = {
        "u0": u0,
        "U": U,
        "Up": Up,
        "alpha": alpha,
        "steps": steps,
        "interval": interval,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "analytic_t_schedule": analytic_t_schedule,
    }
    return fig, anim, replay_state

# ---------- replay & save to GIF ----------

def save_replay_gif(state, outfile="heat_session.gif", hold_frames=50):
    """
    Rebuild the animation from the recorded state and save a GIF.
    We fake a 'pause at the end' by extending the frame range so the
    last frame is repeated hold_frames times *during* encoding.
    This avoids the second pass where we re-quantize everything.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from scipy.ndimage import gaussian_filter
    import numpy as np

    u0      = state["u0"]
    U       = state["U"]
    Up      = state["Up"]
    alpha   = state["alpha"]
    steps   = state["steps"]
    interval= state["interval"]
    cmap    = state["cmap"]
    vmin    = state["vmin"]
    vmax    = state["vmax"]
    sched   = state["analytic_t_schedule"]

    # total frames in the final GIF = original steps+1 plus the extra hold
    total_frames = (steps + 1) + hold_frames

    # set up a fresh figure for replay/record
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(14.4, 4.8), constrained_layout=True
    )
    for ax in (axA, axB, axC):
        ax.set_axis_off()

    imA = axA.imshow(
        U[0],
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest'
    )
    imB = axB.imshow(
        u0,
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest'
    )
    imC = axC.imshow(
        Up[0].numpy()[0,0],
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest'
    )

    cbar = fig.colorbar(imA, ax=[axA, axB, axC], fraction=0.046, pad=0.04)
    cbar.set_label("Temperature")

    def animate_replay(frame_number):
        """
        frame_number runs from 0 .. total_frames-1.

        For frames past 'steps', we just hold the last physical state
        (k = steps). This creates that freeze/pause at the end.
        """
        # clamp k so after steps we just keep using steps
        k = frame_number
        if k > steps:
            k = steps

        # Panel A (SciPy diffusion)
        imA.set_array(U[k])
        if k < steps:
            axA.set_title(f"Live diffusion via SciPy (step {k}/{steps})")
        else:
            axA.set_title(f"Live diffusion via SciPy (completed {steps} steps)")

        # Panel B (analytic heat kernel)
        # pick last known t up to/including k
        t_use = sched[k]
        if t_use is None and k > 0:
            t_use = sched[k-1]

        if t_use is None:
            imB.set_array(u0)
            axB.set_title("Analytic u(t) = G_t * u0 (t = 0, σ = 0)")
        else:
            sigma = float(np.sqrt(2.0 * alpha * t_use))
            analytic = gaussian_filter(u0, sigma=sigma, mode='constant')
            imB.set_array(analytic)
            axB.set_title(
                f"Analytic u(t) = G_t * u0 (t = {t_use:g}, σ = {sigma:.3f})"
            )

        # Panel C (Tinygrad diffusion)
        imC.set_array(Up[k].numpy()[0,0])
        if k < steps:
            axC.set_title(f"Live diffusion via Tinygrad (step {k}/{steps})")
        else:
            axC.set_title(f"Live diffusion via Tinygrad (completed {steps} steps)")

        return [imA, imB, imC]

    # build the replay animation WITH the held tail
    anim2 = FuncAnimation(
        fig,
        animate_replay,
        frames=range(total_frames),
        interval=interval,
        blit=False,
        repeat=False,
    )

    # PillowWriter will quantize to GIF once, using its own approach.
    # fps ~ 1000/interval(ms). interval is already in ms.
    fps = max(1, int(1000/interval))
    writer = PillowWriter(fps=fps)

    anim2.save(outfile, writer=writer, dpi=100)
    plt.close(fig)

    import sys
    print(f"Saved GIF to {outfile} with {hold_frames} frame hold at end", file=sys.stderr)


# ---------- main ----------

def main():
    fig, anim, state = make_animation()
    # Show interactively; user types times; schedule fills during run.
    plt.show()
    # After window closes, replay with the recorded schedule and save.
    save_replay_gif(state, outfile="heat_session.gif")

if __name__ == "__main__":
    main()
