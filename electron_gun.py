# electron_gun.py
# uv run --python 3.13 --with numpy --with pyvista python electron_gun.py
import time
import numpy as np
import pyvista as pv

rng = np.random.default_rng(0)

# ---------- scene scale (meters) ----------
GUN_POS    = np.array([0.0, 0.0, 0.0], dtype=np.float64)
PLATE_Z    = 200e-6          # 200 microns downrange
PLATE_SIZE = 120e-6          # square plate width

# ---------- particle parameters ----------
SPAWN_N = 10
DT = 2e-16
SUBSTEPS = 600

V_MEAN_Z   = 2.0e7
V_SIGMA_XY = 2.0e5
V_SIGMA_Z  = 1.0e5

POINT_SIZE = 12.0  # visual "radius" in pixels (fast)

def init_batch():
    # shape: (10, 6) = [x, y, z, vx, vy, vz]
    return np.c_[
        np.repeat(GUN_POS[None], SPAWN_N, axis=0),
        rng.normal([0.0, 0.0, V_MEAN_Z],
                   [V_SIGMA_XY, V_SIGMA_XY, V_SIGMA_Z],
                   size=(SPAWN_N, 3)),
    ]

batch_buffer: list[np.ndarray] = []

# ---------- visualization ----------
pl = pv.Plotter()
pl.enable_trackball_style()
pl.add_axes()
pl.add_text("Space = spawn 10 particles | q / Esc = quit | Mouse = camera", font_size=12)

pl.add_mesh(pv.Sphere(radius=5e-6, center=tuple(GUN_POS)), opacity=0.35)
pl.add_mesh(
    pv.Plane(center=(0.0, 0.0, PLATE_Z), direction=(0, 0, 1),
             i_size=PLATE_SIZE, j_size=PLATE_SIZE),
    opacity=0.15,
)

# One point cloud actor for ALL particles (no per-frame actor churn)
cloud = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
pl.add_mesh(cloud, render_points_as_spheres=True, point_size=POINT_SIZE)

pending_spawn = False
running = True

def on_space():
    global pending_spawn
    pending_spawn = True

def on_quit():
    global running
    running = False
    pl.close()

pl.add_key_event("space", on_space)
pl.add_key_event("q", on_quit)
pl.add_key_event("Escape", on_quit)

pl.show(auto_close=False, interactive_update=True)
pl.camera.clipping_range = (1e-9, 1.0)

# ---------- main loop ----------
while running:
    try:
        if pl.iren is not None:
            pl.iren.process_events()

        if pending_spawn:
            pending_spawn = False
            batch_buffer.append(init_batch())
            print([b.shape for b in batch_buffer])
            pl.reset_camera()

        # update all batches (straight-line motion)
        step_dt = DT * SUBSTEPS
        alive = []
        for batch in batch_buffer:
            batch[:, :3] += batch[:, 3:] * step_dt
            if np.any(batch[:, 2] <= PLATE_Z):
                alive.append(batch)
        batch_buffer = alive

        # render: concatenate all positions into one (N,3) array
        if batch_buffer:
            pts = np.concatenate([b[:, :3] for b in batch_buffer], axis=0)
            cloud.points = pts
        else:
            cloud.points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        pl.render()
        pl.update()
        time.sleep(0.002)

    except RuntimeError:
        break
    except KeyboardInterrupt:
        break
