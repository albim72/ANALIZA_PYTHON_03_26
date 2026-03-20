import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =========================
# PARAMETRY WIZUALIZACJI
# =========================
width, height = 900, 900
max_iter = 220
frames = 180
fps = 30

x_min, x_max = -1.6, 1.6
y_min, y_max = -1.6, 1.6

# =========================
# SIATKA LICZB ZESPOLONYCH
# =========================
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)
Z0 = X + 1j * Y

# =========================
# FIGURA
# =========================
fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
ax.axis("off")

image = ax.imshow(
    np.zeros((height, width)),
    cmap="magma",
    extent=(x_min, x_max, y_min, y_max),
    origin="lower",
    animated=True
)

title = ax.set_title("", fontsize=11)

# =========================
# FUNKCJA LICZĄCA FRAKTAL
# =========================
def compute_julia(c: complex) -> np.ndarray:
    Z = Z0.copy()
    output = np.zeros(Z.shape, dtype=np.float32)

    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + c
        still_inside = np.abs(Z) < 2.0

        output[mask & (~still_inside)] = i
        mask = still_inside

        if not mask.any():
            break

    output[output == 0] = max_iter
    return output

# =========================
# AKTUALIZACJA KLATKI
# =========================
def update(frame: int):
    angle = 2 * np.pi * frame / frames
    radius = 0.7885

    c = radius * np.exp(1j * angle)

    data = compute_julia(c)
    image.set_array(data)
    title.set_text(
        f"Animowany fraktal Julii\nframe={frame+1}/{frames}, c={c.real:.4f} + {c.imag:.4f}i"
    )
    return image, title

# =========================
# ANIMACJA
# =========================
anim = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=1000 / fps,
    blit=True
)

# =========================
# ZAPIS DO MP4
# =========================
writer = FFMpegWriter(fps=fps, metadata={"artist": "OpenAI"}, bitrate=3000)

anim.save("fraktal_julii.mp4", writer=writer)

print("Zapisano plik: fraktal_julii.mp4")

plt.show()
