"""Poster figures: the digital teacher and the thermodynamic student.

Visualizes the IMITATION with REAL data: every neuron is shaded (oranges) by its
activation on one MNIST test digit.  The teacher shades come from the trained
teacher_thermo.pt; the student shades are the trained computer's actual mean
node values <x_i(t_f)>, from student_proportional_kh100_ko100_tf0.2.pt -- the
locked config (thermo teacher + proportional k=100).

The teacher is layered; the student's hidden neurons are SCATTERED (the
thermodynamic computer has no layer structure -- its couplings are all-to-all
and bidirectional), with the 10 output spins pulled out to the right.  Matching
shades between the panels are the point: the student reproduces the teacher's
activations.

Usage:
    python plot_architecture.py            # -> Graphs/poster/fig_architecture.{pdf,png}
    python plot_architecture.py split      # also writes the two panels separately
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
OUTDIR = os.path.join(_HERE, "..", "..", "Graphs", "poster")

STUDENT_PT = os.path.join(_HERE, "runs",
                          "student_proportional_kh100_ko100_tf0.2.pt")
DIGIT = 7          # which test digit to display
TF = 0.2
M_SAMPLES = 400    # reset samples for the student's mean activations

# neurons drawn: teacher layer-1 [0:6], layer-2 [32:38], outputs [0:5]
H1_IDX = list(range(0, 6))
H2_IDX = list(range(32, 38))
# four output spins, one of which is the PREDICTED class (9 for DIGIT=7) so the
# winner still stands out dark against the other three.  If DIGIT changes, put
# the new predicted class (printed at run time) in this list.
OUT_IDX = [0, 1, 2, 9]
N_HID_SHOW = len(H1_IDX) + len(H2_IDX)

R = 0.135
SPAN = 1.9
PITCH = SPAN / (len(H1_IDX) - 1)   # hidden neuron spacing; outputs reuse it
C_EDGE = "#d3d8dc"
C_FROZEN = "#9aa3ab"
CMAP = plt.cm.Oranges


def shade(v):
    """activation in [0,1] -> orange (kept off the washed-out low end)."""
    return CMAP(0.22 + 0.76 * float(np.clip(v, 0.0, 1.0)))


def _norm(a):
    a = np.asarray(a, dtype=float)
    return (a - a.min()) / max(float(np.ptp(a)), 1e-9)


def get_activations():
    """Real (teacher, student) activations for the shown neurons, each in [0,1]."""
    import torch
    from teacher_net import load_mnist, load_teacher
    from thermo_mnist import ThermoMNIST, HIDDEN

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, Xte, _ = load_mnist(dev)
    px = Xte[DIGIT:DIGIT + 1]

    teacher, tacc = load_teacher(activation="thermo", device=dev)
    with torch.no_grad():
        hid, out = teacher.activations(px)
    t_pick = np.concatenate([hid[0, H1_IDX].cpu().numpy(),
                             hid[0, H2_IDX].cpu().numpy(),
                             out[0, OUT_IDX].cpu().numpy()])

    d = torch.load(STUDENT_PT, map_location=dev)
    student = ThermoMNIST().to(dev)
    student.load_state_dict(d["state_dict"] if "state_dict" in d else d)
    with torch.no_grad():
        x = student._rollout(px, M=M_SAMPLES, tf=TF, seed=0)   # (1, M, 74)
    nodes = x.mean(dim=1)[0].cpu().numpy()                     # (74,)
    s_pick = np.concatenate([nodes[H1_IDX], nodes[H2_IDX],
                             nodes[[HIDDEN + i for i in OUT_IDX]]])
    print(f"teacher {tacc:.4f} | student {os.path.basename(STUDENT_PT)}")
    print(f"predicted class: teacher {OUT_IDX[int(np.argmax(t_pick[N_HID_SHOW:]))]}, "
          f"student {OUT_IDX[int(np.argmax(s_pick[N_HID_SHOW:]))]}")
    # hidden and outputs are different quantities (activations vs class scores),
    # so shade each group on its own scale -- otherwise the softmax outputs
    # (nine ~0, one ~1) compress against the wider hidden range and the winning
    # class does not stand out.
    def _split_norm(a):
        return np.concatenate([_norm(a[:N_HID_SHOW]), _norm(a[N_HID_SHOW:])])
    return _split_norm(t_pick), _split_norm(s_pick)


def _spread(n, span, center=0.0):
    if n == 1:
        return np.array([center])
    return np.linspace(center - span / 2, center + span / 2, n)


def _spread_pitch(n, pitch, center=0.0):
    """n positions at a FIXED neuron-to-neuron spacing, centred."""
    if n == 1:
        return np.array([center])
    half = (n - 1) * pitch / 2.0
    return np.linspace(center - half, center + half, n)


def _pixel_patch(ax, cx, cy, half=0.44, seed=3):
    """Stylised pixel grid standing in for the 784 input pixels."""
    rng = np.random.default_rng(seed)
    g = rng.random((7, 7))
    ax.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half,
                           facecolor="#eef1f3", edgecolor="#2b3036", lw=1.4,
                           zorder=2))
    s = 2 * half / 7
    for i in range(7):
        for j in range(7):
            ax.add_patch(Rectangle((cx - half + j * s, cy + half - (i + 1) * s),
                                   s, s, facecolor=str(1.0 - 0.75 * g[i, j]),
                                   edgecolor="none", zorder=3))


def draw_teacher(ax, v):
    """Layered digital net; neurons shaded by their real activation."""
    xs = [0.0, 1.15, 2.25, 3.35]
    y1 = _spread(len(H1_IDX), SPAN)
    y2 = _spread(len(H2_IDX), SPAN)
    yo = _spread_pitch(len(OUT_IDX), PITCH)

    _pixel_patch(ax, xs[0], 0.0)                     # same visual as the student
    for b in y1:
        ax.plot([xs[0] + 0.45, xs[1] - R], [0.0, b], color=C_EDGE, lw=0.5,
                zorder=1)
    for (xa, ya), (xb, yb) in (((xs[1], y1), (xs[2], y2)),
                               ((xs[2], y2), (xs[3], yo))):
        for a in ya:
            for b in yb:
                ax.plot([xa + R, xb - R], [a, b], color=C_EDGE, lw=0.45, zorder=1)
    n1 = len(H1_IDX); n2 = len(H2_IDX)
    for x, ys, vals in ((xs[1], y1, v[:n1]),
                        (xs[2], y2, v[n1:n1 + n2]),
                        (xs[3], yo, v[n1 + n2:])):
        for y, val in zip(ys, vals):
            ax.add_patch(Circle((x, y), R, facecolor=shade(val),
                                edgecolor="#1f2429", lw=1.4, zorder=3))

    for t, x in (("32 hidden", xs[1]), ("32 hidden", xs[2]),
                 ("10 outputs", xs[3])):
        ax.text(x, SPAN / 2 + 0.24, t, ha="center", va="bottom",
                fontsize=11.5, color="#3a4046")
    ax.text(xs[0], -0.66, "784\npixels", ha="center", va="top",
            fontsize=11.5, color="#3a4046")
    ax.set_xlim(-0.75, 4.10); ax.set_ylim(-1.20, 1.44)


def _scatter_positions(n, seed=5, rx=0.95, ry=1.00, min_d=0.42, cx=0.0):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = np.array([rng.uniform(-rx, rx), rng.uniform(-ry, ry)])
        if p[0] ** 2 / rx ** 2 + p[1] ** 2 / ry ** 2 > 1.0:
            continue
        if all(np.hypot(*(p - q)) > min_d for q in pts):
            pts.append(p)
    pts = np.array(pts); pts[:, 0] += cx
    return pts


def draw_student(ax, v):
    """Scattered all-to-all hidden spins, outputs pulled out to the right."""
    hid = _scatter_positions(N_HID_SHOW, cx=1.70)
    x_out = 3.60
    yo = _spread_pitch(len(OUT_IDX), PITCH)
    out = np.stack([np.full(len(OUT_IDX), x_out), yo], axis=1)

    _pixel_patch(ax, -0.1, 0.0)
    for p in np.vstack([hid, out]):                 # frozen pixels -> every spin
        ax.plot([0.35, p[0]], [0.0, p[1]], color="#e8ecef", lw=0.45, zorder=1)
    allp = np.vstack([hid, out])
    for i in range(len(allp)):                      # all-to-all, bidirectional
        for j in range(i + 1, len(allp)):
            ax.plot([allp[i, 0], allp[j, 0]], [allp[i, 1], allp[j, 1]],
                    color=C_EDGE, lw=0.4, zorder=1)

    for p, val in zip(hid, v[:N_HID_SHOW]):
        ax.add_patch(Circle(tuple(p), R, facecolor=shade(val),
                            edgecolor="#1f2429", lw=1.4, zorder=3))
    for p, val in zip(out, v[N_HID_SHOW:]):
        ax.add_patch(Circle(tuple(p), R, facecolor=shade(val),
                            edgecolor="#1f2429", lw=1.4, zorder=3))

    ax.text(-0.1, -0.66, "784 frozen\npixel spins", ha="center", va="top",
            fontsize=11.5, color="#3a4046")
    ax.text(1.70, SPAN / 2 + 0.24, "64 hidden", ha="center", va="bottom",
            fontsize=11.5, color="#3a4046")
    ax.text(x_out, SPAN / 2 + 0.24, "10 outputs", ha="center", va="bottom",
            fontsize=11.5, color="#3a4046")
    ax.set_xlim(-0.75, 4.10); ax.set_ylim(-1.20, 1.44)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    v_t, v_s = get_activations()

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    draw_teacher(axes[0], v_t)
    draw_student(axes[1], v_s)
    for ax in axes:
        ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(OUTDIR, f"fig_architecture.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"saved {p}")
    plt.close(fig)

    if len(sys.argv) > 1 and sys.argv[1] == "split":
        for name, fn, arg in (("teacher", draw_teacher, v_t),
                              ("student", draw_student, v_s)):
            f, a = plt.subplots(figsize=(6.5, 3.56))
            fn(a, arg); a.set_aspect("equal"); a.axis("off")
            f.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            for ext in ("pdf", "png"):
                p = os.path.join(OUTDIR, f"fig_arch_{name}.{ext}")
                f.savefig(p, dpi=300, facecolor="white")   # no tight-crop
                print(f"saved {p}")
            plt.close(f)


if __name__ == "__main__":
    main()
