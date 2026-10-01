"""Stabilointivaiheen (vaihekorrelaatio, taysi resoluutio) nopeusvertailu omalla koneella.

  python tools/bench_stabilointi.py <video> [ruutuja=200]

Mittaa (ms/ruutu):
  1. nykyinen CPU-polku (ikkuna + DFT + vaihe + kaanteis-DFT + huippu)
  2. vaiheet erikseen + cv2.setNumThreads(1/2/4/8)
  3. useita ruutuja rinnan (1/2/3/4 saiketta) -> lapimeno r/s (stabilointi on ruudusta riippumaton: kukin ruutu verrataan moodikuvaan)
  4. OpenCL (cv2.UMat; esim. Intel UHD Graphics) -> DFT/vaihe/kaanteis-DFT GPU:lla, jos OpenCV:ssa on OpenCL; tulos verrataan CPU:hun
Tulostaa myos OpenCL-laitteen tiedot.
"""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
import stone_tracker

if len(sys.argv) > 1:
    video = sys.argv[1]
else:
    try:                                    # ei argumenttia -> tiedostovalitsin
        import tkinter, tkinter.filedialog
        tkinter.Tk().withdraw()
        video = tkinter.filedialog.askopenfilename(title="Valitse video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
    except Exception:
        video = ""
    if not video:
        sys.exit("Kayto: python tools/bench_stabilointi.py <video> [ruutuja=200]")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
cap = cv2.VideoCapture(video)
grays = []
while len(grays) < N + 1:
    ok, f = cap.read()
    if not ok:
        break
    grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
ref = grays[0]; grays = grays[1:]
h, w = ref.shape
print(f"video {w}x{h}, {len(grays)} ruutua; optimaalinen DFT-koko: {cv2.getOptimalDFTSize(w)}x{cv2.getOptimalDFTSize(h)}")
print("OpenCV", cv2.__version__, "| saikeita", cv2.getNumThreads(), "| OpenCL:", cv2.ocl.haveOpenCL())
if cv2.ocl.haveOpenCL():
    d = cv2.ocl.Device.getDefault()
    print("  OpenCL-laite:", d.name(), "| tyyppi", "GPU" if d.type() == cv2.ocl.Device_TYPE_GPU else d.type(), "| laskentayksikoita", d.maxComputeUnits(), "| globaali muisti MB", d.globalMemSize() // 2**20)

hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
ref_ccs = cv2.dft(stone_tracker.phase_window(np.ascontiguousarray(ref), hann))


def cpu(g):
    f = cv2.dft(stone_tracker.phase_window(np.ascontiguousarray(g), hann))
    p = stone_tracker.phase_mulnorm(ref_ccs, f)
    c = cv2.dft(p, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)
    return stone_tracker.phase_peak(c)


def bench(fn, items):
    fn(items[0])
    t = time.perf_counter()
    out = [fn(g) for g in items]
    return (time.perf_counter() - t) / len(items) * 1000, out


ms, base = bench(cpu, grays)
print(f"\n1. CPU-polku kokonaisuutena: {ms:.2f} ms/ruutu")

g = grays[0]
wnd = stone_tracker.phase_window(np.ascontiguousarray(g), hann)
f = cv2.dft(wnd)
p = stone_tracker.phase_mulnorm(ref_ccs, f)
c = cv2.dft(p, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)
def t_(fn, n=30):
    fn(); t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000
print(f"2. vaiheet: ikkuna {t_(lambda: stone_tracker.phase_window(g, hann)):.2f} | DFT {t_(lambda: cv2.dft(wnd)):.2f} | vaihe {t_(lambda: stone_tracker.phase_mulnorm(ref_ccs, f)):.2f} | kaanteis-DFT {t_(lambda: cv2.dft(p, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)):.2f} | huippu {t_(lambda: stone_tracker.phase_peak(c)):.2f}")
n0 = cv2.getNumThreads()
for nt in (1, 2, 4, 8):
    cv2.setNumThreads(nt)
    print(f"   cv2.setNumThreads({nt}): DFT {t_(lambda: cv2.dft(wnd)):.2f} ms, kaanteis-DFT {t_(lambda: cv2.dft(p, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)):.2f} ms")
cv2.setNumThreads(n0)

print("3. useita ruutuja rinnan (lapimeno):")
for nw in (1, 2, 3, 4):
    it = iter(grays); lock = threading.Lock()
    def worker():
        while True:
            with lock:
                g = next(it, None)
            if g is None: return
            cpu(g)
    ths = [threading.Thread(target=worker) for _ in range(nw)]
    t = time.perf_counter()
    for th in ths: th.start()
    for th in ths: th.join()
    dt = time.perf_counter() - t
    print(f"   {nw} saie(tta): {len(grays) / dt:.1f} r/s ({dt / len(grays) * 1000:.2f} ms/ruutu lapimenona)")

print("4. OpenCL / UMat (GPU):")
if not cv2.ocl.haveOpenCL():
    print("   OpenCL ei kaytettavissa tassa OpenCV-asennuksessa.")
else:
    cv2.ocl.setUseOpenCL(True)
    u_hann = cv2.UMat(hann)
    # referenssi: taysi kompleksi-DFT, normalisoitu
    def spec(img_u):
        return cv2.dft(cv2.multiply(cv2.UMat(img_u).convertTo(cv2.CV_32F) if False else img_u, u_hann), flags=cv2.DFT_COMPLEX_OUTPUT)
    try:
        ref_u = cv2.UMat(np.ascontiguousarray(ref).astype(np.float32))
        R = cv2.dft(cv2.multiply(ref_u, u_hann), flags=cv2.DFT_COMPLEX_OUTPUT)
        def gpu(gimg):
            gu = cv2.UMat(gimg.astype(np.float32))
            F = cv2.dft(cv2.multiply(gu, u_hann), flags=cv2.DFT_COMPLEX_OUTPUT)
            P = cv2.mulSpectrums(R, F, 0, conjB=True)
            parts = cv2.split(P)
            mag = cv2.magnitude(parts[0], parts[1])
            mag = cv2.max(mag, 1e-20)
            P = cv2.merge([cv2.divide(parts[0], mag), cv2.divide(parts[1], mag)])
            C = cv2.dft(P, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            Cn = C.get()
            return stone_tracker.phase_peak(Cn)
        ms_g, out_g = bench(gpu, grays)
        d = np.abs(np.array(out_g) - np.array(base))
        print(f"   GPU-polku: {ms_g:.2f} ms/ruutu | ero CPU:hun: max {d.max():.4f} px, ka {d.mean():.4f} px")
        print("   (huom: jos GPU-polku ei ole selvasti nopeampi kuin kohta 1, sita ei kannata kayttaa)")
    except Exception as e:
        print("   GPU-polku epaonnistui:", repr(e))


print("5. warpAffine + remap (vaihe B) CPU vs OpenCL:")
try:
    fr = cv2.cvtColor(grays[0], cv2.COLOR_GRAY2BGR)
    M = np.array([[1.0, 0.0, -1.3], [0.0, 1.0, 0.7]], np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map1 = xx + 2.0 * np.sin(yy / 50.0).astype(np.float32); map2 = yy + 2.0 * np.cos(xx / 60.0).astype(np.float32)
    def cpu_b(_):
        return cv2.remap(cv2.warpAffine(fr, M, (w, h)), map1, map2, interpolation=cv2.INTER_LINEAR)
    ms_b, ref_b = bench(cpu_b, [0] * 40)
    print(f"   CPU: {ms_b:.2f} ms/ruutu")
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        um1, um2 = cv2.UMat(map1), cv2.UMat(map2)
        def gpu_b(_):
            return cv2.remap(cv2.warpAffine(cv2.UMat(fr), M, (w, h)), um1, um2, interpolation=cv2.INTER_LINEAR).get()
        ms_gb, out_gb = bench(gpu_b, [0] * 40)
        print(f"   GPU (UMat, sis. siirrot): {ms_gb:.2f} ms/ruutu | ero CPU:hun max {np.abs(out_gb[0].astype(int) - ref_b[0].astype(int)).max()}")
except Exception as e:
    print("   epaonnistui:", repr(e))
