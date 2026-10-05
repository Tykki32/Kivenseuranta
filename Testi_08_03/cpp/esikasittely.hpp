// ============================================================
// esikasittely.hpp - ruudun esikasittely (liukuhihnan vaiheet A ja B, Python: esikasittely.py):
// varjosietoinen taustanvaimennus + valotasapaino ja vaihekorrelaation elementtikohtaiset vaiheet.
// ============================================================
#pragma once
#include "maskit.hpp"

// Rivijako omilla std::thread-saikeilla: cv::parallel_for_ jumittui, kun liukuhihnan saie kutsui sita samaan aikaan
// kuin SEURANTA piti OpenCV:n yksisaikeisena (ScopedSingleThreadedOpenCV).
struct RowRange { int start, end; };

template <class F>
static void parallelRowsStd(int rows, F fn)
{
    unsigned hw = std::thread::hardware_concurrency();
    int nt = (int)std::max(1u, std::min(4u, hw == 0 ? 2u : hw));
    if (nt <= 1 || rows < 64) { fn(RowRange{ 0, rows }); return; }
    int chunk = (rows + nt - 1) / nt;
    std::vector<std::thread> th;
    for (int i = 1; i < nt; ++i) {
        int a = i * chunk, b = std::min(rows, a + chunk);
        if (a >= b) break;
        th.emplace_back([&fn, a, b]() { fn(RowRange{ a, b }); });
    }
    fn(RowRange{ 0, std::min(rows, chunk) });
    for (auto& t : th) t.join();
}

// ============================================================
// VARJOSIETOINEN TAUSTANVAIMENNUS + VALOTASAPAINO yhdella lapikaynnilla. Pikseli on taustaa (valkaistaan), jos
//   harmaasavyero moodikuvaan < diff_threshold, TAI kirkkaus (V) on pudonnut moodikuvasta valilla
//   (v_drop_min, v_drop_max) (varjo), TAI pikseli on jaata (S < ice_s_max ja V > ice_v_min),
// ja variportti (colorGateOk) sallii sen. Valotasapaino: kanavakohtainen LUT (float, clip, katkaisu).
// OpenCV:n kokonaislukukaavat (BGR2GRAY, HSV:n V ja S).
// ============================================================
static py::array_t<uint8_t> suppress_shadow_background(
    ArrU8 frame, ArrU8 reference, ArrD gains, ArrD biases,
    double diff_threshold, double v_drop_min, double v_drop_max, int ice_s_max, int ice_v_min)
{
    auto fb = frame.request();
    auto rb = reference.request();
    if (fb.ndim != 3 || fb.shape[2] != 3 || rb.ndim != 3 || rb.shape[2] != 3 || fb.shape[0] != rb.shape[0] || fb.shape[1] != rb.shape[1])
        throw std::runtime_error("suppress_shadow_background: frame/reference must be HxWx3 uint8 of equal size");

    const int H = (int)fb.shape[0], W = (int)fb.shape[1];
    double g[3] = {1, 1, 1}, bi[3] = {0, 0, 0};
    bool photo = gains.size() >= 3 && biases.size() >= 3;
    if (photo) { auto gu = gains.unchecked<1>(); auto bu = biases.unchecked<1>(); for (int c = 0; c < 3; ++c) { g[c] = gu(c); bi[c] = bu(c); } }

    py::array_t<uint8_t> out({ (py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)3 });
    uint8_t* dst = (uint8_t*)out.request().ptr;
    const uint8_t* src = (const uint8_t*)fb.ptr;
    const uint8_t* ref = (const uint8_t*)rb.ptr;

    static int sdiv[256];
    static bool sdiv_init = false;
    if (!sdiv_init) {
        sdiv[0] = 0;
        for (int i = 1; i < 256; ++i) sdiv[i] = (int)std::lround((255 << 12) / (1.0 * i));
        sdiv_init = true;
    }
    // valotasapaino LUT per kanava: uint8 -> uint8 (Pythonin float32-clip-astype(uint8) -kaava)
    uint8_t lut[3][256];
    for (int c = 0; c < 3; ++c)
        for (int v = 0; v < 256; ++v) {
            if (!photo) { lut[c][v] = (uint8_t)v; continue; }
            float x = (float)((double)v * g[c] + bi[c]);
            x = std::min(std::max(x, 0.f), 255.f);
            lut[c][v] = (uint8_t)x;
        }

    {
        py::gil_scoped_release release;
        parallelRowsStd(H, [&](const RowRange& rr) {
            for (int y = rr.start; y < rr.end; ++y) {
                const uint8_t* sp = src + (size_t)y * W * 3;
                const uint8_t* rp = ref + (size_t)y * W * 3;
                uint8_t* dp = dst + (size_t)y * W * 3;
                for (int x = 0; x < W; ++x, sp += 3, rp += 3, dp += 3) {
                    const int b = lut[0][sp[0]], gg = lut[1][sp[1]], r = lut[2][sp[2]];
                    const int db = std::abs(b - rp[0]), dg = std::abs(gg - rp[1]), dr = std::abs(r - rp[2]);
                    const int gray = (db * 1868 + dg * 9617 + dr * 4899 + 8192) >> 14;
                    bool bg = (double)gray < diff_threshold;
                    const int vmax = std::max(b, std::max(gg, r));
                    const int vmin = std::min(b, std::min(gg, r));
                    if (!bg) {
                        const int rv = std::max((int)rp[0], std::max((int)rp[1], (int)rp[2]));
                        const double vdrop = (double)(rv - vmax);
                        if (vdrop > v_drop_min && vdrop < v_drop_max) bg = true;
                        else {
                            const int sat = ((vmax - vmin) * sdiv[vmax] + 2048) >> 12;
                            if (sat < ice_s_max && vmax > ice_v_min) bg = true;
                        }
                    }
                    if (bg && !colorGateOk(b, gg, r, rp, vmax, vmin)) bg = false;
                    if (bg) { dp[0] = dp[1] = dp[2] = 255; }
                    else { dp[0] = (uint8_t)b; dp[1] = (uint8_t)gg; dp[2] = (uint8_t)r; }
                }
            }
        });
    }
    return out;
}



// ============================================================
// VAIHEKORRELAATIO MOODIKUVAA VASTEN (stabilointi). Moodikuvan ikkunoitu FFT on sama joka ruudulla, joten Python
// pitaa sen muistissa ja laskee FFT:t cv2:lla; tama moduuli tekee elementtikohtaiset vaiheet rinnakkain:
//   phase_window   : uint8 -> float32 * Hanning-ikkuna
//   phase_mulnorm  : P = ref * conj(F) / |ref * conj(F)| CCS-pakkauksessa (tasmalleen kuin OpenCV)
//   phase_peak     : huippu + 5x5-painopiste kuten cv::phaseCorrelate (fftShift huomioiden)
// Tarkistettu numeerisesti cv2.phaseCorrelatea vastaan (virhe < 1e-6 px).
// ============================================================
static py::array_t<float> phase_window(ArrU8 gray, ArrF window)
{
    auto gb = gray.request(); auto wb = window.request();
    if (gb.ndim != 2 || wb.ndim != 2 || gb.shape[0] != wb.shape[0] || gb.shape[1] != wb.shape[1])
        throw std::runtime_error("phase_window: size mismatch");
    const int H = (int)gb.shape[0], W = (int)gb.shape[1];
    py::array_t<float> out({ (py::ssize_t)H, (py::ssize_t)W });
    float* o = (float*)out.request().ptr;
    const uint8_t* g = (const uint8_t*)gb.ptr; const float* w = (const float*)wb.ptr;
    {
        py::gil_scoped_release release;
        parallelRowsStd(H, [&](const RowRange& r) {
            for (int y = r.start; y < r.end; ++y)
                for (int x = 0; x < W; ++x) o[(size_t)y * W + x] = (float)g[(size_t)y * W + x] * w[(size_t)y * W + x];
        });
    }
    return out;
}

static py::array_t<float> phase_mulnorm(ArrF ref_ccs, ArrF f_ccs)
{
    auto rb = ref_ccs.request(); auto fb = f_ccs.request();
    if (rb.ndim != 2 || fb.ndim != 2 || rb.shape[0] != fb.shape[0] || rb.shape[1] != fb.shape[1])
        throw std::runtime_error("phase_mulnorm: size mismatch");
    const int M = (int)rb.shape[0], N = (int)rb.shape[1];
    py::array_t<float> out({ (py::ssize_t)M, (py::ssize_t)N });
    float* P = (float*)out.request().ptr;
    const float* R = (const float*)rb.ptr; const float* F = (const float*)fb.ptr;
    auto cmul = [](float ar, float ai, float br, float bi, float& re, float& im) {
        float pr = ar * br + ai * bi, pi = ai * br - ar * bi;      // a * conj(b)
        float m = std::sqrt(pr * pr + pi * pi);
        if (m > 1e-30f) { re = pr / m; im = pi / m; } else { re = 0.f; im = 0.f; }
    };
    auto sgn = [](float a, float b) { float v = a * b; return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); };
    {
        py::gil_scoped_release release;
        parallelRowsStd(M, [&](const RowRange& rr) {
            for (int r = rr.start; r < rr.end; ++r) {
                const float* a = R + (size_t)r * N; const float* b = F + (size_t)r * N; float* o = P + (size_t)r * N;
                for (int j = 1; j + 1 < N; j += 2) cmul(a[j], a[j + 1], b[j], b[j + 1], o[j], o[j + 1]);
            }
        });
        for (int col : { 0, N - 1 }) {
            P[col] = sgn(R[col], F[col]);
            for (int r = 1; r + 1 < M; r += 2)
                cmul(R[(size_t)r * N + col], R[(size_t)(r + 1) * N + col], F[(size_t)r * N + col], F[(size_t)(r + 1) * N + col],
                     P[(size_t)r * N + col], P[(size_t)(r + 1) * N + col]);
            P[(size_t)(M - 1) * N + col] = sgn(R[(size_t)(M - 1) * N + col], F[(size_t)(M - 1) * N + col]);
        }
    }
    return out;
}

// C = kaanteis-DFT (reaalinen, ei fftShiftattu). Palauttaa (dx, dy) kuten cv2.phaseCorrelate.
static py::tuple phase_peak(ArrF C_arr)
{
    auto cb = C_arr.request();
    if (cb.ndim != 2) throw std::runtime_error("phase_peak: C must be 2-D");
    const int M = (int)cb.shape[0], N = (int)cb.shape[1];
    cv::Mat C(M, N, CV_32F, (void*)cb.ptr);
    cv::Point maxLoc;
    cv::minMaxLoc(C, nullptr, nullptr, nullptr, &maxLoc);
    int px = (maxLoc.x + N / 2) % N, py_ = (maxLoc.y + M / 2) % M;
    int minr = std::max(py_ - 2, 0), maxr = std::min(py_ + 2, M - 1);
    int minc = std::max(px - 2, 0), maxc = std::min(px + 2, N - 1);
    double sum_i = 0, cx = 0, cy = 0;
    for (int y = minr; y <= maxr; ++y)
        for (int x = minc; x <= maxc; ++x) {
            double v = (double)C.at<float>((y + M / 2) % M, (x + N / 2) % N);
            sum_i += v; cx += (double)x * v; cy += (double)y * v;
        }
    return py::make_tuple((double)(N / 2) - cx / sum_i, (double)(M / 2) - cy / sum_i);
}
