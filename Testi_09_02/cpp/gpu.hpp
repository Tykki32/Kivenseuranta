// ============================================================
// gpu.hpp - GPU-VAIHE B (OpenCL, esim. Intel UHD): warpAffine + remap (stabilointi + linssikorjaus) ja varjosietoinen
// taustanvaimennus + valotasapaino. Sama laskenta kuin CPU:lla (OpenCV 4.x:n kiintopisteinen bilineaarinen interpolointi:
// AB_BITS 10, INTER_BITS 5, 15-bittiset painot ja OpenCV:n painotaulukon korjauskuvio; suppress_shadow_background:n
// kokonaislukuaritmetiikka ja savylaskenta ilman FMA:ta) -> bitti-identtinen tulos.
// OpenCL-kirjasto ladataan ajonaikaisesti (OpenCL.dll / libOpenCL.so); ilman sita kaytetaan CPU-polkua.
// Laite: GPU (ymparistomuuttuja GPU_LAITE=cpu|any testaukseen, esim. pocl).
// Python (esikasittely.LivePrep): gpu_b_init(map1, map2, ref, ...) -> info, gpu_b_warp(frame, M) -> frame_u,
// gpu_b_suppress(gains, biases, H, W) -> seurantakuva (kayttaa edellisen warpin tulosta GPU:lla).
// ============================================================
#pragma once
#include "maskit.hpp"

namespace gpucl {

typedef int32_t cl_int; typedef uint32_t cl_uint; typedef uint64_t cl_ulong;
#ifdef _WIN32
#define GPUCL_CALL __stdcall
#else
#define GPUCL_CALL
#endif
typedef cl_int  (GPUCL_CALL *fnGetPlatformIDs)(cl_uint, void**, cl_uint*);
typedef cl_int  (GPUCL_CALL *fnGetDeviceIDs)(void*, cl_ulong, cl_uint, void**, cl_uint*);
typedef cl_int  (GPUCL_CALL *fnGetDeviceInfo)(void*, cl_uint, size_t, void*, size_t*);
typedef void*   (GPUCL_CALL *fnCreateContext)(const void*, cl_uint, void* const*, void*, void*, cl_int*);
typedef void*   (GPUCL_CALL *fnCreateCommandQueue)(void*, void*, cl_ulong, cl_int*);
typedef void*   (GPUCL_CALL *fnCreateProgramWithSource)(void*, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int  (GPUCL_CALL *fnBuildProgram)(void*, cl_uint, void* const*, const char*, void*, void*);
typedef cl_int  (GPUCL_CALL *fnGetProgramBuildInfo)(void*, void*, cl_uint, size_t, void*, size_t*);
typedef void*   (GPUCL_CALL *fnCreateKernel)(void*, const char*, cl_int*);
typedef cl_int  (GPUCL_CALL *fnSetKernelArg)(void*, cl_uint, size_t, const void*);
typedef void*   (GPUCL_CALL *fnCreateBuffer)(void*, cl_ulong, size_t, void*, cl_int*);
typedef cl_int  (GPUCL_CALL *fnEnqueueNDRangeKernel)(void*, void*, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int  (GPUCL_CALL *fnEnqueueReadBuffer)(void*, void*, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int  (GPUCL_CALL *fnEnqueueWriteBuffer)(void*, void*, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int  (GPUCL_CALL *fnRelease)(void*);

struct State {
    bool ok = false;
    std::string info = "ei alustettu";
    void *ctx = nullptr, *queue = nullptr, *dev = nullptr;
    fnCreateKernel createKernel = nullptr; fnSetKernelArg setArg = nullptr; fnCreateBuffer createBuffer = nullptr;
    fnEnqueueNDRangeKernel enqueue = nullptr; fnEnqueueReadBuffer readBuf = nullptr; fnEnqueueWriteBuffer writeBuf = nullptr;
    fnRelease releaseMem = nullptr, releaseKernel = nullptr;
    fnCreateProgramWithSource createProgram = nullptr; fnBuildProgram buildProgram = nullptr; fnGetProgramBuildInfo getBuildInfo = nullptr;
    // Kaantaa ohjelman; palauttaa sen tai nullptr (err = syy)
    void* buildSource(const char* src, std::string& err) const {
        cl_int e = 0; size_t len = std::strlen(src);
        void* p = createProgram(ctx, 1, &src, &len, &e);
        if (!p || e) { err = "clCreateProgramWithSource epaonnistui"; return nullptr; }
        if (buildProgram(p, 1, &dev, "", nullptr, nullptr) != 0) {
            char log[4096] = {0};
            getBuildInfo(p, dev, 0x1183, sizeof(log) - 1, log, nullptr);
            err = std::string("OpenCL-ytimen kaannos epaonnistui: ") + log; return nullptr;
        }
        return p;
    }
};

static void* openLib()
{
#ifdef _WIN32
    return (void*)LoadLibraryA("OpenCL.dll");
#else
    void* h = dlopen("libOpenCL.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libOpenCL.so", RTLD_LAZY);
    return h;
#endif
}

static void* getSym(void* lib, const char* name)
{
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void initState(State& st)
{
    void* lib = openLib();
    if (!lib) { st.info = "OpenCL-kirjastoa ei loytynyt"; return; }
#define GPUCL_LOAD(var, type, name) type var = (type)getSym(lib, name); if (!var) { st.info = std::string("OpenCL-funktio puuttuu: ") + name; return; }
    GPUCL_LOAD(getPlatforms, fnGetPlatformIDs, "clGetPlatformIDs")
    GPUCL_LOAD(getDevices, fnGetDeviceIDs, "clGetDeviceIDs")
    GPUCL_LOAD(getDeviceInfo, fnGetDeviceInfo, "clGetDeviceInfo")
    GPUCL_LOAD(createContext, fnCreateContext, "clCreateContext")
    GPUCL_LOAD(createQueue, fnCreateCommandQueue, "clCreateCommandQueue")
    GPUCL_LOAD(createProgram, fnCreateProgramWithSource, "clCreateProgramWithSource")
    GPUCL_LOAD(buildProgram, fnBuildProgram, "clBuildProgram")
    GPUCL_LOAD(getBuildInfo, fnGetProgramBuildInfo, "clGetProgramBuildInfo")
#undef GPUCL_LOAD
    st.createKernel = (fnCreateKernel)getSym(lib, "clCreateKernel");
    st.setArg = (fnSetKernelArg)getSym(lib, "clSetKernelArg");
    st.createBuffer = (fnCreateBuffer)getSym(lib, "clCreateBuffer");
    st.enqueue = (fnEnqueueNDRangeKernel)getSym(lib, "clEnqueueNDRangeKernel");
    st.readBuf = (fnEnqueueReadBuffer)getSym(lib, "clEnqueueReadBuffer");
    st.releaseMem = (fnRelease)getSym(lib, "clReleaseMemObject");
    st.releaseKernel = (fnRelease)getSym(lib, "clReleaseKernel");
    st.writeBuf = (fnEnqueueWriteBuffer)getSym(lib, "clEnqueueWriteBuffer");
    st.createProgram = createProgram; st.buildProgram = buildProgram; st.getBuildInfo = getBuildInfo;
    if (!st.createKernel || !st.setArg || !st.createBuffer || !st.enqueue || !st.readBuf || !st.releaseMem || !st.releaseKernel || !st.writeBuf) {
        st.info = "OpenCL-funktioita puuttuu"; return;
    }

    // laitevalinta: GPU_LAITE=gpu (oletus) | cpu | any
    const char* dv = getenv("GPU_LAITE");
    std::string want = dv ? dv : "gpu";
    cl_ulong dtype = want == "cpu" ? (1ULL << 1) : (want == "any" ? 0xFFFFFFFFULL : (1ULL << 2));
    void* platforms[16]; cl_uint np = 0;
    if (getPlatforms(16, platforms, &np) != 0 || np == 0) { st.info = "OpenCL-alustoja ei loytynyt"; return; }
    void* dev = nullptr;
    for (cl_uint p = 0; p < np && !dev; ++p) {
        void* devs[8]; cl_uint nd = 0;
        if (getDevices(platforms[p], dtype, 8, devs, &nd) == 0 && nd > 0) dev = devs[0];
    }
    if (!dev) { st.info = "OpenCL-laitetta (" + want + ") ei loytynyt"; return; }
    char name[256] = {0};
    getDeviceInfo(dev, 0x102B /*CL_DEVICE_NAME*/, sizeof(name) - 1, name, nullptr);

    cl_int err = 0;
    st.ctx = createContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (!st.ctx || err) { st.info = "clCreateContext epaonnistui"; return; }
    st.queue = createQueue(st.ctx, dev, 0, &err);
    if (!st.queue || err) { st.info = "clCreateCommandQueue epaonnistui"; return; }
    st.dev = dev;
    st.ok = true;
    st.info = std::string(name);
}

static State& state()
{
    static State st;
    static std::once_flag once;
    std::call_once(once, [&]() { initState(st); });
    return st;
}

}   // namespace gpucl

// GPU-VAIHE B (Testi_05_03): warpAffine + remap (stabilointi + linssikorjaus) ja varjotoleranssi-taustanvaimennus + valotasapaino
// OpenCL:lla. TASMALLEEN sama laskenta kuin CPU:lla (OpenCV 4.x kiintopisteinen bilineaarinen interpolointi: AB_BITS=10, INTER_BITS=5,
// 15-bittiset painot, OpenCV:n oma painotaulukon korjauskuvio; suppress_shadow_background:n kokonaislukuaritmetiikka ja
// float-sävylaskenta ilman FMA-yhdistelyä) -> tulos bitti-identtinen OpenCV 4.12:n kanssa (testattu).
// API (Python): gpu_b_init(map1, map2, ref_bgr, diff_thr, vdrop_min, vdrop_max, ice_s_max, ice_v_min) -> info
//               gpu_b_warp(frame_bgr, M_2x3) -> frame_u ; gpu_b_suppress(gains, biases) -> frame_for_tracking (kayttaa edellisen warp:n tulosta GPU:lla)
// ============================================================
namespace gpub {
using namespace gpucl;

static const char* KERNEL_B_SRC = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
inline int fetch(__global const uchar* S, int W, int H, int x, int y, int c)
{
    return (x >= 0 && x < W && y >= 0 && y < H) ? (int)S[((size_t)y * W + x) * 3 + c] : 0;
}
inline void bilin(__global const uchar* S, __global uchar* D, int W, int H, int x, int y, int xi, int yi, int a, __global const int* itab)
{
    int w0 = itab[a], w1 = itab[a + 1], w2 = itab[a + 2], w3 = itab[a + 3];
    for (int c = 0; c < 3; ++c) {
        int v = w0 * fetch(S, W, H, xi, yi, c) + w1 * fetch(S, W, H, xi + 1, yi, c) + w2 * fetch(S, W, H, xi, yi + 1, c) + w3 * fetch(S, W, H, xi + 1, yi + 1, c);
        v = (v + 16384) >> 15;
        D[((size_t)y * W + x) * 3 + c] = (uchar)clamp(v, 0, 255);
    }
}
__kernel void warp_k(__global const uchar* src, __global uchar* dst, int W, int H,
                     __global const int* ad, __global const int* bd, __global const int* x0, __global const int* y0, __global const int* itab)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int X = (x0[y] + ad[x]) >> 5, Y = (y0[y] + bd[x]) >> 5;
    bilin(src, dst, W, H, x, y, X >> 5, Y >> 5, ((Y & 31) * 32 + (X & 31)) * 4, itab);
}
__kernel void remap_k(__global const uchar* src, __global uchar* dst, __global const float* m1, __global const float* m2, int W, int H, __global const int* itab)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int X = convert_int_rte(m1[(size_t)y * W + x] * 32.0f), Y = convert_int_rte(m2[(size_t)y * W + x] * 32.0f);
    bilin(src, dst, W, H, x, y, X >> 5, Y >> 5, ((Y & 31) * 32 + (X & 31)) * 4, itab);
}
inline int satFrom(int vmax, int vmin, __global const int* sdiv) { return ((vmax - vmin) * sdiv[vmax] + 2048) >> 12; }
inline int hueOf(int b, int g, int r, __global const float* hs)
{
    int vmax = max(b, max(g, r)), vmin = min(b, min(g, r)), diff = vmax - vmin;
    if (diff == 0) return 0;
    float hscale = hs[diff];
    float h;
    if (vmax == r) h = (float)(g - b) * hscale; else if (vmax == g) h = (float)(b - r) * hscale + 60.f; else h = (float)(r - g) * hscale + 120.f;
    if (h < 0.f) h += 180.f;
    return (int)round(h);
}
__kernel void shadow_k(__global const uchar* fu, __global const uchar* ref, __global uchar* ft, int W, int H,
                       __global const uchar* lut, __global const int* sdiv, __global const float* hs,
                       float diff_thr, float vd_min, float vd_max, int ice_s, int ice_v,
                       int gs, int gh)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t o = ((size_t)y * W + x) * 3;
    int b = lut[fu[o]], g = lut[256 + fu[o + 1]], r = lut[512 + fu[o + 2]];
    int rb = ref[o], rg = ref[o + 1], rr = ref[o + 2];
    int db = abs(b - rb), dg = abs(g - rg), dr = abs(r - rr);
    int gray = (db * 1868 + dg * 9617 + dr * 4899 + 8192) >> 14;
    bool bg = (float)gray < diff_thr;
    int vmax = max(b, max(g, r)), vmin = min(b, min(g, r));
    if (!bg) {
        int rv = max(rb, max(rg, rr));
        float vdrop = (float)(rv - vmax);
        if (vdrop > vd_min && vdrop < vd_max) bg = true;
        else { int sat = satFrom(vmax, vmin, sdiv); if (sat < ice_s && vmax > ice_v) bg = true; }
    }
    if (bg && gs < 256) {
        int rmax = max(rb, max(rg, rr)), rmin = min(rb, min(rg, rr));
        bool ok = satFrom(rmax, rmin, sdiv) < gs;
        if (!ok) {
            int dh = abs(hueOf(b, g, r, hs) - hueOf(rb, rg, rr, hs));
            if (dh > 90) dh = 180 - dh;
            ok = dh <= gh && satFrom(vmax, vmin, sdiv) > gs;
        }
        if (!ok) bg = false;
    }
    if (bg) { ft[o] = 255; ft[o + 1] = 255; ft[o + 2] = 255; }
    else { ft[o] = (uchar)b; ft[o + 1] = (uchar)g; ft[o + 2] = (uchar)r; }
}
)CLC";

struct B {
    std::mutex mx;
    bool ready = false;
    int W = 0, H = 0;
    void *prog = nullptr, *k_warp = nullptr, *k_remap = nullptr, *k_shadow = nullptr;
    void *b_src = nullptr, *b_stab = nullptr, *b_fu = nullptr, *b_ft = nullptr, *b_map1 = nullptr, *b_map2 = nullptr, *b_ref = nullptr;
    void *b_itab = nullptr, *b_sdiv = nullptr, *b_hs = nullptr, *b_lut = nullptr, *b_ad = nullptr, *b_bd = nullptr, *b_x0 = nullptr, *b_y0 = nullptr;
    std::vector<int> ad, bd, x0, y0;
};
static B g_b;

// OpenCV:n bilineaarinen kiintopistepainotaulukko (initInterTab2D, INTER_LINEAR, fixpt) - mukaan lukien sen korjauskuvio (indeksit ksize=2:lla ylittavat taulukon
// ja kirjoittavat seuraavan alkion, kuten OpenCV:ssa; seuraavan taulukon laskenta ylikirjoittaa ne).
static std::vector<int> buildBilinearItab()
{
    const int T = 32, ksize = 2;
    std::vector<short> flat((size_t)T * T * 4 + 16, 0);
    float tab1d[T][2];
    const float scale = 1.f / (float)T;
    for (int i = 0; i < T; ++i) { const float x = (float)i * scale; tab1d[i][0] = 1.f - x; tab1d[i][1] = x; }
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < T; ++j) {
            short* itab = &flat[(size_t)(i * T + j) * 4];
            int isum = 0;
            for (int k1 = 0; k1 < 2; ++k1) {
                const float vy = tab1d[i][k1];
                for (int k2 = 0; k2 < 2; ++k2) {
                    const float v = vy * tab1d[j][k2];
                    itab[k1 * ksize + k2] = cv::saturate_cast<short>(v * 32768.f);
                    isum += itab[k1 * ksize + k2];
                }
            }
            if (isum != 32768) {
                const int diff = isum - 32768;
                int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
                for (int k1 = ksize2; k1 < ksize2 + 2; ++k1)
                    for (int k2 = ksize2; k2 < ksize2 + 2; ++k2) {
                        if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2]) { mk1 = k1; mk2 = k2; }
                        else if (itab[k1 * ksize + k2] > itab[Mk1 * ksize + Mk2]) { Mk1 = k1; Mk2 = k2; }
                    }
                if (diff < 0) itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
                else itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
            }
        }
    std::vector<int> out((size_t)T * T * 4);
    for (size_t k = 0; k < out.size(); ++k) out[k] = flat[k];
    return out;
}

static void* mkBuf(State& st, cl_ulong flags, size_t bytes, void* host, cl_int& err)
{
    cl_int e = 0;
    void* b = st.createBuffer(st.ctx, flags, bytes ? bytes : 1, host, &e);
    if (!b || e) err = e ? e : -1;
    return b;
}

}   // namespace gpub


static std::string gpuBInit(const cv::Mat& map1, const cv::Mat& map2, const cv::Mat& ref,
                            double diff_thr, double vd_min, double vd_max, int ice_s, int ice_v)
{
    using namespace gpucl;
    State& st = state();
    if (!st.ok) return "EI KAYTETTAVISSA: " + st.info;
    std::lock_guard<std::mutex> lk(gpub::g_b.mx);
    gpub::B& g = gpub::g_b;
    g.ready = false;
    if (map1.type() != CV_32FC1 || map2.type() != CV_32FC1 || map1.size() != map2.size() || ref.type() != CV_8UC3 || ref.size() != map1.size())
        return "EI KAYTETTAVISSA: karttojen/referenssin tyyppi tai koko vaara";
    g.W = map1.cols; g.H = map1.rows;
    std::string err;
    if (!g.prog) g.prog = st.buildSource(gpub::KERNEL_B_SRC, err);
    if (!g.prog) return "EI KAYTETTAVISSA: " + err;
    cl_int e = 0;
    g.k_warp = st.createKernel(g.prog, "warp_k", &e); if (e) return "EI KAYTETTAVISSA: warp_k";
    g.k_remap = st.createKernel(g.prog, "remap_k", &e); if (e) return "EI KAYTETTAVISSA: remap_k";
    g.k_shadow = st.createKernel(g.prog, "shadow_k", &e); if (e) return "EI KAYTETTAVISSA: shadow_k";

    const size_t npix = (size_t)g.W * g.H, nbytes = npix * 3;
    const cl_ulong RW = (1ULL << 0), RO = (1ULL << 2), COPY = (1ULL << 5);
    cv::Mat m1c = map1.isContinuous() ? map1 : map1.clone(), m2c = map2.isContinuous() ? map2 : map2.clone(), rc = ref.isContinuous() ? ref : ref.clone();
    auto itab = gpub::buildBilinearItab();
    std::vector<int> sdiv(256, 0);
    for (int i = 1; i < 256; ++i) sdiv[(size_t)i] = (int)std::lround((255 << 12) / (1.0 * i));
    std::vector<float> hs(256, 0.f);
    for (int d = 1; d < 256; ++d) hs[(size_t)d] = 30.f / (float)d;
    g.ad.assign((size_t)g.W, 0); g.bd.assign((size_t)g.W, 0); g.x0.assign((size_t)g.H, 0); g.y0.assign((size_t)g.H, 0);
    cl_int be = 0;
    g.b_src = gpub::mkBuf(st, RW, nbytes, nullptr, be);
    g.b_stab = gpub::mkBuf(st, RW, nbytes, nullptr, be);
    g.b_fu = gpub::mkBuf(st, RW, nbytes, nullptr, be);
    g.b_ft = gpub::mkBuf(st, RW, nbytes, nullptr, be);
    g.b_map1 = gpub::mkBuf(st, RO | COPY, npix * sizeof(float), m1c.data, be);
    g.b_map2 = gpub::mkBuf(st, RO | COPY, npix * sizeof(float), m2c.data, be);
    g.b_ref = gpub::mkBuf(st, RO | COPY, nbytes, rc.data, be);
    g.b_itab = gpub::mkBuf(st, RO | COPY, itab.size() * sizeof(int), itab.data(), be);
    g.b_sdiv = gpub::mkBuf(st, RO | COPY, sdiv.size() * sizeof(int), sdiv.data(), be);
    g.b_hs = gpub::mkBuf(st, RO | COPY, hs.size() * sizeof(float), hs.data(), be);
    g.b_lut = gpub::mkBuf(st, RW, 768, nullptr, be);
    g.b_ad = gpub::mkBuf(st, RW, (size_t)g.W * sizeof(int), nullptr, be);
    g.b_bd = gpub::mkBuf(st, RW, (size_t)g.W * sizeof(int), nullptr, be);
    g.b_x0 = gpub::mkBuf(st, RW, (size_t)g.H * sizeof(int), nullptr, be);
    g.b_y0 = gpub::mkBuf(st, RW, (size_t)g.H * sizeof(int), nullptr, be);
    if (be) return "EI KAYTETTAVISSA: GPU-puskurien luonti epaonnistui";

    cl_int r = 0; cl_uint a;
    int W = g.W, H = g.H;
    a = 0; r |= st.setArg(g.k_warp, a++, sizeof(void*), &g.b_src); r |= st.setArg(g.k_warp, a++, sizeof(void*), &g.b_stab);
    r |= st.setArg(g.k_warp, a++, sizeof(int), &W); r |= st.setArg(g.k_warp, a++, sizeof(int), &H);
    r |= st.setArg(g.k_warp, a++, sizeof(void*), &g.b_ad); r |= st.setArg(g.k_warp, a++, sizeof(void*), &g.b_bd);
    r |= st.setArg(g.k_warp, a++, sizeof(void*), &g.b_x0); r |= st.setArg(g.k_warp, a++, sizeof(void*), &g.b_y0); r |= st.setArg(g.k_warp, a++, sizeof(void*), &g.b_itab);
    a = 0; r |= st.setArg(g.k_remap, a++, sizeof(void*), &g.b_stab); r |= st.setArg(g.k_remap, a++, sizeof(void*), &g.b_fu);
    r |= st.setArg(g.k_remap, a++, sizeof(void*), &g.b_map1); r |= st.setArg(g.k_remap, a++, sizeof(void*), &g.b_map2);
    r |= st.setArg(g.k_remap, a++, sizeof(int), &W); r |= st.setArg(g.k_remap, a++, sizeof(int), &H); r |= st.setArg(g.k_remap, a++, sizeof(void*), &g.b_itab);
    float fthr = (float)diff_thr, fvmin = (float)vd_min, fvmax = (float)vd_max;
    a = 0; r |= st.setArg(g.k_shadow, a++, sizeof(void*), &g.b_fu); r |= st.setArg(g.k_shadow, a++, sizeof(void*), &g.b_ref); r |= st.setArg(g.k_shadow, a++, sizeof(void*), &g.b_ft);
    r |= st.setArg(g.k_shadow, a++, sizeof(int), &W); r |= st.setArg(g.k_shadow, a++, sizeof(int), &H);
    r |= st.setArg(g.k_shadow, a++, sizeof(void*), &g.b_lut); r |= st.setArg(g.k_shadow, a++, sizeof(void*), &g.b_sdiv); r |= st.setArg(g.k_shadow, a++, sizeof(void*), &g.b_hs);
    r |= st.setArg(g.k_shadow, a++, sizeof(float), &fthr); r |= st.setArg(g.k_shadow, a++, sizeof(float), &fvmin); r |= st.setArg(g.k_shadow, a++, sizeof(float), &fvmax);
    r |= st.setArg(g.k_shadow, a++, sizeof(int), &ice_s); r |= st.setArg(g.k_shadow, a++, sizeof(int), &ice_v);
    r |= st.setArg(g.k_shadow, a++, sizeof(int), &g_gate_s_max); r |= st.setArg(g.k_shadow, a++, sizeof(int), &g_gate_h_tol);
    if (r) return "EI KAYTETTAVISSA: ytimen argumenttien asetus epaonnistui";
    g.ready = true;
    return st.info + " (vaihe B: warp+remap+varjosuodatus GPU:lla)";
}

static bool gpuBWarp(const cv::Mat& frame, const cv::Mat& M, cv::Mat& out)
{
    using namespace gpucl;
    State& st = state();
    std::lock_guard<std::mutex> lk(gpub::g_b.mx);
    gpub::B& g = gpub::g_b;
    if (!st.ok || !g.ready || frame.cols != g.W || frame.rows != g.H || frame.type() != CV_8UC3 || M.rows != 2 || M.cols != 3 || M.type() != CV_64F) return false;
    cv::Mat iM;
    cv::invertAffineTransform(M, iM);
    const double* m = iM.ptr<double>();
    const int AB_BITS = 10, AB_SCALE = 1 << AB_BITS, round_delta = AB_SCALE / 32 / 2;
    for (int x = 0; x < g.W; ++x) { g.ad[(size_t)x] = cv::saturate_cast<int>(m[0] * x * AB_SCALE); g.bd[(size_t)x] = cv::saturate_cast<int>(m[3] * x * AB_SCALE); }
    for (int y = 0; y < g.H; ++y) {
        g.x0[(size_t)y] = cv::saturate_cast<int>((m[1] * y + m[2]) * AB_SCALE) + round_delta;
        g.y0[(size_t)y] = cv::saturate_cast<int>((m[4] * y + m[5]) * AB_SCALE) + round_delta;
    }
    cv::Mat fc = frame.isContinuous() ? frame : frame.clone();
    const size_t nbytes = (size_t)g.W * g.H * 3;
    cl_int r = 0;
    r |= st.writeBuf(st.queue, g.b_src, 0, 0, nbytes, fc.data, 0, nullptr, nullptr);
    r |= st.writeBuf(st.queue, g.b_ad, 0, 0, (size_t)g.W * sizeof(int), g.ad.data(), 0, nullptr, nullptr);
    r |= st.writeBuf(st.queue, g.b_bd, 0, 0, (size_t)g.W * sizeof(int), g.bd.data(), 0, nullptr, nullptr);
    r |= st.writeBuf(st.queue, g.b_x0, 0, 0, (size_t)g.H * sizeof(int), g.x0.data(), 0, nullptr, nullptr);
    r |= st.writeBuf(st.queue, g.b_y0, 0, 0, (size_t)g.H * sizeof(int), g.y0.data(), 0, nullptr, nullptr);
    size_t gs[2] = { (size_t)g.W, (size_t)g.H };
    if (!r) r = st.enqueue(st.queue, g.k_warp, 2, nullptr, gs, nullptr, 0, nullptr, nullptr);
    if (!r) r = st.enqueue(st.queue, g.k_remap, 2, nullptr, gs, nullptr, 0, nullptr, nullptr);
    out.create(g.H, g.W, CV_8UC3);
    if (!r) r = st.readBuf(st.queue, g.b_fu, 1, 0, nbytes, out.data, 0, nullptr, nullptr);
    return r == 0;
}

static bool gpuBSuppress(const double* gains, const double* biases, bool photo, cv::Mat& out)
{
    using namespace gpucl;
    State& st = state();
    std::lock_guard<std::mutex> lk(gpub::g_b.mx);
    gpub::B& g = gpub::g_b;
    if (!st.ok || !g.ready) return false;
    uint8_t lut[768];
    for (int c = 0; c < 3; ++c)
        for (int v = 0; v < 256; ++v) {
            if (!photo) { lut[c * 256 + v] = (uint8_t)v; continue; }
            float x = (float)((double)v * gains[c] + biases[c]);
            x = std::min(std::max(x, 0.f), 255.f);
            lut[c * 256 + v] = (uint8_t)x;
        }
    const size_t nbytes = (size_t)g.W * g.H * 3;
    cl_int r = st.writeBuf(st.queue, g.b_lut, 0, 0, 768, lut, 0, nullptr, nullptr);
    size_t gs[2] = { (size_t)g.W, (size_t)g.H };
    if (!r) r = st.enqueue(st.queue, g.k_shadow, 2, nullptr, gs, nullptr, 0, nullptr, nullptr);
    out.create(g.H, g.W, CV_8UC3);
    if (!r) r = st.readBuf(st.queue, g.b_ft, 1, 0, nbytes, out.data, 0, nullptr, nullptr);
    return r == 0;
}


// ------------------------------------------------------------
// Python-rajapinta
// ------------------------------------------------------------
static std::string gpu_b_init(ArrF m1, ArrF m2, ArrU8 ref, double diff_thr, double vd_min, double vd_max, int ice_s, int ice_v)
{
    auto b1 = m1.request(); auto b2 = m2.request(); auto br = ref.request();
    if (b1.ndim != 2 || b2.ndim != 2 || br.ndim != 3 || br.shape[2] != 3) return "EI KAYTETTAVISSA: vaarat taulukkomuodot";
    cv::Mat mm1((int)b1.shape[0], (int)b1.shape[1], CV_32FC1, b1.ptr), mm2((int)b2.shape[0], (int)b2.shape[1], CV_32FC1, b2.ptr);
    cv::Mat rr((int)br.shape[0], (int)br.shape[1], CV_8UC3, br.ptr);
    return gpuBInit(mm1, mm2, rr, diff_thr, vd_min, vd_max, ice_s, ice_v);
}

// stabiloitu + linssikorjattu ruutu tai None (GPU ei kaytettavissa)
static py::object gpu_b_warp(ArrU8 frame, ArrD M)
{
    auto bf = frame.request(); auto bm = M.request();
    if (bf.ndim != 3 || bf.shape[2] != 3 || bm.size != 6) return py::none();
    cv::Mat f((int)bf.shape[0], (int)bf.shape[1], CV_8UC3, bf.ptr);
    cv::Mat mm(2, 3, CV_64F, bm.ptr);
    py::array_t<uint8_t> out({ bf.shape[0], bf.shape[1], (py::ssize_t)3 });
    cv::Mat o((int)bf.shape[0], (int)bf.shape[1], CV_8UC3, out.mutable_data());
    bool ok;
    {
        py::gil_scoped_release rel;
        cv::Mat tmp;
        ok = gpuBWarp(f, mm, tmp);
        if (ok) std::memcpy(o.data, tmp.data, (size_t)tmp.total() * 3);
    }
    if (!ok) return py::none();
    return out;
}

// edellisen gpu_b_warp-tuloksen taustanvaimennus + valotasapaino tai None
static py::object gpu_b_suppress(ArrD gains, ArrD biases, int H, int W)
{
    const bool photo = gains.size() >= 3 && biases.size() >= 3;
    double g[3] = {1, 1, 1}, bi[3] = {0, 0, 0};
    if (photo) { auto gu = gains.unchecked<1>(); auto bu = biases.unchecked<1>(); for (int c = 0; c < 3; ++c) { g[c] = gu(c); bi[c] = bu(c); } }
    py::array_t<uint8_t> out({ (py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)3 });
    bool ok;
    {
        py::gil_scoped_release rel;
        cv::Mat tmp;
        ok = gpuBSuppress(g, bi, photo, tmp);
        if (ok) std::memcpy(out.mutable_data(), tmp.data, (size_t)tmp.total() * 3);
    }
    if (!ok) return py::none();
    return out;
}
