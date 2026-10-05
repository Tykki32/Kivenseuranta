// ============================================================
// profiili.hpp - kiviprofiilin opettelu (Python: profiili.py):
//   scan_stone_candidates: kivikandidaatit yhdesta ruudusta (taustanvaimennus + graniittimaski -> kontuurit ->
//                          ellipsi -> suodatus -> fyysinen paikka jaatasolla)
//   fit_stone_profile_cpp: kiviprofiilin (sade, korkeus, muoto, kahvan lovi) sovitus usean havainnon aariviivoihin
//   alpha_observation_cpp / alpha_contour_cpp: kiven reuna alipikselitarkasti peittavyydesta (alfa-kartta)
// ============================================================
#pragma once
#include "esikasittely.hpp"

// ------------------------------------------------------------
// KIVIKANDIDAATIT
// ------------------------------------------------------------
static std::pair<double, double> rayPlaneIntersectionZ0(
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double u, double v)
{
    cv::Matx33d K_inv = K.inv();
    cv::Vec3d uv1(u, v, 1.0);
    cv::Vec3d ray_cam = K_inv * uv1;

    cv::Matx33d R_t = R.t();
    cv::Vec3d C = -(R_t * t);
    cv::Vec3d d = R_t * ray_cam;

    double s = (0.0 - C[2]) / d[2];
    double X = C[0] + s * d[0];
    double Y = C[1] + s * d[1];

    return { X, Y };
}


struct StoneCandidateFast {
    cv::RotatedRect ellipse;
    std::vector<cv::Point> contour;
    double area = 0.0;
    double X_cm = 0.0, Y_cm = 0.0;
};

static std::vector<StoneCandidateFast> findStoneCandidatesFast(
    const cv::Mat& mask, const cv::Matx33d& H_final,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double min_area, double max_area, double min_fill_ratio, double min_aspect_ratio,
    double x_min, double x_max, double y_min, double y_max,
    double pixels_per_cm, double output_x_min_cm, double output_y_max_cm)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<StoneCandidateFast> candidates;

    for (auto& contour : contours) {

        double area = cv::contourArea(contour);

        if (area < min_area || area > max_area || contour.size() < 5)
            continue;

        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        double cx = ellipse.center.x, cy = ellipse.center.y;
        double w = ellipse.size.width, h = ellipse.size.height;

        double ellipse_area = M_PI * (w / 2.0) * (h / 2.0);
        if (ellipse_area < 1e-6)
            continue;

        double fill_ratio = area / ellipse_area;
        if (fill_ratio < min_fill_ratio)
            continue;

        double aspect_ratio = std::min(w, h) / std::max(w, h);
        if (aspect_ratio < min_aspect_ratio)
            continue;

        cv::Vec3d hp = H_final * cv::Vec3d(cx, cy, 1.0);
        double out_px_x = hp[0] / hp[2];
        double out_px_y = hp[1] / hp[2];

        double phys_x = out_px_x / pixels_per_cm + output_x_min_cm;
        double phys_y = output_y_max_cm - out_px_y / pixels_per_cm;

        if (!(phys_x >= x_min && phys_x <= x_max && phys_y >= y_min && phys_y <= y_max))
            continue;

        StoneCandidateFast cand;
        cand.ellipse = ellipse;
        cand.contour = contour;
        cand.area = area;

        auto pos = rayPlaneIntersectionZ0(K, R, t, cx, cy);
        cand.X_cm = pos.first;
        cand.Y_cm = pos.second;

        candidates.push_back(cand);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const StoneCandidateFast& a, const StoneCandidateFast& b) { return a.area > b.area; });

    return candidates;
}


static std::vector<StoneCandidateFast> scanStoneCandidates(
    const cv::Mat& frame_bgr, const cv::Mat& background_reference, double diff_threshold,
    const cv::Matx33d& H_final, const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double min_area, double max_area, double min_fill_ratio, double min_aspect_ratio,
    double x_min, double x_max, double y_min, double y_max,
    double pixels_per_cm, double output_x_min_cm, double output_y_max_cm,
    int roi_x0 = -1, int roi_y0 = -1, int roi_x1 = -1, int roi_y1 = -1)
{
    cv::Mat mask;
    const bool use_roi = roi_x1 > roi_x0 && roi_y1 > roi_y0;
    if (use_roi) {
        // Testi_03_03: taustanvaimennus + granittimaski VAIN pelialueen kuva-alueella (ROI); ulkopuolella
        // maski on nolla. Laskenta reunuksella (PAD) ja reunus nollataan (taustan sumennus ei vaikuta reunaan).
        const int PAD = 40;
        cv::Rect full(0, 0, frame_bgr.cols, frame_bgr.rows);
        cv::Rect inner = cv::Rect(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0) & full;
        mask = cv::Mat::zeros(frame_bgr.size(), CV_8UC1);
        if (inner.width > 0 && inner.height > 0) {
            cv::Rect padded = cv::Rect(inner.x - PAD, inner.y - PAD, inner.width + 2 * PAD, inner.height + 2 * PAD) & full;
            cv::Mat crop = frame_bgr(padded);
            cv::Mat ref_crop = (!background_reference.empty() && background_reference.size() == frame_bgr.size())
                ? background_reference(padded) : cv::Mat();
            cv::Mat filtered = suppressStaticBackground(crop, ref_crop, diff_threshold);
            cv::Mat sub = createGraniteMask(filtered);
            cv::Rect in_sub(inner.x - padded.x, inner.y - padded.y, inner.width, inner.height);
            sub(in_sub).copyTo(mask(inner));
        }
    } else {
        cv::Mat frame_filtered = suppressStaticBackground(frame_bgr, background_reference, diff_threshold);
        mask = createGraniteMask(frame_filtered);
    }
    return findStoneCandidatesFast(
        mask, H_final, K, R, t, min_area, max_area, min_fill_ratio, min_aspect_ratio,
        x_min, x_max, y_min, y_max, pixels_per_cm, output_x_min_cm, output_y_max_cm
    );
}


static py::list stoneCandidatesToList(const std::vector<StoneCandidateFast>& cands)
{
    py::list out;

    for (auto& c : cands) {

        py::dict d;

        py::tuple center = py::make_tuple(c.ellipse.center.x, c.ellipse.center.y);
        py::tuple size = py::make_tuple(c.ellipse.size.width, c.ellipse.size.height);
        d["ellipse"] = py::make_tuple(center, size, c.ellipse.angle);

        std::vector<py::ssize_t> shape{ (py::ssize_t)c.contour.size(), 1, 2 };
        py::array_t<int32_t> contour_arr(shape);
        auto buf = contour_arr.mutable_unchecked<3>();
        for (py::ssize_t i = 0; i < (py::ssize_t)c.contour.size(); ++i) {
            buf(i, 0, 0) = c.contour[(size_t)i].x;
            buf(i, 0, 1) = c.contour[(size_t)i].y;
        }
        d["contour"] = contour_arr;

        d["pos_cm"] = py::make_tuple(c.X_cm, c.Y_cm);

        out.append(d);
    }

    return out;
}


static py::list scan_stone_candidates(
    ArrU8 frame_bgr, ArrU8 background_reference, ArrD H_final_arr, ArrD K_arr, ArrD R_arr, ArrD t_arr,
    double min_area, double max_area, double min_fill_ratio, double min_aspect_ratio,
    double x_min, double x_max, double y_min, double y_max,
    double pixels_per_cm, double output_x_min_cm, double output_y_max_cm,
    double diff_threshold = 30.0,
    int roi_x0 = -1, int roi_y0 = -1, int roi_x1 = -1, int roi_y1 = -1)
{
    cv::Mat frame_mat = bgrView(frame_bgr);
    if (frame_mat.empty())
        throw std::runtime_error("frame_bgr must be HxWx3 uint8 BGR");
    cv::Mat ref_mat = bgrView(background_reference);

    auto H_final = parseMat33(H_final_arr);
    auto K = parseMat33(K_arr);
    auto R = parseMat33(R_arr);
    auto t = parseVec3(t_arr);

    std::vector<StoneCandidateFast> result;
    {
        py::gil_scoped_release release;
        result = scanStoneCandidates(
            frame_mat, ref_mat, diff_threshold, H_final, K, R, t,
            min_area, max_area, min_fill_ratio, min_aspect_ratio,
            x_min, x_max, y_min, y_max, pixels_per_cm, output_x_min_cm, output_y_max_cm,
            roi_x0, roi_y0, roi_x1, roi_y1
        );
    }

    return stoneCandidatesToList(result);
}

// ------------------------------------------------------------
// KIVIPROFIILIN SOVITUS: Levenberg-Marquardt profiilin parametreille (sade, korkeus, muotopoikkeamat, kahvan lovi)
// ja jokaisen havainnon paikalle (X0, Y0). Jakobiaani lasketaan harvasti: havainnon oma paikka vaikuttaa vain sen
// omaan jaannoslohkoon. Profiilimalli kuten kivimalli.py (Catmull-Rom-sade korkeuden funktiona).
// ------------------------------------------------------------
namespace pfit {

static const int HALF_LEN = 5;                       // _TEMPLATE_HALF_LEN
static const int FULL_LEN = 2 * HALF_LEN - 1;        // 9
static const int N_SHAPE = HALF_LEN - 1;             // 4
static const int EQ_IDX = HALF_LEN - 1;              // 4
static const double HALF_Z[HALF_LEN] = { 0.00, 0.18, 0.30, 0.45, 0.50 };
static const double HALF_R[HALF_LEN] = { 0.75, 0.94, 0.98, 1.00, 1.00 };

struct Ctx {
    cv::Matx33d K, R;
    cv::Vec3d t;
    std::vector<std::vector<cv::Point2d>> sampled;   // kiven aariviivapisteet
    double zt[FULL_LEN], rt[FULL_LEN];               // taysi (peilattu) malli
    double hmin, hmax, hdmin, hdmax, reg_eff;
    int n_stone = 0;
    int n_theta = 28, n_per_segment = 5;
};

static void buildTemplate(Ctx& c)
{
    for (int i = 0; i < HALF_LEN; ++i) { c.zt[i] = HALF_Z[i]; c.rt[i] = HALF_R[i]; }
    // z_top = 1 - z_half[-2::-1], r_top = r_half[-2::-1]
    for (int k = 0; k < HALF_LEN - 1; ++k) {
        int src = HALF_LEN - 2 - k;
        c.zt[HALF_LEN + k] = 1.0 - HALF_Z[src];
        c.rt[HALF_LEN + k] = HALF_R[src];
    }
}

static double sigmoidBounded(double x, double lo, double hi) { return lo + (hi - lo) / (1.0 + std::exp(-x)); }
static double inverseSigmoidBounded(double v, double lo, double hi)
{
    double p = (v - lo) / (hi - lo);
    p = std::min(std::max(p, 1e-6), 1.0 - 1e-6);
    return std::log(p / (1.0 - p));
}

// _catmull_rom_r_frac yhdelle z:lle
static double catmullRom(const Ctx& c, const double* rf, double z)
{
    const int n = FULL_LEN;
    double zext_first = c.zt[0], zext_last = c.zt[n - 1];
    (void)zext_first; (void)zext_last;
    double rext[FULL_LEN + 2];
    rext[0] = rf[0];
    for (int i = 0; i < n; ++i) rext[i + 1] = rf[i];
    rext[n + 1] = rf[n - 1];
    double result = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        double z0 = c.zt[i], z1 = c.zt[i + 1];
        if (z >= z0 && z <= z1) {
            double span = z1 - z0;
            double t = span > 1e-12 ? (z - z0) / span : 0.0;
            double p0 = rext[i], p1 = rext[i + 1], p2 = rext[i + 2], p3 = rext[i + 3];
            double m1 = (p2 - p0) / 2.0, m2 = (p3 - p1) / 2.0;
            double t2 = t * t, t3 = t2 * t;
            double h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
            result = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2;      // myohempi segmentti ylikirjoittaa (kuten numpy-silmukka)
        }
    }
    return result;
}

static inline bool projectPt(const Ctx& c, double X, double Y, double Z, cv::Point2f& out)
{
    cv::Vec3d pc = c.R * cv::Vec3d(X, Y, Z) + c.t;
    cv::Vec3d pi = c.K * pc;
    double u = pi[0] / pi[2], v = pi[1] / pi[2];
    if (!std::isfinite(u) || !std::isfinite(v)) return false;
    out = cv::Point2f((float)u, (float)v);
    return true;
}

// _predicted_stone_hull: false jos ei kelvollinen
static bool predictedHullP(const Ctx& c, double X0, double Y0, double R_max, double H_total,
                           const double* deltas_full, std::vector<cv::Point2f>& hull)
{
    R_max = std::abs(R_max);
    H_total = std::max(std::abs(H_total), 1e-6);
    double rf[FULL_LEN];
    for (int i = 0; i < FULL_LEN; ++i) rf[i] = c.rt[i] + deltas_full[i];
    rf[EQ_IDX] = 1.0;
    for (int i = 0; i < FULL_LEN; ++i) rf[i] = std::min(std::max(rf[i], 0.05), 1.0);

    const int n_dense = std::max(FULL_LEN * c.n_per_segment, 2);
    std::vector<cv::Point2f> pts;
    pts.reserve((size_t)n_dense * (size_t)c.n_theta);
    static thread_local std::vector<double> cs, sn;
    if ((int)cs.size() != c.n_theta) {
        cs.resize((size_t)c.n_theta); sn.resize((size_t)c.n_theta);
        for (int k = 0; k < c.n_theta; ++k) { double th = 2.0 * M_PI * (double)k / (double)c.n_theta; cs[(size_t)k] = std::cos(th); sn[(size_t)k] = std::sin(th); }
    }
    for (int i = 0; i < n_dense; ++i) {
        double zf = (n_dense == 1) ? 0.0 : (double)i / (double)(n_dense - 1);
        double r = catmullRom(c, rf, zf) * R_max;
        double z = zf * H_total;
        for (int k = 0; k < c.n_theta; ++k) {
            cv::Point2f q;
            if (!projectPt(c, X0 + r * cs[(size_t)k], Y0 + r * sn[(size_t)k], z, q)) return false;
            pts.push_back(q);
        }
    }
    cv::convexHull(pts, hull);
    return hull.size() >= 3;
}

static bool notchHullP(const Ctx& c, double X0, double Y0, double R_max, double H_total, double r_frac,
                       std::vector<cv::Point2f>& hull)
{
    double r_cm = std::abs(r_frac) * std::abs(R_max);
    std::vector<cv::Point2f> pts;
    pts.reserve((size_t)c.n_theta);
    for (int k = 0; k < c.n_theta; ++k) {
        double th = 2.0 * M_PI * (double)k / (double)c.n_theta;
        cv::Point2f q;
        if (!projectPt(c, X0 + r_cm * std::cos(th), Y0 + r_cm * std::sin(th), H_total, q)) return false;
        pts.push_back(q);
    }
    cv::convexHull(pts, hull);
    return true;
}

// Kiven yhden residuaalilohkon (aariviivapisteiden etumerkilliset etaisyydet)
static void stoneBlock(const Ctx& c, int si, double X0, double Y0, double R_max, double H_total,
                       double handle, const double* deltas_full, double* out)
{
    const auto& pts = c.sampled[(size_t)si];
    std::vector<cv::Point2f> hull, notch;
    if (!predictedHullP(c, X0, Y0, R_max, H_total, deltas_full, hull)) {
        for (size_t i = 0; i < pts.size(); ++i) out[i] = 1000.0;
        return;
    }
    bool has_notch = notchHullP(c, X0, Y0, R_max, H_total, handle, notch);
    for (size_t i = 0; i < pts.size(); ++i) {
        cv::Point2f p((float)pts[i].x, (float)pts[i].y);
        double d_outer = cv::pointPolygonTest(hull, p, true);
        double d;
        if (!has_notch || d_outer <= 0) d = d_outer;
        else {
            double d_notch = cv::pointPolygonTest(notch, p, true);
            if (d_notch > 0) d = -d_notch;
            else d = std::min(d_outer, -d_notch);
        }
        out[i] = d;
    }
}

struct Unpacked { double R, H, handle, deltas_full[FULL_LEN]; };

static Unpacked unpack(const Ctx& c, const std::vector<double>& p, double hd_lo, double hd_hi)
{
    Unpacked u;
    u.R = p[0];
    u.H = sigmoidBounded(p[1], c.hmin, c.hmax);
    u.handle = sigmoidBounded(p[2], hd_lo, hd_hi);
    for (int i = 0; i < FULL_LEN; ++i) u.deltas_full[i] = 0.0;
    for (int i = 0; i < N_SHAPE; ++i) {
        u.deltas_full[i] = p[3 + (size_t)i];
        u.deltas_full[HALF_LEN + (N_SHAPE - 1 - i)] = p[3 + (size_t)i];   // peilaus: full[HALF_LEN:] = half[::-1]
    }
    return u;
}

static bool solveLinear(std::vector<double>& A, std::vector<double>& b, int n, std::vector<double>& x)
{
    for (int col = 0; col < n; ++col) {
        int piv = col; double mx = std::abs(A[(size_t)col * n + col]);
        for (int r = col + 1; r < n; ++r) { double v = std::abs(A[(size_t)r * n + col]); if (v > mx) { mx = v; piv = r; } }
        if (mx == 0.0) return false;
        if (piv != col) { for (int j = 0; j < n; ++j) std::swap(A[(size_t)col * n + j], A[(size_t)piv * n + j]); std::swap(b[(size_t)col], b[(size_t)piv]); }
        for (int r = col + 1; r < n; ++r) {
            double f = A[(size_t)r * n + col] / A[(size_t)col * n + col];
            if (f == 0.0) continue;
            for (int j = col; j < n; ++j) A[(size_t)r * n + j] -= f * A[(size_t)col * n + j];
            b[(size_t)r] -= f * b[(size_t)col];
        }
    }
    x.assign((size_t)n, 0.0);
    for (int r = n - 1; r >= 0; --r) {
        double sum = b[(size_t)r];
        for (int j = r + 1; j < n; ++j) sum -= A[(size_t)r * n + j] * x[(size_t)j];
        x[(size_t)r] = sum / A[(size_t)r * n + r];
    }
    return true;
}

// v5.7: riippumattomat tehtavat 0..n-1 usealle saikeelle (dynaaminen jako). Jokainen tehtava kirjoittaa vain omaan
// tulospaikkaansa -> tulos on bitti-identtinen perakkaisen silmukan kanssa.
template <class F>
static void parallelTasks(int n, F fn)
{
    unsigned hw = std::thread::hardware_concurrency();
    int nt = std::min(n, (int)std::max(1u, std::min(8u, hw == 0 ? 2u : hw)));
    if (nt <= 1) { for (int i = 0; i < n; ++i) fn(i); return; }
    std::atomic<int> next(0);
    auto worker = [&]() { for (int i = next.fetch_add(1); i < n; i = next.fetch_add(1)) fn(i); };
    std::vector<std::thread> th;
    th.reserve((size_t)nt - 1);
    for (int k = 1; k < nt; ++k) th.emplace_back(worker);
    worker();
    for (auto& x : th) x.join();
}

} // namespace pfit


static py::dict fit_stone_profile_cpp(
    ArrD K_arr, ArrD R_arr, ArrD t_arr, py::list contours, ArrD positions0_arr,
    int n_sample, double initial_radius, double height_min, double height_max,
    double handle_min, double handle_max, double handle_init, double reg_weight, int max_iterations,
    double r_fixed = -1.0)
{
    using namespace pfit;
    Ctx c;
    c.K = parseMat33(K_arr); c.R = parseMat33(R_arr); c.t = parseVec3(t_arr);
    c.hmin = height_min; c.hmax = height_max;
    buildTemplate(c);

    const int ns = (int)py::len(contours);
    c.n_stone = ns;
    if (ns < 2) throw std::runtime_error("fit_stone_profile_cpp: vahintaan 2 kiveä");
    for (int i = 0; i < ns; ++i) {
        // Testi_03_04: ääriviivat liukulukuina (alipikselireuna, alfa-ääriviiva); kokonaisluvut (cv2-ääriviivat) kelpaavat edelleen (forcecast)
        auto arr = py::cast<ArrD>(contours[(size_t)i]);
        auto b = arr.request();
        const double* d = (const double*)b.ptr;
        size_t total = (size_t)b.size / 2;
        std::vector<cv::Point2d> all(total);
        for (size_t k = 0; k < total; ++k) all[k] = cv::Point2d(d[2 * k], d[2 * k + 1]);
        std::vector<cv::Point2d> smp;
        if ((int)total <= n_sample) smp = all;
        else {
            double step = (double)(total - 1) / (double)(n_sample - 1);
            for (int k = 0; k < n_sample; ++k) {
                size_t idx = (k == n_sample - 1) ? total - 1 : (size_t)((double)k * step);
                smp.push_back(all[idx]);
            }
        }
        c.sampled.push_back(std::move(smp));
    }
    c.reg_eff = reg_weight * ((double)ns / 3.0);

    // residuaalilohkojen offsetit
    std::vector<size_t> off((size_t)ns + 1, 0);
    for (int i = 0; i < ns; ++i) off[(size_t)i + 1] = off[(size_t)i] + c.sampled[(size_t)i].size();
    const size_t m_data = off[(size_t)ns];
    const size_t m = m_data + (size_t)N_SHAPE;
    const int np = 3 + N_SHAPE + 2 * ns;

    auto pos0 = positions0_arr.unchecked<2>();
    std::vector<double> params((size_t)np, 0.0);
    params[0] = (r_fixed > 0.0) ? r_fixed : initial_radius;   // r_fixed > 0: R pidetaan kiinteana (ei Jacobian-saraketta)
    params[1] = inverseSigmoidBounded(0.5 * (height_min + height_max), height_min, height_max);
    params[2] = inverseSigmoidBounded(handle_init, handle_min, handle_max);
    for (int i = 0; i < ns; ++i) { params[(size_t)(3 + N_SHAPE + 2 * i)] = pos0(i, 0); params[(size_t)(3 + N_SHAPE + 2 * i + 1)] = pos0(i, 1); }

    auto blockFor = [&](const std::vector<double>& p, int si, double* out) {
        Unpacked u = unpack(c, p, handle_min, handle_max);
        double X0 = p[(size_t)(3 + N_SHAPE + 2 * si)], Y0 = p[(size_t)(3 + N_SHAPE + 2 * si + 1)];
        stoneBlock(c, si, X0, Y0, u.R, u.H, u.handle, u.deltas_full, out);
    };
    // v5.7: par = true -> kivien lohkot rinnan (lohkot ovat toisistaan riippumattomia, kukin omaan kohtaansa r:ssa -> sama tulos)
    auto fullResiduals = [&](const std::vector<double>& p, std::vector<double>& r, bool par = false) {
        r.assign(m, 0.0);
        Unpacked u = unpack(c, p, handle_min, handle_max);
        auto one = [&](int i) {
            double X0 = p[(size_t)(3 + N_SHAPE + 2 * i)], Y0 = p[(size_t)(3 + N_SHAPE + 2 * i + 1)];
            stoneBlock(c, i, X0, Y0, u.R, u.H, u.handle, u.deltas_full, &r[off[(size_t)i]]);
        };
        if (par) parallelTasks(ns, one);
        else for (int i = 0; i < ns; ++i) one(i);
        for (int k = 0; k < N_SHAPE; ++k) r[m_data + (size_t)k] = p[(size_t)(3 + k)] * c.reg_eff;
    };

    std::vector<double> resid, trial_resid;
    double lam = 1e-3, cost = 0.0;
    const double eps = 1e-6;
    {
        py::gil_scoped_release release;
        fullResiduals(params, resid, true);
        for (double v : resid) cost += v * v;

        std::vector<double> J(m * (size_t)np), JTJ((size_t)np * np), JTr((size_t)np), A, b, delta, tmp(m);
        for (int it = 0; it < max_iterations; ++it) {
            std::fill(J.begin(), J.end(), 0.0);
            // v5.7: Jacobin sarakkeet rinnan - kukin sarake j kirjoittaa vain J[.., j]:hin ja lukee params/resid -taulukoita
            // (ei muutu taman vaiheen aikana) -> bitti-identtinen perakkaisen silmukan kanssa.
            parallelTasks(np, [&](int j) {
                if (j == 0 && r_fixed > 0.0) return;      // kiinteä R: sarake nolla
                double step = eps * std::max(1.0, std::abs(params[(size_t)j]));
                std::vector<double> pp = params;
                pp[(size_t)j] += step;
                if (j >= 3 + N_SHAPE) {
                    // kiven oma sijainti: vain sen lohko muuttuu
                    int si = (j - 3 - N_SHAPE) / 2;
                    size_t len = c.sampled[(size_t)si].size();
                    std::vector<double> blk(len);
                    blockFor(pp, si, blk.data());
                    for (size_t k = 0; k < len; ++k)
                        J[(off[(size_t)si] + k) * (size_t)np + (size_t)j] = (blk[k] - resid[off[(size_t)si] + k]) / step;
                } else {
                    std::vector<double> rp;
                    fullResiduals(pp, rp);
                    for (size_t k = 0; k < m; ++k) J[k * (size_t)np + (size_t)j] = (rp[k] - resid[k]) / step;
                }
            });
            std::fill(JTJ.begin(), JTJ.end(), 0.0);
            std::fill(JTr.begin(), JTr.end(), 0.0);
            for (size_t k = 0; k < m; ++k) {
                const double* row = &J[k * (size_t)np];
                double rk = resid[k];
                for (int a = 0; a < np; ++a) {
                    double ra = row[a];
                    if (ra == 0.0) continue;
                    JTr[(size_t)a] += ra * rk;
                    for (int bb = 0; bb < np; ++bb) JTJ[(size_t)a * np + bb] += ra * row[bb];
                }
            }
            std::vector<double> diag((size_t)np);
            for (int a = 0; a < np; ++a) diag[(size_t)a] = JTJ[(size_t)a * np + a] + 1e-12;

            bool step_taken = false; double rel_improvement = 0.0;
            for (int inner = 0; inner < 12; ++inner) {
                A = JTJ;
                for (int a = 0; a < np; ++a) A[(size_t)a * np + a] += lam * diag[(size_t)a];
                b.assign((size_t)np, 0.0);
                for (int a = 0; a < np; ++a) b[(size_t)a] = -JTr[(size_t)a];
                if (!solveLinear(A, b, np, delta)) { lam *= 10.0; continue; }
                std::vector<double> trial = params;
                for (int a = 0; a < np; ++a) trial[(size_t)a] += delta[(size_t)a];
                fullResiduals(trial, trial_resid, true);
                double trial_cost = 0.0;
                for (double v : trial_resid) trial_cost += v * v;
                if (trial_cost < cost) {
                    rel_improvement = (cost - trial_cost) / std::max(cost, 1e-12);
                    params = trial; resid = trial_resid; cost = trial_cost;
                    lam = std::max(lam / 5.0, 1e-12);
                    step_taken = true;
                    break;
                }
                lam *= 5.0;
            }
            if (!step_taken || rel_improvement < 1e-10) break;
        }
    }

    Unpacked u = unpack(c, params, handle_min, handle_max);
    py::dict out;
    out["R_max_cm"] = std::abs(u.R);
    out["H_total_cm"] = u.H;
    out["handle_r_frac"] = u.handle;
    py::array_t<double> sd(FULL_LEN);
    for (int i = 0; i < FULL_LEN; ++i) sd.mutable_at(i) = u.deltas_full[i];
    out["shape_deltas"] = sd;
    py::list positions;
    for (int i = 0; i < ns; ++i) positions.append(py::make_tuple(params[(size_t)(3 + N_SHAPE + 2 * i)], params[(size_t)(3 + N_SHAPE + 2 * i + 1)]));
    out["positions_cm"] = positions;
    py::array_t<double> rs((py::ssize_t)m_data);
    double ss = 0.0;
    for (size_t k = 0; k < m_data; ++k) { rs.mutable_at((py::ssize_t)k) = resid[k]; ss += resid[k] * resid[k]; }
    out["residuals_px"] = rs;
    out["residual_rms_px"] = std::sqrt(ss / (double)m_data);
    return out;
}


// ------------------------------------------------------------
// ALFA-AARIVIIVA (Python: profiili.py, A.ALFA_*)
//   alpha_observation_cpp: varjonvaimennus (pikselikohtainen, sama kaava kuin suppress_shadow_background) + granittimaski (TARKKA
//                          sigma=25 -sumennus, avaus 5x5, sulkeminen 3x3 - ei GRANITE_DOWN-approksimaatiota) rajatulla alueella
//                          (kiven ymparilla +-160 px) -> alfa-kartta + laatumitat (suhde, tumma alue).
//   alpha_contour_cpp:     4x bicubic ylinaytteistys, tasa-arvokayra, kuminauha, rajaus graniittimaskiin, liukulukukoordinaatit.
// ------------------------------------------------------------
static double medianOfFloats(std::vector<float>& v)
{
    const size_t n = v.size();
    if (n == 0) return 0.0;
    const size_t m = n / 2;
    std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)m, v.end());
    const float hi = v[m];
    if (n % 2 == 1) return (double)hi;
    const float lo = *std::max_element(v.begin(), v.begin() + (std::ptrdiff_t)m);
    return 0.5 * ((double)lo + (double)hi);
}

static py::object alpha_observation_cpp(
    ArrU8 frame, ArrU8 reference, ArrD gains, ArrD biases,
    int cx, int cy, double diff_threshold, double v_drop_min, double v_drop_max, int ice_s_max, int ice_v_min,
    int crop_half, int a_dilate, int dark_v_max)
{
    auto fb = frame.request(); auto rb = reference.request();
    if (fb.ndim != 3 || fb.shape[2] != 3 || rb.ndim != 3 || rb.shape[2] != 3 || fb.shape[0] != rb.shape[0] || fb.shape[1] != rb.shape[1])
        throw std::runtime_error("alpha_observation_cpp: frame/reference must be HxWx3 uint8 of equal size");
    const int H = (int)fb.shape[0], W = (int)fb.shape[1];
    cv::Mat fullF(H, W, CV_8UC3, (void*)fb.ptr), fullR(H, W, CV_8UC3, (void*)rb.ptr);
    double g[3] = {1, 1, 1}, bi[3] = {0, 0, 0};
    if (gains.size() >= 3 && biases.size() >= 3) { auto gu = gains.unchecked<1>(); auto bu = biases.unchecked<1>(); for (int c = 0; c < 3; ++c) { g[c] = gu(c); bi[c] = bu(c); } }
    uint8_t lut[3][256];
    for (int c = 0; c < 3; ++c)
        for (int v = 0; v < 256; ++v) {
            float x = (float)v * (float)g[c] + (float)bi[c];
            x = std::min(std::max(x, 0.f), 255.f);
            lut[c][v] = (uint8_t)x;
        }

    // --- aluerajaus: kiven ymparilla +-160 px (sumennuksen reunavaikutus hyvin pieni, tausta vaimennettu valkoiseksi)
    const int RM = 160;
    const int rx0 = std::max(0, cx - RM), ry0 = std::max(0, cy - RM), rx1 = std::min(W, cx + RM), ry1 = std::min(H, cy + RM);
    if (rx1 - rx0 < 16 || ry1 - ry0 < 16) return py::none();
    const cv::Rect region(rx0, ry0, rx1 - rx0, ry1 - ry0);
    cv::Mat fR = fullF(region), rR = fullR(region);
    const int h = region.height, w = region.width;

    // varjonvaimennus (sama kaava kuin suppress_shadow_background) sekä korjattu ruutu (valotasapaino)
    static int sdiv[256]; static bool sdiv_init = false;
    if (!sdiv_init) { sdiv[0] = 0; for (int i = 1; i < 256; ++i) sdiv[i] = (int)std::lround((255 << 12) / (1.0 * i)); sdiv_init = true; }
    cv::Mat sup(h, w, CV_8UC3), corr(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        const uint8_t* sp = fR.ptr<uint8_t>(y); const uint8_t* rp = rR.ptr<uint8_t>(y);
        uint8_t* dp = sup.ptr<uint8_t>(y); uint8_t* cp = corr.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x, sp += 3, rp += 3, dp += 3, cp += 3) {
            const int b = lut[0][sp[0]], gg = lut[1][sp[1]], r = lut[2][sp[2]];
            cp[0] = (uint8_t)b; cp[1] = (uint8_t)gg; cp[2] = (uint8_t)r;
            const int db = std::abs(b - rp[0]), dg = std::abs(gg - rp[1]), dr = std::abs(r - rp[2]);
            const int gray = (db * 1868 + dg * 9617 + dr * 4899 + 8192) >> 14;
            bool bg = (double)gray < diff_threshold;
            const int vmax = std::max(b, std::max(gg, r)), vmin = std::min(b, std::min(gg, r));
            if (!bg) {
                const int rv = std::max((int)rp[0], std::max((int)rp[1], (int)rp[2]));
                const double vdrop = (double)(rv - vmax);
                if (vdrop > v_drop_min && vdrop < v_drop_max) bg = true;
                else { const int sat = ((vmax - vmin) * sdiv[vmax] + 2048) >> 12; if (sat < ice_s_max && vmax > ice_v_min) bg = true; }
            }
            if (bg && !colorGateOk(b, gg, r, rp, vmax, vmin)) bg = false;
            if (bg) { dp[0] = dp[1] = dp[2] = 255; } else { dp[0] = (uint8_t)b; dp[1] = (uint8_t)gg; dp[2] = (uint8_t)r; }
        }
    }

    // granittimaski (tarkka): sat < 60 & tummuus > 15, avaus 5x5, sulkeminen 3x3
    cv::Mat gray8, gray_f, bgb, darkness;
    cv::cvtColor(sup, gray8, cv::COLOR_BGR2GRAY);
    gray8.convertTo(gray_f, CV_32F);
    cv::GaussianBlur(gray_f, bgb, cv::Size(0, 0), STONE_DARKNESS_SIGMA);
    darkness = bgb - gray_f;
    cv::Mat satc = satU8FromBgr(sup);
    cv::Mat gm(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        const uint8_t* sp = satc.ptr<uint8_t>(y); const float* dk = darkness.ptr<float>(y); uint8_t* mp = gm.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) mp[x] = (sp[x] < STONE_MAX_SATURATION && dk[x] > (float)STONE_MIN_DARKNESS) ? 255 : 0;
    }
    cv::morphologyEx(gm, gm, cv::MORPH_OPEN, cv::Mat::ones(5, 5, CV_8U));
    cv::morphologyEx(gm, gm, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U));

    // kiveä lahin yhtenainen alue (pinta-ala >= 60)
    const double lcx0 = (double)(cx - rx0), lcy0 = (double)(cy - ry0);
    cv::Mat lab, cst, cen;
    const int ncc = cv::connectedComponentsWithStats(gm > 0, lab, cst, cen, 8, CV_32S);
    int best = -1; double bestd = 0;
    for (int i = 1; i < ncc; ++i) {
        if (cst.at<int>(i, cv::CC_STAT_AREA) < 60) continue;
        const double d = std::hypot(cen.at<double>(i, 0) - lcx0, cen.at<double>(i, 1) - lcy0);
        if (best < 0 || d < bestd) { best = i; bestd = d; }
    }
    if (best < 0) return py::none();
    cv::Mat A = (lab == best);        // 0/255
    A.convertTo(A, CV_8U, 1.0 / 255.0);

    // --- alfa-kartta kiven ymparistossa (crop_half)
    const int x0 = std::max(0, cx - crop_half), y0 = std::max(0, cy - crop_half), x1 = std::min(W, cx + crop_half), y1 = std::min(H, cy + crop_half);
    const cv::Rect crop(x0 - rx0, y0 - ry0, x1 - x0, y1 - y0);
    const int ch = crop.height, cw = crop.width;
    cv::Mat Ac = A(crop).clone();
    cv::Mat vf(ch, cw, CV_32F), k(ch, cw, CV_32F);
    for (int y = 0; y < ch; ++y) {
        const uint8_t* cp = corr(crop).ptr<uint8_t>(y); const uint8_t* rp = rR(crop).ptr<uint8_t>(y);
        float* vfp = vf.ptr<float>(y); float* kp = k.ptr<float>(y);
        for (int x = 0; x < cw; ++x, cp += 3, rp += 3) {
            const float v = (float)std::max(cp[0], std::max(cp[1], cp[2]));
            const float vr = (float)std::max(rp[0], std::max(rp[1], rp[2]));
            vfp[x] = v; kp[x] = v / std::max(vr, 1.0f);
        }
    }
    cv::Mat core; cv::erode(Ac, core, cv::Mat::ones(3, 3, CV_8U));
    if (cv::countNonZero(core) < 5) core = Ac;
    std::vector<float> tmp;
    for (int y = 0; y < ch; ++y) { const uint8_t* cr = core.ptr<uint8_t>(y); const float* kp = k.ptr<float>(y); for (int x = 0; x < cw; ++x) if (cr[x]) tmp.push_back(kp[x]); }
    const double kg = medianOfFloats(tmp);
    cv::Mat excl, nearm;
    cv::dilate(Ac, excl, cv::Mat::ones(15, 15, CV_8U)); cv::dilate(Ac, nearm, cv::Mat::ones(45, 45, CV_8U));
    cv::Mat wgt(ch, cw, CV_32F, cv::Scalar(0)); int n_bg = 0;
    std::vector<float> vbg;
    for (int y = 0; y < ch; ++y) {
        const uint8_t* ex = excl.ptr<uint8_t>(y); const uint8_t* nr = nearm.ptr<uint8_t>(y); const float* kp = k.ptr<float>(y);
        const float* vfp = vf.ptr<float>(y); float* wp = wgt.ptr<float>(y);
        for (int x = 0; x < cw; ++x) {
            if (ex[x] == 0 && nr[x] > 0) {
                if (kp[x] > 0.6f) { ++n_bg; vbg.push_back(vfp[x]); }
                if (kp[x] > (float)(kg + 0.25)) wp[x] = 1.0f;
            }
        }
    }
    if (n_bg < 50) return py::none();
    cv::Mat kw, num, den;
    cv::multiply(k, wgt, kw);
    cv::GaussianBlur(kw, num, cv::Size(0, 0), 5); cv::GaussianBlur(wgt, den, cv::Size(0, 0), 5);
    cv::Mat alpha(ch, cw, CV_32F);
    const float klo = (float)(kg + 0.25);
    for (int y = 0; y < ch; ++y) {
        const float* nm = num.ptr<float>(y); const float* dn = den.ptr<float>(y); const float* kp = k.ptr<float>(y); float* ap = alpha.ptr<float>(y);
        for (int x = 0; x < cw; ++x) {
            float ks = (dn[x] > 1e-3f) ? nm[x] / std::max(dn[x], 1e-3f) : 1.0f;
            ks = std::min(std::max(ks, klo), 1.05f);
            float a = (ks - kp[x]) / std::max(ks - (float)kg, 0.15f);
            ap[x] = std::min(std::max(a, 0.0f), 1.0f);
        }
    }
    // laatumitat
    std::vector<float> vcore;
    for (int y = 0; y < ch; ++y) { const uint8_t* cr = core.ptr<uint8_t>(y); const float* vfp = vf.ptr<float>(y); for (int x = 0; x < cw; ++x) if (cr[x]) vcore.push_back(vfp[x]); }
    const double ratio = medianOfFloats(vcore) / std::max(medianOfFloats(vbg), 1.0);
    cv::Mat d31, d9;
    cv::dilate(Ac, d31, cv::Mat::ones(31, 31, CV_8U)); cv::dilate(Ac, d9, cv::Mat::ones(9, 9, CV_8U));
    int dark_px = 0;
    for (int y = 0; y < ch; ++y) {
        const uint8_t* a31 = d31.ptr<uint8_t>(y); const uint8_t* a9 = d9.ptr<uint8_t>(y); const float* vfp = vf.ptr<float>(y);
        for (int x = 0; x < cw; ++x) if (a31[x] > 0 && a9[x] == 0 && vfp[x] < (float)dark_v_max) ++dark_px;
    }
    cv::Mat adil; cv::dilate(Ac, adil, cv::Mat::ones(a_dilate, a_dilate, CV_8U));

    py::array_t<float> alpha_out({ (py::ssize_t)ch, (py::ssize_t)cw });
    std::memcpy(alpha_out.mutable_data(), alpha.ptr<float>(0), (size_t)ch * (size_t)cw * sizeof(float));
    py::array_t<uint8_t> adil_out({ (py::ssize_t)ch, (py::ssize_t)cw });
    std::memcpy(adil_out.mutable_data(), adil.ptr<uint8_t>(0), (size_t)ch * (size_t)cw);
    py::dict d;
    d["alpha"] = alpha_out; d["adil"] = adil_out;
    d["origin"] = py::make_tuple(x0, y0); d["center"] = py::make_tuple(cx - x0, cy - y0);
    d["ratio"] = ratio; d["dark_px"] = dark_px; d["a_area"] = cv::countNonZero(Ac);
    return d;
}

static py::tuple alpha_contour_cpp(
    ArrF alpha_arr,
    ArrU8 adil_arr,
    double lcx_native, double lcy_native, int ox, int oy, double level, int up)
{
    auto ab = alpha_arr.request(); auto db = adil_arr.request();
    const int ch = (int)ab.shape[0], cw = (int)ab.shape[1];
    cv::Mat alpha(ch, cw, CV_32F, (void*)ab.ptr), adil(ch, cw, CV_8U, (void*)db.ptr);
    cv::Mat au, adu;
    cv::resize(alpha, au, cv::Size(), up, up, cv::INTER_CUBIC);
    cv::resize(adil, adu, cv::Size(), up, up, cv::INTER_NEAREST);
    const double lcx = lcx_native * up, lcy = lcy_native * up;
    cv::Mat reg(au.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < au.rows; ++y) {
        if (!(std::abs((double)y - lcy) < 50.0 * up)) continue;
        const float* ap = au.ptr<float>(y); uint8_t* rp = reg.ptr<uint8_t>(y);
        for (int x = 0; x < au.cols; ++x)
            if (ap[x] > (float)level && std::abs((double)x - lcx) < 45.0 * up) rp[x] = 1;
    }
    cv::Mat lab, cst, cen;
    const int n = cv::connectedComponentsWithStats(reg, lab, cst, cen, 8, CV_32S);
    std::vector<int> keep;
    for (int i = 1; i < n; ++i)
        if (cst.at<int>(i, cv::CC_STAT_AREA) >= 6 * up * up && std::hypot(cen.at<double>(i, 0) - lcx, cen.at<double>(i, 1) - lcy) < 28.0 * up) keep.push_back(i);
    if (keep.empty()) return py::make_tuple(py::none(), 0.0);
    cv::Mat comp(au.size(), CV_8U, cv::Scalar(0));
    {
        std::vector<uint8_t> is_keep((size_t)n, 0); for (int i : keep) is_keep[(size_t)i] = 1;
        for (int y = 0; y < lab.rows; ++y) { const int* lp = lab.ptr<int>(y); uint8_t* cp = comp.ptr<uint8_t>(y); for (int x = 0; x < lab.cols; ++x) if (lp[x] > 0 && is_keep[(size_t)lp[x]]) cp[x] = 1; }
    }
    std::vector<cv::Point> pts; cv::findNonZero(comp, pts);
    std::vector<cv::Point> hull; cv::convexHull(pts, hull);
    cv::Mat hm(au.size(), CV_8U, cv::Scalar(0));
    cv::fillConvexPoly(hm, hull, cv::Scalar(1));
    cv::Mat fin(au.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < fin.rows; ++y) { const uint8_t* hp = hm.ptr<uint8_t>(y); const uint8_t* ap = adu.ptr<uint8_t>(y); uint8_t* fp = fin.ptr<uint8_t>(y); for (int x = 0; x < fin.cols; ++x) fp[x] = (hp[x] > 0 && ap[x] > 0) ? 1 : 0; }
    std::vector<std::vector<cv::Point>> cs;
    cv::Mat fin_c = fin.clone();
    cv::findContours(fin_c, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (cs.empty()) return py::make_tuple(py::none(), 0.0);
    size_t bi = 0; double ba = -1;
    for (size_t i = 0; i < cs.size(); ++i) { const double a = cv::contourArea(cs[i]); if (a > ba) { ba = a; bi = i; } }
    const auto& c = cs[bi];
    py::array_t<double> out({ (py::ssize_t)c.size(), (py::ssize_t)2 });
    double* o = out.mutable_data();
    for (size_t i = 0; i < c.size(); ++i) {
        o[2 * i] = ((double)c[i].x + 0.5) / up - 0.5 + (double)ox;
        o[2 * i + 1] = ((double)c[i].y + 0.5) / up - 0.5 + (double)oy;
    }
    return py::make_tuple(out, (double)cv::countNonZero(fin) / (double)(up * up));
}
