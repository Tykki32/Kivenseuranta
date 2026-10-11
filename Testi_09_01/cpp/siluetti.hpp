// ============================================================
// siluetti.hpp - kiven paikan siluettitarkennus graniittimaskista (Python: siluetti.py).
// Kiven 3D-mallin siluettia (kupera peite miinus kahvan lovi) siirretaan kuvatasossa (+-max_shift_px, 0.5 px askel)
// ja valitaan siirto, jolla sisapuoli palkitaan, ylitulo (paitsi kiven ylapuolella = kuvassa oikealla) rangaistaan
// siluetin ymparisella kaistalla (band_px), ja massakeskipisteiden keskitys antaa pienen bonuksen. Paras siirto
// muutetaan maatasoon (X, Y) Jakobiaanilla. HAKU kutsuu silhouette_refine_cpp:ta; SEURANTA laskee saman
// kivisaikeissa (set_seuranta_silhouette).
// ============================================================
#pragma once
#include "sovitus.hpp"

struct SilResult {
    bool ok = false;
    double X = 0.0, Y = 0.0;
    double score0 = 0.0, score1 = 0.0, inside0 = 0.0, inside1 = 0.0, shift_px = 0.0, shift_cm = 0.0;
};

// SEURANNAN siluettiasetus (set_seuranta_silhouette): track_stones_batch tarkentaa jokaisen loydetyn kiven heti
// paivityksen jalkeen samassa kivisaikeessa.
struct SeurantaSilCfg {
    bool on = false;
    std::vector<cv::Point3d> body;
    cv::Matx33d K, R; cv::Vec3d t;
    double R_max = 0, H_total = 0, lovi = 0, w = 0, lam = 0, sigfrac = 0;
    int max_shift = 0, half = 0, margin = 0, open_size = 3, band = 3;
};
static SeurantaSilCfg g_sil_cfg;
static std::mutex g_sil_cfg_mx;

static py::tuple silResultToTuple(const SilResult& r)
{
    py::dict d;
    d["ok"] = r.ok;
    if (r.ok) {
        d["score0"] = r.score0; d["score1"] = r.score1; d["inside0"] = r.inside0; d["inside1"] = r.inside1;
        d["shift_px"] = r.shift_px; d["shift_cm"] = r.shift_cm;
    }
    return py::make_tuple(r.X, r.Y, d);
}

static void silFillPoly(cv::Mat& m, const std::vector<cv::Point2f>& poly, int ox, int oy, int up)
{
    std::vector<cv::Point> pts; pts.reserve(poly.size());
    for (const auto& p : poly) {
        const double x = ((double)p.x - ox + 0.5) * up - 0.5, y = ((double)p.y - oy + 0.5) * up - 0.5;
        pts.emplace_back((int)std::nearbyint(x), (int)std::nearbyint(y));      // np.round (puolet parilliseen)
    }
    cv::fillPoly(m, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(1));
}

// Laskentaydin ilman Python-olioita (ajetaan usealle kivelle rinnan GIL vapaana)
static SilResult silhouetteRefineCore(
    const cv::Mat& img, const std::vector<cv::Point3d>& body,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double R_max_cm, double H_total_cm, double handle_r_frac, double X0, double Y0,
    double w_leak, double lam, double sigma_frac, int max_shift_px, int half, int margin, int open_size, int band_px)
{
    const int H = img.rows, W = img.cols;
    const int UPS = 2;
    const int S = max_shift_px * UPS;
    auto fail = [&]() { SilResult r; r.ok = false; r.X = X0; r.Y = Y0; return r; };

    auto proj = [&](double x, double y, double z) {
        cv::Vec3d pc = R * cv::Vec3d(x, y, z) + t; cv::Vec3d pi = K * pc;
        return cv::Point2d(pi[0] / pi[2], pi[1] / pi[2]);
    };
    const cv::Point2d c0 = proj(X0, Y0, H_total_cm / 2.0);
    const int cxi = (int)std::nearbyint(c0.x), cyi = (int)std::nearbyint(c0.y);
    const int ox = cxi - half, oy = cyi - half;
    if (ox < 0 || oy < 0 || ox + 2 * half > W || oy + 2 * half > H) return fail();
    const int x0c = std::max(0, ox - margin), y0c = std::max(0, oy - margin);
    const int x1c = std::min(W, ox + 2 * half + margin), y1c = std::min(H, oy + 2 * half + margin);
    if (x1c - x0c < 40 || y1c - y0c < 40) return fail();

    cv::Mat gm = createGraniteMask(img(cv::Rect(x0c, y0c, x1c - x0c, y1c - y0c)).clone(), nullptr, open_size);
    cv::Mat maskw = gm(cv::Rect(ox - x0c, oy - y0c, 2 * half, 2 * half)) > 0;
    if (cv::countNonZero(maskw) == 0) return fail();
    cv::Mat m2u, m2;
    cv::resize(maskw, m2u, cv::Size(), UPS, UPS, cv::INTER_NEAREST);
    m2u.convertTo(m2, CV_32F, 1.0 / 255.0);
    const int sh = m2.rows, sw = m2.cols;

    auto hull = predictedHull(body, X0, Y0, K, R, t);
    auto notch = predictedNotchHull(X0, Y0, R_max_cm, H_total_cm, handle_r_frac, K, R, t, 28);
    if (hull.size() < 3 || notch.size() < 3) return fail();
    cv::Mat mh8 = cv::Mat::zeros(sh, sw, CV_8U), mn8 = cv::Mat::zeros(sh, sw, CV_8U);
    silFillPoly(mh8, hull, ox, oy, UPS); silFillPoly(mn8, notch, ox, oy, UPS);
    // --- siluetin rivit: kupera peite = yksi vali / rivi, siluetti (peite miinus lovi) = 0..2 valia / rivi (run-length)
    struct Run { int x1, x2; };
    std::vector<int> h1(sh, -1), h2(sh, -1);
    std::vector<std::vector<Run>> sil_runs((size_t)sh);
    double a = 0.0, sx_sum = 0.0, sy_sum = 0.0;
    for (int y = 0; y < sh; ++y) {
        const uint8_t* hp = mh8.ptr<uint8_t>(y); const uint8_t* np_ = mn8.ptr<uint8_t>(y);
        for (int x = 0; x < sw; ++x) if (hp[x]) { if (h1[y] < 0) h1[y] = x; h2[y] = x; }
        int run_start = -1;
        for (int x = 0; x <= sw; ++x) {
            const bool insil = (x < sw) && hp[x] && !np_[x];
            if (insil) { if (run_start < 0) run_start = x; a += 1.0; sx_sum += x; sy_sum += y; }
            else if (run_start >= 0) { sil_runs[(size_t)y].push_back({ run_start, x - 1 }); run_start = -1; }
        }
    }
    if (a < 10) return fail();
    // --- ylitysrangaistus vain siluetin ymparilla olevalla kaistalla (band_px, kuvapikseleina): kupera peite laajennettuna (nelio) -> rivikohtainen vali [d1, d2]
    std::vector<int> d1(sh, -1), d2(sh, -1);
    {
        cv::Mat dil; const int kb = 2 * band_px * UPS + 1;
        cv::dilate(mh8, dil, cv::Mat::ones(kb, kb, CV_8U));
        for (int y = 0; y < sh; ++y) { const uint8_t* dp = dil.ptr<uint8_t>(y); for (int x = 0; x < sw; ++x) if (dp[x]) { if (d1[y] < 0) d1[y] = x; d2[y] = x; } }
    }
    const double sil_cx = sx_sum / a, sil_cy = sy_sum / a;
    // --- maski (nollatayte S reunoilla) + rivikohtaiset kumulatiiviset summat: PR = sum M, PX = sum x*M
    const int BH = sh + 2 * S, BW = sw + 2 * S;
    std::vector<double> PR((size_t)BH * (BW + 1), 0.0), PX((size_t)BH * (BW + 1), 0.0);
    for (int r = 0; r < BH; ++r) {
        const bool in_m = (r >= S && r < S + sh);
        double acc = 0.0, accx = 0.0, rowsum_prev = 0.0;
        for (int x = 0; x < BW; ++x) {
            const double v = (in_m && x >= S && x < S + sw) ? (double)m2.at<float>(r - S, x - S) : 0.0;
            PR[(size_t)r * (BW + 1) + x] = acc; PX[(size_t)r * (BW + 1) + x] = accx;
            acc += v; accx += v * (double)x;
        }
        PR[(size_t)r * (BW + 1) + BW] = acc; PX[(size_t)r * (BW + 1) + BW] = accx;
        (void)rowsum_prev;
    }
    auto rowint = [&](const std::vector<double>& P, int r, int xa, int xb) { return P[(size_t)r * (BW + 1) + xb] - P[(size_t)r * (BW + 1) + xa]; };   // [xa, xb)
    const double sig = sigma_frac * std::sqrt(a / (double)(UPS * UPS));
    const int NS = 2 * S + 1;
    double best = -1e300; int bix = S, biy = S;
    std::vector<double> ins_map((size_t)NS * NS), score_map((size_t)NS * NS);
    for (int iy = 0; iy < NS; ++iy) {
        for (int ix = 0; ix < NS; ++ix) {
            double ins = 0.0, cnt = 0.0, mx = 0.0, my = 0.0, fr = 0.0, band_tot = 0.0;
            for (int y = 0; y < sh; ++y) {
                if (d1[y] >= 0) band_tot += rowint(PR, y + iy, d1[y] + ix, d2[y] + 1 + ix);
                if (h1[y] < 0) continue;
                const int r = y + iy;
                const double ci = rowint(PR, r, h1[y] + ix, h2[y] + 1 + ix);
                cnt += ci; my += (double)y * ci;
                mx += rowint(PX, r, h1[y] + ix, h2[y] + 1 + ix) - (double)ix * ci;
                for (const Run& ru : sil_runs[(size_t)y]) ins += rowint(PR, r, ru.x1 + ix, ru.x2 + 1 + ix);
                if (!sil_runs[(size_t)y].empty()) fr += rowint(PR, r, h2[y] + 1 + ix, d2[y] + 1 + ix);   // vapaa alue (kaistan sisalla): siluetin oikealla puolella samalla rivilla
            }
            const double inside = ins / a, leak = (band_tot - cnt - fr) / a;
            double bonus = 0.0;
            if (cnt > 0.5) {
                const double d = std::hypot(mx / cnt - sil_cx, my / cnt - sil_cy) / (double)UPS;
                bonus = lam * std::exp(-(d / sig) * (d / sig));
            }
            const double sc = inside - w_leak * leak + bonus;
            ins_map[(size_t)iy * NS + ix] = inside; score_map[(size_t)iy * NS + ix] = sc;
            if (sc > best + 1e-7) { best = sc; bix = ix; biy = iy; }
            else if (sc >= best - 1e-7) {           // tasapeli: valitaan pienin siirto (vakaa, ei riipu lukutarkkuudesta)
                const int dd = (ix - S) * (ix - S) + (iy - S) * (iy - S), db = (bix - S) * (bix - S) + (biy - S) * (biy - S);
                if (dd < db) { best = std::max(best, sc); bix = ix; biy = iy; }
            }
        }
    }
    const double du = (double)(bix - S) / UPS, dv = (double)(biy - S) / UPS;
    const double s0 = score_map[(size_t)S * NS + S], s1 = score_map[(size_t)biy * NS + bix];
    const double in0 = ins_map[(size_t)S * NS + S], in1 = ins_map[(size_t)biy * NS + bix];
    SilResult res; res.ok = true; res.score0 = s0; res.score1 = s1; res.inside0 = in0; res.inside1 = in1;
    if (du == 0.0 && dv == 0.0) { res.shift_px = 0.0; res.shift_cm = 0.0; res.X = X0; res.Y = Y0; return res; }
    const cv::Point2d p0 = c0, px = proj(X0 + 1.0, Y0, H_total_cm / 2.0), py_ = proj(X0, Y0 + 1.0, H_total_cm / 2.0);
    const double j00 = px.x - p0.x, j01 = py_.x - p0.x, j10 = px.y - p0.y, j11 = py_.y - p0.y;
    const double det = j00 * j11 - j01 * j10;
    if (std::abs(det) < 1e-12) return fail();
    const double dX = (du * j11 - j01 * dv) / det, dY = (j00 * dv - j10 * du) / det;
    res.shift_px = std::hypot(du, dv); res.shift_cm = std::hypot(dX, dY);
    res.X = X0 + dX; res.Y = Y0 + dY;
    return res;
}


// HAKU: yhden ehdokkaan tarkennus -> (X, Y, info)
static py::tuple silhouette_refine_cpp(
    ArrU8 frame, ArrD local_pts_body_arr, ArrD K_arr, ArrD R_arr, ArrD t_arr,
    double R_max_cm, double H_total_cm, double handle_r_frac, double X0, double Y0,
    double w_leak, double lam, double sigma_frac, int max_shift_px, int half, int margin, int open_size, int band_px)
{
    cv::Mat img = bgrView(frame);
    if (img.empty()) throw std::runtime_error("silhouette_refine_cpp: frame must be HxWx3 uint8");
    auto body = parsePts3(local_pts_body_arr);
    auto K = parseMat33(K_arr); auto R = parseMat33(R_arr); auto t = parseVec3(t_arr);
    SilResult r;
    {
        py::gil_scoped_release release;
        r = silhouetteRefineCore(img, body, K, R, t, R_max_cm, H_total_cm, handle_r_frac, X0, Y0,
                                 w_leak, lam, sigma_frac, max_shift_px, half, margin, open_size, band_px);
    }
    return silResultToTuple(r);
}
