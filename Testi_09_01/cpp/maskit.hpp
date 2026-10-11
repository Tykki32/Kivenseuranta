// ============================================================
// maskit.hpp - kuvamaskit: saturaatio, graniittimaski (kivi = matala kylläisyys + paikallista taustaa tummempi),
// variportti, staattisen taustan vaimennus, etualamaski, ROI-rajaus ja bilineaarinen naytteistys.
// ============================================================
#pragma once
#include "geometria.hpp"

// ------------------------------------------------------------
// SATURAATIO (OpenCV:n uint8-kaava: sdiv[i] = round(255 * 4096 / i), s = ((max - min) * sdiv[max] + 2048) >> 12).
// cvtColor(BGR2HSV) laskee saman SIMD:lla; satSimdOk() tarkistaa kerran, etta tulos on sama ja etta cvtColor on
// talla koneella nopeampi (GCC vektoroi skalaarisilmukan, MSVC ei valttamatta).
// ------------------------------------------------------------
static double g_sat_us[2] = {0.0, 0.0};   // kaynnistysmittaus 300x300: [skalaari, cvtColor] mikrosekuntia

static cv::Mat satU8FromBgrScalar(const cv::Mat& bgr)
{
    static int sdiv[256];
    static const bool init = []() {
        sdiv[0] = 0;
        for (int i = 1; i < 256; ++i) sdiv[i] = (int)std::lround((255 << 12) / (1.0 * i));
        return true;
    }();
    (void)init;
    cv::Mat sat(bgr.rows, bgr.cols, CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y) {
        const uint8_t* p = bgr.ptr<uint8_t>(y);
        uint8_t* o = sat.ptr<uint8_t>(y);
        for (int x = 0; x < bgr.cols; ++x, p += 3) {
            const int b = p[0], g = p[1], r = p[2];
            const int mx = std::max(b, std::max(g, r)), mn = std::min(b, std::min(g, r));
            o[x] = (uint8_t)(((mx - mn) * sdiv[mx] + 2048) >> 12);
        }
    }
    return sat;
}

static bool satSimdOk()
{
    static const bool ok = []() {
        cv::Mat img(257, 263, CV_8UC3);
        cv::RNG rng(12345);
        rng.fill(img, cv::RNG::UNIFORM, 0, 256);
        for (int i = 0; i < 256; ++i)          // kaikki maksimiarvot ja harmaat (diff = 0) mukaan
            img.at<cv::Vec3b>(i, 0) = cv::Vec3b((uchar)i, (uchar)i, (uchar)(i / 2)), img.at<cv::Vec3b>(i, 1) = cv::Vec3b((uchar)i, (uchar)i, (uchar)i);
        cv::Mat hsv, a;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::extractChannel(hsv, a, 1);
        cv::Mat b = satU8FromBgrScalar(img);
        if (cv::countNonZero(a != b) != 0)
            return false;
        cv::Mat big(300, 300, CV_8UC3);
        rng.fill(big, cv::RNG::UNIFORM, 0, 256);
        auto best = [&](auto fn) {
            double bmin = 1e9;
            for (int k = 0; k < 5; ++k) {
                auto t0 = std::chrono::steady_clock::now();
                fn();
                bmin = std::min(bmin, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            return bmin;
        };
        const double t_simd = best([&]() { cv::Mat h, o; cv::cvtColor(big, h, cv::COLOR_BGR2HSV); cv::extractChannel(h, o, 1); });
        const double t_scalar = best([&]() { cv::Mat o = satU8FromBgrScalar(big); });
        g_sat_us[0] = t_scalar; g_sat_us[1] = t_simd;
        return t_simd < t_scalar;
    }();
    return ok;
}

static cv::Mat satU8FromBgr(const cv::Mat& bgr)
{
    if (satSimdOk()) {
        cv::Mat hsv, sat;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        cv::extractChannel(hsv, sat, 1);
        return sat;
    }
    return satU8FromBgrScalar(bgr);
}

static cv::Mat computeSat(const cv::Mat& frame_bgr)
{
    cv::Mat sat_f;
    satU8FromBgr(frame_bgr).convertTo(sat_f, CV_32F);
    return sat_f;
}

// ------------------------------------------------------------
// GRANIITTIMASKI: darkness = (harmaasavy sumennettuna sigma 25 px) - harmaasavy; maski = S < 60 ja darkness > 15,
// sitten avaus (5x5, siluettitarkennus 3x3) ja sulkeminen (3x3). Taustan sumennus lasketaan 4x pienennetylla kuvalla.
// ------------------------------------------------------------
static cv::Mat graniteDarkness(const cv::Mat& frame_bgr)
{
    cv::Mat gray, gray_f, bg;
    PT gpt;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray_f, CV_32F);
    profAdd(P_GM_CVT, gpt.lap());
    if (gray_f.cols >= 8 * GRANITE_DOWN && gray_f.rows >= 8 * GRANITE_DOWN) {
        cv::Mat small, small_bg;
        cv::resize(gray_f, small, cv::Size(), 1.0 / GRANITE_DOWN, 1.0 / GRANITE_DOWN, cv::INTER_AREA);
        cv::GaussianBlur(small, small_bg, cv::Size(0, 0), STONE_DARKNESS_SIGMA / GRANITE_DOWN);
        cv::resize(small_bg, bg, gray_f.size(), 0, 0, cv::INTER_LINEAR);
    } else {
        cv::GaussianBlur(gray_f, bg, cv::Size(0, 0), STONE_DARKNESS_SIGMA);
    }
    cv::Mat darkness = bg - gray_f;
    profAdd(P_GM_BLUR, gpt.lap());
    return darkness;
}

static cv::Mat createGraniteMaskFromParts(const cv::Mat& sat_ch, const cv::Mat& darkness, int open_override)
{
    PT gpt;
    cv::Mat mask, low_sat, dark_enough;
    cv::compare(sat_ch, cv::Scalar(STONE_MAX_SATURATION), low_sat, cv::CMP_LT);
    cv::compare(darkness, cv::Scalar((float)STONE_MIN_DARKNESS), dark_enough, cv::CMP_GT);
    cv::bitwise_and(low_sat, dark_enough, mask);
    profAdd(P_GM_LOOP, gpt.lap());

    const int open_k = open_override >= 0 ? open_override : GRANITE_OPEN;
    if (open_k > 0)
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(open_k, open_k, CV_8U));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat::ones(GRANITE_CLOSE, GRANITE_CLOSE, CV_8U));
    profAdd(P_GM_MORPH, gpt.lap());
    return mask;
}

// sat_in: jo laskettu saturaatio (sama kuva), muuten lasketaan
static cv::Mat createGraniteMask(const cv::Mat& frame_bgr, const cv::Mat* sat_in = nullptr, int open_override = -1)
{
    cv::Mat sat_ch = (sat_in != nullptr && !sat_in->empty()) ? *sat_in : satU8FromBgr(frame_bgr);
    return createGraniteMaskFromParts(sat_ch, graniteDarkness(frame_bgr), open_override);
}

// ------------------------------------------------------------
// VARIPORTTI (varjosietoinen taustanvaimennus, esikasittely.hpp ja GPU): pikseli saa olla taustaa vain jos moodikuvan
// pikseli on matalakylläinen (S < s_max) tai sama sävy (|dH| <= h_tol) ja ruudun pikseli kylläinen (S > s_max).
// Kylläinen mutta eri-sävyinen kohde (keltainen kahva) ei siis vaimene. s_max >= 256 = portti pois.
// Python: set_color_gate(A.VARIPORTTI_S_MAX, A.VARIPORTTI_H_TOL).
// ------------------------------------------------------------
static int g_gate_s_max = 256;
static int g_gate_h_tol = 5;

static inline int satFromMaxMin(int vmax, int vmin)
{
    static int tab[256]; static bool init = false;
    if (!init) { tab[0] = 0; for (int i = 1; i < 256; ++i) tab[i] = (int)std::lround((255 << 12) / (1.0 * i)); init = true; }
    return ((vmax - vmin) * tab[vmax] + 2048) >> 12;
}

static inline int hueOf(int b, int g, int r)   // OpenCV 8U HSV: H 0..179
{
    const int vmax = std::max(b, std::max(g, r)), vmin = std::min(b, std::min(g, r)), diff = vmax - vmin;
    if (diff == 0) return 0;
    const float hscale = 30.f / (float)diff;
    float h;
    if (vmax == r) h = (g - b) * hscale; else if (vmax == g) h = (b - r) * hscale + 60.f; else h = (r - g) * hscale + 120.f;
    if (h < 0) h += 180.f;
    return (int)std::lround(h);
}

static inline bool colorGateOk(int b, int g, int r, const uint8_t* rp, int vmax, int vmin)
{
    if (g_gate_s_max >= 256) return true;
    const int rmax = std::max((int)rp[0], std::max((int)rp[1], (int)rp[2])), rmin = std::min((int)rp[0], std::min((int)rp[1], (int)rp[2]));
    if (satFromMaxMin(rmax, rmin) < g_gate_s_max) return true;
    int dh = std::abs(hueOf(b, g, r) - hueOf((int)rp[0], (int)rp[1], (int)rp[2]));
    if (dh > 90) dh = 180 - dh;
    if (dh > g_gate_h_tol) return false;
    return satFromMaxMin(vmax, vmin) > g_gate_s_max;
}

// ------------------------------------------------------------
// STAATTISEN TAUSTAN VAIMENNUS: pikselit, joiden harmaasavyero moodikuvaan on alle kynnyksen, valkaistaan
// (255, 255, 255), jottei radan merkinta / mainos tule tunnistetuksi kiveksi.
// ------------------------------------------------------------
static cv::Mat suppressStaticBackground(const cv::Mat& frame_bgr, const cv::Mat& reference_bgr, double diff_threshold)
{
    if (reference_bgr.empty() || frame_bgr.rows != reference_bgr.rows || frame_bgr.cols != reference_bgr.cols
        || frame_bgr.type() != reference_bgr.type())
        return frame_bgr.clone();

    cv::Mat diff, diff_gray;
    cv::absdiff(frame_bgr, reference_bgr, diff);
    cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
    cv::Mat out = frame_bgr.clone();
    for (int r = 0; r < out.rows; ++r) {
        const uchar* dptr = diff_gray.ptr<uchar>(r);
        cv::Vec3b* optr = out.ptr<cv::Vec3b>(r);
        for (int c = 0; c < out.cols; ++c)
            if ((double)dptr[c] < diff_threshold)
                optr[c] = cv::Vec3b(255, 255, 255);
    }
    return out;
}

// ------------------------------------------------------------
// ETUALAMASKI jo vaimennetusta kuvasta: ei-valkaistu pikseli = etuala. SEURANTA yhdistaa sen (OR) graniittimaskiin:
// pelkka graniittimaski on kaukana harva (heikko paikallinen kontrasti), jolloin runkopisteita on vahan ja
// MAD-poikkeamien hylkays hyppii ruudusta toiseen. Etuala luetaan valkaisusta eika lasketa uudelleen absdiff:lla,
// koska kynnys 0 (kuva vaimennettu jo Pythonissa) tekisi jokaisesta pikselista etualaa.
// ------------------------------------------------------------
static cv::Mat createForegroundFromWhitened(const cv::Mat& crop_filtered_bgr)
{
    cv::Mat white, mask;
    cv::inRange(crop_filtered_bgr, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), white);
    cv::bitwise_not(white, mask);
    return mask;
}

// ------------------------------------------------------------
// ROI: hakualueen (+ kiven sade) kulmat jaatasolla ja kiven korkeudella projisoituna + marginaali, kuvan sisalla
// ------------------------------------------------------------
static cv::Rect trackRoiBounds(
    double X_center, double Y_center, double half_range_cm,
    double R_max_cm, double H_total_cm, int frame_w, int frame_h,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    int margin_px = MASK_ROI_MARGIN_PX)
{
    double reach = half_range_cm + R_max_cm;
    std::vector<cv::Point3d> corners;
    for (double zx : {X_center - reach, X_center + reach})
        for (double zy : {Y_center - reach, Y_center + reach})
            for (double zz : {0.0, H_total_cm})
                corners.push_back(cv::Point3d(zx, zy, zz));
    auto proj = project3d(K, R, t, corners);

    double minu = 1e18, maxu = -1e18, minv = 1e18, maxv = -1e18;
    for (auto& p : proj) {
        minu = std::min(minu, p.x);
        maxu = std::max(maxu, p.x);
        minv = std::min(minv, p.y);
        maxv = std::max(maxv, p.y);
    }
    int x0 = std::max((int)std::floor(minu) - margin_px, 0);
    int x1 = std::min((int)std::ceil(maxu) + margin_px, frame_w);
    int y0 = std::max((int)std::floor(minv) - margin_px, 0);
    int y1 = std::min((int)std::ceil(maxv) + margin_px, frame_h);
    if (x1 < x0) x1 = x0;
    if (y1 < y0) y1 = y0;
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

// ------------------------------------------------------------
// BILINEAARINEN NAYTE nollareunustetusta CV_32F-kuvasta ilman kopiota: koordinaatisto on kuin roi olisi reunustettu
// pl/pt nollapikselilla (cols/rows = reunustetun kuvan koko).
// ------------------------------------------------------------
struct PaddedView {
    const cv::Mat* roi;
    int pl, pt, cols, rows;
    inline float at(int y, int x) const
    {
        const int yy = y - pt, xx = x - pl;
        if (yy < 0 || xx < 0 || yy >= roi->rows || xx >= roi->cols) return 0.0f;
        return roi->at<float>(yy, xx);
    }
};

static bool bilinearSample(const PaddedView& channel, double x, double y, double& out)
{
    int w = channel.cols, h = channel.rows;
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h)
        return false;
    double fx = x - x0, fy = y - y0;
    double v00 = channel.at(y0, x0);
    double v10 = channel.at(y0, x1);
    double v01 = channel.at(y1, x0);
    double v11 = channel.at(y1, x1);
    out = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
    return true;
}
