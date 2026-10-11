// ============================================================
// meanshift.hpp - SEURANNAN mean-shift-paikannus.
//
// Kiven ennustettu hulli ja sen 1.1-kertainen laajennus jaetaan hullin painopisteen kautta kulkevilla vaaka- ja
// pystyviivoilla neljaan osaan. Maskin pikselit painotetaan (hullin sisalla inner_weight, pelkalla kehalla 1) ja
// puolikkaiden erotus
//   s_y = sum(w [maski] g(y - cy)) / sum(w),  s_x = sum(w [maski] g(x - cx)) / sum(w)
// kertoo siirtosuunnan (g = pehmea etumerkki, leveys tau). Signaali kyllastyy, joten se muunnetaan pikselisiirtymaksi
// d = gain * r * f^-1(s), missa f on yksikkoympyran vastefunktio (getResponseLUT, samat painot ja marginaali) ja
// r hullin puolileveys / -korkeus. Pikselisiirtyma muunnetaan (X, Y)-senteiksi paikallisella Jakobiaanilla ja
// toistetaan kunnes siirtyma < tol_px. Lopuksi paikallinen viimeistely peitto-osuudella (polish_step_cm, puolittuu
// kunnes < 0.75 cm).
// ============================================================
#pragma once
#include "geometria.hpp"

struct MeanShiftParams {
    int max_iter = 10;
    double gain = 1.0;
    double tol_px = 0.10;
    double inner_weight = 2.0;
    double margin_scale = HULL_MARGIN_SCALE;
    double tau = 0.0;                 // 0 = kova etumerkki
    double polish_step_cm = 0.0;      // 0 = ei viimeistelya
    // Yhdistelmahaun valinta (seuranta.hpp): pistemaara - rangaistukset
    double ens_back_tol_cm = 2.0;     // sallittu taaksepain-siirtyma ilman rangaistusta
    double ens_back_pen = 0.15;       // rangaistus / 10 cm taaksepain (yli toleranssin)
    double ens_pred_pen = 0.02;       // rangaistus / 10 cm poikkeamasta ennustetusta paikasta
};

static const double MS_RING_WEIGHT = 1.0;          // pelkan 1.1x-kehan paino
static const double MS_MIN_MASK_FRACTION = 0.02;   // maskia vahintaan tama osuus painosta, muuten lopetetaan
static const double MS_POLISH_MIN_STEP_CM = 0.75;

struct ResponseLUT {
    double dd = 0.01;
    std::vector<double> f;
    int imax = 0;
};

// Vastefunktio f(delta): yksikkoympyran signaali, kun malli on siirtynyt delta sateen monikertaa (valimuistissa)
static std::shared_ptr<const ResponseLUT> getResponseLUT(double w_in, double margin, double tau, double w_ring)
{
    static std::mutex mtx;
    static std::map<std::pair<long long, long long>, std::shared_ptr<const ResponseLUT>> cache;

    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_pair((long long)std::llround(w_in * 1000.0) * 100000LL + (long long)std::llround(tau * 1000.0),
                              (long long)std::llround(margin * 1000.0) * 100000LL + (long long)std::llround((w_ring + 50.0) * 1000.0));
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    auto lut = std::make_shared<ResponseLUT>();
    const int G = 241;
    const double ext = margin + 0.05;
    const double cell = 2.0 * ext / (double)(G - 1);

    std::vector<cv::Point3d> pts;   // (x, y, w)
    double wtot = 0.0;
    for (int i = 0; i < G; ++i) {
        double x = -ext + cell * i;
        for (int j = 0; j < G; ++j) {
            double y = -ext + cell * j;
            double r2 = x * x + y * y;
            double w = r2 <= 1.0 ? w_in : (r2 <= margin * margin ? w_ring : 0.0);
            if (w == 0.0)
                continue;
            pts.emplace_back(x, y, w);
            wtot += std::fabs(w);
        }
    }

    const int N = 221;
    lut->f.assign((size_t)N, 0.0);
    for (int k = 0; k < N; ++k) {
        double delta = lut->dd * k;
        double sum = 0.0;
        for (auto& p : pts) {
            double dy = p.y - delta;
            if (p.x * p.x + dy * dy <= 1.0) {
                double g = tau > 0.0 ? std::max(-1.0, std::min(1.0, p.y / tau))
                                     : (p.y > 0.0 ? 1.0 : (p.y < 0.0 ? -1.0 : 0.0));
                sum += p.z * g;
            }
        }
        lut->f[(size_t)k] = sum / wtot;
    }

    lut->imax = 0;
    for (int k = 1; k < N; ++k) {
        if (lut->f[(size_t)k] > lut->f[(size_t)lut->imax])
            lut->imax = k;
    }

    cache[key] = lut;
    return lut;
}

// f^-1: signaali -> normalisoitu siirtyma (r:n monikertoina), vain nousevalla haaralla.
static double responseInverse(const ResponseLUT& lut, double s)
{
    double a = std::fabs(s);
    double sign = s < 0.0 ? -1.0 : 1.0;
    double fmax = lut.f[(size_t)lut.imax];
    if (a >= fmax)
        return sign * lut.dd * lut.imax;

    int lo = 0, hi = lut.imax;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (lut.f[(size_t)mid] <= a)
            lo = mid;
        else
            hi = mid;
    }
    double f0 = lut.f[(size_t)lo], f1 = lut.f[(size_t)hi];
    double u = (f1 - f0) > 1e-12 ? (a - f0) / (f1 - f0) : 0.0;
    return sign * lut.dd * ((double)lo + u);
}

struct MsSignals {
    bool valid = false;
    double sx = 0.0, sy = 0.0;       // painotetut summat sign(x-cx)*w, sign(y-cy)*w (maski)
    double wtot = 0.0, wmask = 0.0;  // koko alueen paino / maskin peittama paino
    double rx = 0.0, ry = 0.0;       // hullin puolileveys / -korkeus (px)
    double cx = 0.0, cy = 0.0;       // hullin painopiste (crop-koordinaatit)
};

static MsSignals computeMsSignals(
    const cv::Mat& mask_crop, int off_x, int off_y,
    const std::vector<cv::Point2f>& hull, const MeanShiftParams& P)
{
    MsSignals out;

    std::vector<cv::Point2f> h(hull.size());
    for (size_t i = 0; i < hull.size(); ++i)
        h[i] = cv::Point2f(hull[i].x - (float)off_x, hull[i].y - (float)off_y);

    cv::Moments m = cv::moments(h);
    if (std::fabs(m.m00) < 4.0)
        return out;
    double cx = m.m10 / m.m00, cy = m.m01 / m.m00;

    auto extents = [](const std::vector<cv::Point2f>& poly) {
        float mnx = poly[0].x, mxx = poly[0].x, mny = poly[0].y, mxy = poly[0].y;
        for (auto& q : poly) {
            mnx = std::min(mnx, q.x); mxx = std::max(mxx, q.x);
            mny = std::min(mny, q.y); mxy = std::max(mxy, q.y);
        }
        return cv::Rect2f(mnx, mny, mxx - mnx, mxy - mny);
    };
    cv::Rect2f hb = extents(h);
    double rx = hb.width / 2.0, ry = hb.height / 2.0;
    if (rx < 2.0 || ry < 2.0)
        return out;

    std::vector<cv::Point2f> h2(h.size());
    for (size_t i = 0; i < h.size(); ++i)
        h2[i] = cv::Point2f((float)(cx + (h[i].x - cx) * P.margin_scale),
                            (float)(cy + (h[i].y - cy) * P.margin_scale));
    cv::Rect2f hb2 = extents(h2);

    int x0 = std::max(0, (int)std::floor(hb2.x) - 1);
    int y0 = std::max(0, (int)std::floor(hb2.y) - 1);
    int x1 = std::min(mask_crop.cols, (int)std::ceil(hb2.x + hb2.width) + 2);
    int y1 = std::min(mask_crop.rows, (int)std::ceil(hb2.y + hb2.height) + 2);
    if (x1 <= x0 || y1 <= y0)
        return out;

    // fillPoly sub-pikselitarkkuudella (shift=3 -> 1/8 px): tekee vasteesta
    // tasaisen siirtyman suhteen (ei kokonaislukupikselin porrastusta).
    const int SH = 3;
    const double SC = (double)(1 << SH);
    auto toFixed = [&](const std::vector<cv::Point2f>& poly) {
        std::vector<cv::Point> o(poly.size());
        for (size_t i = 0; i < poly.size(); ++i)
            o[i] = cv::Point((int)std::lround((poly[i].x - x0) * SC),
                             (int)std::lround((poly[i].y - y0) * SC));
        return o;
    };

    cv::Mat c1 = cv::Mat::zeros(y1 - y0, x1 - x0, CV_8UC1);
    cv::Mat c2 = cv::Mat::zeros(y1 - y0, x1 - x0, CV_8UC1);
    {
        std::vector<std::vector<cv::Point>> p1{toFixed(h)}, p2{toFixed(h2)};
        cv::fillPoly(c1, p1, cv::Scalar(1), cv::LINE_8, SH);
        cv::fillPoly(c2, p2, cv::Scalar(1), cv::LINE_8, SH);
    }

    double lcx = cx - x0, lcy = cy - y0;
    double wtot = 0.0, wmask = 0.0, sx = 0.0, sy = 0.0;
    const double tau = P.tau;
    auto gfun = [tau](double u) {
        if (tau > 0.0)
            return std::max(-1.0, std::min(1.0, u / tau));
        return u > 0.0 ? 1.0 : (u < 0.0 ? -1.0 : 0.0);
    };
    const double inv_rx = 1.0 / rx, inv_ry = 1.0 / ry;

    for (int r = 0; r < c1.rows; ++r) {
        const uchar* p1 = c1.ptr<uchar>(r);
        const uchar* p2 = c2.ptr<uchar>(r);
        const uchar* mp = mask_crop.ptr<uchar>(r + y0) + x0;
        double gy = gfun(((double)r - lcy) * inv_ry);
        for (int c = 0; c < c1.cols; ++c) {
            double w = p1[c] ? P.inner_weight : (p2[c] ? MS_RING_WEIGHT : 0.0);
            if (w == 0.0)
                continue;
            wtot += std::fabs(w);
            if (mp[c] > 0) {
                wmask += w;
                sx += w * gfun(((double)c - lcx) * inv_rx);
                sy += w * gy;
            }
        }
    }

    out.valid = wtot > 0.0;
    out.sx = sx; out.sy = sy; out.wtot = wtot; out.wmask = wmask;
    out.rx = rx; out.ry = ry; out.cx = cx; out.cy = cy;
    return out;
}

struct MeanShiftResult {
    cv::Point2d xy;
    double score = -1.0;
    int iters = 0;
    bool converged = false;
};

static MeanShiftResult meanShiftLocate(
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Mat& mask_crop, int off_x, int off_y,
    double X0, double Y0, double x_half_range, double y_half_range,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    const MeanShiftParams& P, double start_X = std::numeric_limits<double>::quiet_NaN(),
    double start_Y = std::numeric_limits<double>::quiet_NaN())
{
    auto lut_ptr = getResponseLUT(P.inner_weight, P.margin_scale, P.tau, MS_RING_WEIGHT);
    const ResponseLUT& lut = *lut_ptr;

    double zc = 0.0;
    for (auto& p : local_pts_search)
        zc += p.z;
    zc /= std::max<size_t>(1, local_pts_search.size());

    MeanShiftResult res;
    res.xy = cv::Point2d(X0, Y0);

    // Hakulaatikko on aina (X0,Y0) keskella; aloituspiste voi olla ennustettu paikka.
    double X = std::isnan(start_X) ? X0 : std::max(X0 - x_half_range, std::min(X0 + x_half_range, start_X));
    double Y = std::isnan(start_Y) ? Y0 : std::max(Y0 - y_half_range, std::min(Y0 + y_half_range, start_Y));
    double gain = P.gain;
    double prev_mag = 1e30;

    for (int it = 0; it < P.max_iter; ++it) {

        auto hull = predictedHull(local_pts_search, X, Y, K, R, t);
        if (hull.size() < 3)
            break;

        MsSignals sg = computeMsSignals(mask_crop, off_x, off_y, hull, P);
        if (!sg.valid)
            break;
        double rx = sg.rx, ry = sg.ry, wtot = sg.wtot, wmask = sg.wmask, sx = sg.sx, sy = sg.sy;

        if (wtot <= 0.0 || wmask < MS_MIN_MASK_FRACTION * wtot)
            break;

        double dpx = gain * rx * responseInverse(lut, sx / wtot);
        double dpy = gain * ry * responseInverse(lut, sy / wtot);

        // Paikallinen Jakobiaani (px per cm) kiven runkokorkeudella.
        std::vector<cv::Point3d> jp{
            cv::Point3d(X, Y, zc), cv::Point3d(X + 1.0, Y, zc), cv::Point3d(X, Y + 1.0, zc)
        };
        auto jq = project3d(K, R, t, jp);
        cv::Matx22d J(jq[1].x - jq[0].x, jq[2].x - jq[0].x,
                      jq[1].y - jq[0].y, jq[2].y - jq[0].y);
        double det = J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0);
        if (std::fabs(det) < 1e-9)
            break;
        double dX = ( J(1, 1) * dpx - J(0, 1) * dpy) / det;
        double dY = (-J(1, 0) * dpx + J(0, 0) * dpy) / det;

        double newX = std::max(X0 - x_half_range, std::min(X0 + x_half_range, X + dX));
        double newY = std::max(Y0 - y_half_range, std::min(Y0 + y_half_range, Y + dY));

        double mag = std::hypot(dpx, dpy);
        if (it >= 2 && mag > 0.9 * prev_mag)
            gain = std::max(0.2, gain * 0.5);
        prev_mag = mag;

        X = newX;
        Y = newY;
        res.iters = it + 1;

        if (mag < P.tol_px) {
            res.converged = true;
            break;
        }
    }

    res.xy = cv::Point2d(X, Y);
    auto hull = predictedHull(local_pts_search, X, Y, K, R, t);
    res.score = hullOverlapScore(mask_crop, hull, off_x, off_y);

    // Viimeistely: maennousu 8-naapurustossa peitto-osuudella, askel puolittuu (maskin epasymmetrian vinouma)
    if (P.polish_step_cm > 0.0 && res.score > -0.5) {
        double step = P.polish_step_cm;
        double bx = X, by = Y, bs = res.score;
        while (step >= MS_POLISH_MIN_STEP_CM) {
            bool improved = true;
            int guard = 0;
            while (improved && guard++ < 6) {
                improved = false;
                double nx = bx, ny = by, ns = bs;
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0)
                            continue;
                        double cxp = std::max(X0 - x_half_range, std::min(X0 + x_half_range, bx + di * step));
                        double cyp = std::max(Y0 - y_half_range, std::min(Y0 + y_half_range, by + dj * step));
                        auto hh = predictedHull(local_pts_search, cxp, cyp, K, R, t);
                        double sc = hullOverlapScore(mask_crop, hh, off_x, off_y);
                        if (sc > ns + 1e-9) {
                            ns = sc; nx = cxp; ny = cyp;
                        }
                    }
                }
                if (ns > bs + 1e-9) {
                    bs = ns; bx = nx; by = ny;
                    improved = true;
                }
            }
            step *= 0.5;
        }
        res.xy = cv::Point2d(bx, by);
        res.score = bs;
    }
    return res;
}
