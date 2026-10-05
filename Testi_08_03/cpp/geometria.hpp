// ============================================================
// geometria.hpp - kameraprojektio, kiven ennustettu aariviiva (hulli), peitto-osuus maskiin ja ristikkohaut
// (HAKU: koko hakualue linearisoidulla hullilla; SEURANTA: laajeneva haku edellisesta paikasta + maennousu).
// ============================================================
#pragma once
#include "yhteiset.hpp"

// 3D-pisteet (cm, jaataso Z = 0) kuvaan: p = K (R X + t)
static std::vector<cv::Point2d> project3d(
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    const std::vector<cv::Point3d>& pts)
{
    std::vector<cv::Point2d> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        cv::Vec3d p(pts[i].x, pts[i].y, pts[i].z);
        cv::Vec3d pc = R * p + t;
        cv::Vec3d pi = K * pc;
        out[i] = cv::Point2d(pi[0] / pi[2], pi[1] / pi[2]);
    }
    return out;
}

// Kiven ennustettu aariviiva: paikallinen pistejoukko siirrettyna paikkaan (X0, Y0), projisoituna ja kuperana peitteena.
// Tyhja jos projektio ei ole aarellinen tai peitteessa on alle 3 karkea.
static std::vector<cv::Point2f> predictedHull(
    const std::vector<cv::Point3d>& local_pts, double X0, double Y0,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t)
{
    std::vector<cv::Point2f> proj_f(local_pts.size());
    bool all_finite = true;
    for (size_t i = 0; i < local_pts.size(); ++i) {
        cv::Vec3d p(local_pts[i].x + X0, local_pts[i].y + Y0, local_pts[i].z);
        cv::Vec3d pc = R * p + t;
        cv::Vec3d pi = K * pc;
        double px = pi[0] / pi[2], py = pi[1] / pi[2];
        if (!std::isfinite(px) || !std::isfinite(py))
            all_finite = false;
        proj_f[i] = cv::Point2f((float)px, (float)py);
    }
    if (!all_finite)
        return {};
    std::vector<cv::Point2f> hull;
    cv::convexHull(proj_f, hull);
    if (hull.size() < 3)
        return {};
    return hull;
}

// ============================================================
// PEITTO-OSUUS: (maskipikselit hullin sisalla / hullin ala) - HULL_OUTSIDE_PENALTY_WEIGHT * (maskipikselit
// turvavyohykkeella / vyohykkeen ala). Hull annetaan koko kuvan koordinaatistossa; off_x/off_y siirtaa sen
// ROI-rajatun maskin koordinaatistoon. Pisteet katkaistaan kokonaisluvuiksi ENNEN siirtoa (sama tulos kuin
// koko kuvan kokoisella maskilla).
// ============================================================
static double hullOverlapScore(
    const cv::Mat& mask_crop, const std::vector<cv::Point2f>& hull,
    int off_x, int off_y)
{
    if (hull.size() < 3)
        return 0.0;

    std::vector<cv::Point> hull_int(hull.size());
    for (size_t i = 0; i < hull.size(); ++i)
        hull_int[i] = cv::Point((int)(hull[i].x), (int)(hull[i].y));
    for (auto& p : hull_int) { p.x -= off_x; p.y -= off_y; }

    cv::Point2d centroid(0.0, 0.0);
    for (auto& p : hull_int) { centroid.x += p.x; centroid.y += p.y; }
    centroid.x /= (double)hull_int.size();
    centroid.y /= (double)hull_int.size();

    std::vector<cv::Point> margin_int(hull_int.size());
    for (size_t i = 0; i < hull_int.size(); ++i) {
        margin_int[i] = cv::Point(
            (int)std::lround(centroid.x + (hull_int[i].x - centroid.x) * HULL_MARGIN_SCALE),
            (int)std::lround(centroid.y + (hull_int[i].y - centroid.y) * HULL_MARGIN_SCALE)
        );
    }

    cv::Rect bbox = cv::boundingRect(margin_int);
    int x0 = std::max(bbox.x, 0);
    int y0 = std::max(bbox.y, 0);
    int x1 = std::min(bbox.x + bbox.width, mask_crop.cols);
    int y1 = std::min(bbox.y + bbox.height, mask_crop.rows);
    if (x1 <= x0 || y1 <= y0)
        return 0.0;

    // Yksi saiekohtainen pohja: turvavyohyke arvolla 1, hulli sen paalle arvolla 2
    std::vector<cv::Point> shifted(hull_int.size()), margin_sh(margin_int.size());
    for (size_t i = 0; i < hull_int.size(); ++i)
        shifted[i] = cv::Point(hull_int[i].x - x0, hull_int[i].y - y0);
    for (size_t i = 0; i < margin_int.size(); ++i)
        margin_sh[i] = cv::Point(margin_int[i].x - x0, margin_int[i].y - y0);
    const int W = x1 - x0, Hh = y1 - y0;
    thread_local std::vector<uchar> buf;
    if (buf.size() < (size_t)W * (size_t)Hh) buf.resize((size_t)W * (size_t)Hh);
    cv::Mat cv_(Hh, W, CV_8UC1, buf.data());
    cv_.setTo(cv::Scalar(0));
    const cv::Point* mp = margin_sh.data(); int mn = (int)margin_sh.size();
    cv::fillPoly(cv_, &mp, &mn, 1, cv::Scalar(1));
    const cv::Point* hp = shifted.data(); int hn = (int)shifted.size();
    cv::fillPoly(cv_, &hp, &hn, 1, cv::Scalar(2));

    int h_area = 0, ov = 0, band = 0, out_ov = 0;
    for (int r = 0; r < Hh; ++r) {
        const uchar* c = cv_.ptr<uchar>(r);
        const uchar* m = mask_crop.ptr<uchar>(y0 + r) + x0;
        for (int k = 0; k < W; ++k) {
            const int v = c[k];
            const int mm = m[k] > 0;
            h_area += (v == 2);
            ov += (v == 2) & mm;
            band += (v == 1);
            out_ov += (v == 1) & mm;
        }
    }
    if (h_area == 0)
        return 0.0;
    double ins = (double)ov / (double)h_area;
    double outs = band > 0 ? (double)out_ov / (double)band : 0.0;
    return ins - HULL_OUTSIDE_PENALTY_WEIGHT * outs;
}

// ============================================================
// RISTIKKOHAKU
// ============================================================

// numpy.arange: pituus ceil((stop - start) / step), arvot start + i * step
static std::vector<double> arangeVec(double start, double stop, double step)
{
    std::vector<double> out;
    if (step <= 0.0)
        return out;
    long count = (long)std::ceil((stop - start) / step);
    if (count < 0)
        count = 0;
    out.reserve((size_t)count);
    for (long i = 0; i < count; ++i)
        out.push_back(start + (double)i * step);
    return out;
}

// Nopea ennustettu hulli: projektio K (R (local + (X, Y, 0)) + t) on lineaarinen (X, Y):n suhteen ennen jakoa, joten
// K (R local + t) lasketaan kerran ja jokainen kokeilu on A_i + X kx + Y ky + jako. Kuperan peitteen karjet lasketaan
// uudelleen vain kun (X, Y) on yli 15 cm edellisesta vertailupisteesta (sen sisalla karjet eivat muutu).
struct HullProjector {
    std::vector<cv::Vec3d> A;
    cv::Vec3d kx, ky;
    std::vector<cv::Point2f> pts;
    std::vector<int> ref_idx;
    double ref_x = 0, ref_y = 0;

    HullProjector(const std::vector<cv::Point3d>& local_pts, const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t)
        : A(local_pts.size()), pts(local_pts.size())
    {
        for (size_t i = 0; i < local_pts.size(); ++i)
            A[i] = K * (R * cv::Vec3d(local_pts[i].x, local_pts[i].y, local_pts[i].z) + t);
        kx = K * cv::Vec3d(R(0, 0), R(1, 0), R(2, 0));
        ky = K * cv::Vec3d(R(0, 1), R(1, 1), R(2, 1));
    }

    std::vector<cv::Point2f> hull(double X, double Y)
    {
        const double ox = X * kx[0] + Y * ky[0], oy = X * kx[1] + Y * ky[1], oz = X * kx[2] + Y * ky[2];
        const bool fast = !ref_idx.empty() && std::abs(X - ref_x) < 15.0 && std::abs(Y - ref_y) < 15.0;
        if (fast) {
            std::vector<cv::Point2f> h(ref_idx.size());
            for (size_t k = 0; k < ref_idx.size(); ++k) {
                const cv::Vec3d& a = A[(size_t)ref_idx[k]];
                const double z = a[2] + oz;
                h[k] = cv::Point2f((float)((a[0] + ox) / z), (float)((a[1] + oy) / z));
            }
            return h;
        }
        bool all_finite = true;
        for (size_t i = 0; i < A.size(); ++i) {
            const double z = A[i][2] + oz;
            const double px = (A[i][0] + ox) / z, py = (A[i][1] + oy) / z;
            if (!std::isfinite(px) || !std::isfinite(py)) all_finite = false;
            pts[i] = cv::Point2f((float)px, (float)py);
        }
        if (!all_finite) return {};
        std::vector<int> idx;
        cv::convexHull(pts, idx, false, false);
        if (idx.size() < 3) return {};
        std::vector<cv::Point2f> h(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) h[k] = pts[(size_t)idx[k]];
        ref_idx = idx; ref_x = X; ref_y = Y;
        return h;
    }
};

// Tyhjentava haku: paras pistemaara (aidosti suurempi voittaa, rivi kerrallaan)
static std::pair<cv::Point2d, double> gridSearchBest(
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Mat& mask_crop, int off_x, int off_y,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    const std::vector<double>& x_vals, const std::vector<double>& y_vals)
{
    double best_score = -1.0;
    cv::Point2d best_xy(x_vals.empty() ? 0.0 : x_vals[0], y_vals.empty() ? 0.0 : y_vals[0]);
    HullProjector proj(local_pts_search, K, R, t);
    for (double X : x_vals) {
        for (double Y : y_vals) {
            double score = hullOverlapScore(mask_crop, proj.hull(X, Y), off_x, off_y);
            if (score > best_score) {
                best_score = score;
                best_xy = cv::Point2d(X, Y);
            }
        }
    }
    return {best_xy, best_score};
}

// HAKU: hakualue on kapea ja kaukana kamerasta, joten kiven projektion koko ja muoto eivat juuri muutu sen sisalla.
// Hulli lasketaan kerran alueen keskella ja siirretaan muihin pisteisiin paikallisella Jakobiaanilla (px / cm).
static std::pair<cv::Point2d, double> gridSearchBestLinearized(
    const cv::Mat& mask_crop, int off_x, int off_y,
    const std::vector<cv::Point2f>& ref_hull, double ref_x, double ref_y,
    const cv::Point2d& jac_col_x, const cv::Point2d& jac_col_y,
    const std::vector<double>& x_vals, const std::vector<double>& y_vals)
{
    double best_score = -1.0;
    cv::Point2d best_xy(x_vals.empty() ? 0.0 : x_vals[0], y_vals.empty() ? 0.0 : y_vals[0]);
    std::vector<cv::Point2f> hull(ref_hull.size());
    for (double X : x_vals) {
        double dX = X - ref_x;
        for (double Y : y_vals) {
            double dY = Y - ref_y;
            double dpx = dX * jac_col_x.x + dY * jac_col_y.x;
            double dpy = dX * jac_col_x.y + dY * jac_col_y.y;
            for (size_t i = 0; i < ref_hull.size(); ++i)
                hull[i] = cv::Point2f(ref_hull[i].x + (float)dpx, ref_hull[i].y + (float)dpy);
            double score = hullOverlapScore(mask_crop, hull, off_x, off_y);
            if (score > best_score) {
                best_score = score;
                best_xy = cv::Point2d(X, Y);
            }
        }
    }
    return {best_xy, best_score};
}

// HAKU: karkea ristikko koko alueelle, sitten hieno ristikko parhaan pisteen ymparille (+- karkea askel)
static std::pair<cv::Point2d, double> locateByGridSearchFast(
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Mat& mask_crop, int off_x, int off_y,
    double x_center, double x_half_range, double y_center, double y_half_range,
    double coarse_step, double fine_step,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t)
{
    static const double JAC_DELTA_CM = 10.0;
    auto ref_hull = predictedHull(local_pts_search, x_center, y_center, K, R, t);
    std::vector<cv::Point3d> jac_pts{
        cv::Point3d(x_center, y_center, 0.0),
        cv::Point3d(x_center + JAC_DELTA_CM, y_center, 0.0),
        cv::Point3d(x_center, y_center + JAC_DELTA_CM, 0.0),
    };
    auto jac_proj = project3d(K, R, t, jac_pts);
    cv::Point2d jac_col_x((jac_proj[1].x - jac_proj[0].x) / JAC_DELTA_CM, (jac_proj[1].y - jac_proj[0].y) / JAC_DELTA_CM);
    cv::Point2d jac_col_y((jac_proj[2].x - jac_proj[0].x) / JAC_DELTA_CM, (jac_proj[2].y - jac_proj[0].y) / JAC_DELTA_CM);

    auto x_vals = arangeVec(x_center - x_half_range, x_center + x_half_range + 1e-6, coarse_step);
    auto y_vals = arangeVec(y_center - y_half_range, y_center + y_half_range + 1e-6, coarse_step);
    auto best1 = gridSearchBestLinearized(mask_crop, off_x, off_y, ref_hull, x_center, y_center, jac_col_x, jac_col_y,
                                          x_vals, y_vals);

    auto x_vals2 = arangeVec(best1.first.x - coarse_step, best1.first.x + coarse_step + 1e-6, fine_step);
    auto y_vals2 = arangeVec(best1.first.y - coarse_step, best1.first.y + coarse_step + 1e-6, fine_step);
    return gridSearchBestLinearized(mask_crop, off_x, off_y, ref_hull, x_center, y_center, jac_col_x, jac_col_y,
                                    x_vals2, y_vals2);
}

// ============================================================
// SEURANNAN RISTIKKOHAKU. Karkea vaihe laajenee lahtopisteesta (x_center, y_center) rengas kerrallaan
// (Tsebysevin etaisyys ruudukkoindekseissa) ja pysahtyy kun pistemaara >= TRACK_EARLY_STOP_SCORE; sen jalkeen
// maennousu 8-naapurustossa. Jos riittavaa pistemaaraa ei loydy, koko alue kaydaan lapi. Hieno vaihe: maennousu
// hienossa ristikossa (+- karkea askel) karkean parhaan pisteen lahimmasta ruudusta.
// ============================================================
static std::pair<cv::Point2d, double> locateByGridSearchTrackingFast(
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Mat& mask_crop, int off_x, int off_y,
    double x_center, double x_half_range, double y_center, double y_half_range,
    double coarse_step, double fine_step,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t)
{
    auto x_vals = arangeVec(x_center - x_half_range, x_center + x_half_range + 1e-6, coarse_step);
    auto y_vals = arangeVec(y_center - y_half_range, y_center + y_half_range + 1e-6, coarse_step);
    int nx = (int)x_vals.size(), ny = (int)y_vals.size();

    std::pair<cv::Point2d, double> best1;
    if (nx == 0 || ny == 0) {
        best1 = gridSearchBest(local_pts_search, mask_crop, off_x, off_y, K, R, t, x_vals, y_vals);
    } else {
        std::vector<char> visited((size_t)nx * (size_t)ny, 0);
        auto idx = [ny](int i, int j) { return (size_t)i * (size_t)ny + (size_t)j; };
        HullProjector hull_proj(local_pts_search, K, R, t);

        auto evalPoint = [&](int i, int j) -> double {
            visited[idx(i, j)] = 1;
            PT gpt;
            auto hull = hull_proj.hull(x_vals[(size_t)i], y_vals[(size_t)j]);
            profAdd(P_GS_HULL, gpt.lap());
            double sc = hullOverlapScore(mask_crop, hull, off_x, off_y);
            profAdd(P_GS_SCORE, gpt.lap());
            return sc;
        };

        int ix0 = (int)std::lround((x_center - x_vals[0]) / coarse_step);
        int iy0 = (int)std::lround((y_center - y_vals[0]) / coarse_step);
        ix0 = std::max(0, std::min(nx - 1, ix0));
        iy0 = std::max(0, std::min(ny - 1, iy0));

        double best_score = -1.0;
        int best_i = ix0, best_j = iy0;
        bool early_stop = false;
        int max_ring = std::max({ ix0, nx - 1 - ix0, iy0, ny - 1 - iy0 });

        for (int ring = 0; ring <= max_ring && !early_stop; ++ring) {
            for (int i = std::max(0, ix0 - ring); i <= std::min(nx - 1, ix0 + ring) && !early_stop; ++i) {
                for (int j = std::max(0, iy0 - ring); j <= std::min(ny - 1, iy0 + ring); ++j) {
                    if (std::max(std::abs(i - ix0), std::abs(j - iy0)) != ring)
                        continue;
                    if (visited[idx(i, j)])
                        continue;
                    double score = evalPoint(i, j);
                    if (score > best_score) {
                        best_score = score;
                        best_i = i; best_j = j;
                    }
                    if (score >= TRACK_EARLY_STOP_SCORE) {
                        early_stop = true;
                        break;
                    }
                }
            }
        }

        if (early_stop) {
            int ci = best_i, cj = best_j;
            double cscore = best_score;
            while (true) {
                int ni_best = ci, nj_best = cj;
                double n_best_score = cscore;
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0)
                            continue;
                        int ni = ci + di, nj = cj + dj;
                        if (ni < 0 || ni >= nx || nj < 0 || nj >= ny)
                            continue;
                        if (visited[idx(ni, nj)])
                            continue;
                        double s = evalPoint(ni, nj);
                        if (s > best_score) {
                            best_score = s;
                            best_i = ni; best_j = nj;
                        }
                        if (s > n_best_score) {
                            n_best_score = s;
                            ni_best = ni; nj_best = nj;
                        }
                    }
                }
                if (ni_best == ci && nj_best == cj)
                    break;
                ci = ni_best; cj = nj_best; cscore = n_best_score;
            }
        }
        best1 = { cv::Point2d(x_vals[(size_t)best_i], y_vals[(size_t)best_j]), best_score };
    }

    auto x_vals2 = arangeVec(best1.first.x - coarse_step, best1.first.x + coarse_step + 1e-6, fine_step);
    auto y_vals2 = arangeVec(best1.first.y - coarse_step, best1.first.y + coarse_step + 1e-6, fine_step);
    if (x_vals2.empty() || y_vals2.empty())
        return gridSearchBest(local_pts_search, mask_crop, off_x, off_y, K, R, t, x_vals2, y_vals2);

    const int nx2 = (int)x_vals2.size(), ny2 = (int)y_vals2.size();
    std::vector<double> sc((size_t)nx2 * ny2, std::numeric_limits<double>::quiet_NaN());
    HullProjector hp(local_pts_search, K, R, t);
    auto ev = [&](int i, int j) {
        double& v = sc[(size_t)i * ny2 + j];
        if (std::isnan(v))
            v = hullOverlapScore(mask_crop, hp.hull(x_vals2[(size_t)i], y_vals2[(size_t)j]), off_x, off_y);
        return v;
    };
    int ci = (int)std::lround((best1.first.x - x_vals2[0]) / fine_step);
    int cj = (int)std::lround((best1.first.y - y_vals2[0]) / fine_step);
    ci = std::max(0, std::min(nx2 - 1, ci)); cj = std::max(0, std::min(ny2 - 1, cj));
    double cs = ev(ci, cj);
    while (true) {
        int bi = ci, bj = cj; double bs = cs;
        for (int di = -1; di <= 1; ++di)
            for (int dj = -1; dj <= 1; ++dj) {
                if (!di && !dj) continue;
                int ni = ci + di, nj = cj + dj;
                if (ni < 0 || nj < 0 || ni >= nx2 || nj >= ny2) continue;
                double v = ev(ni, nj);
                if (v > bs) { bs = v; bi = ni; bj = nj; }
            }
        if (bi == ci && bj == cj) break;
        ci = bi; cj = bj; cs = bs;
    }
    return { cv::Point2d(x_vals2[(size_t)ci], y_vals2[(size_t)cj]), cs };
}
