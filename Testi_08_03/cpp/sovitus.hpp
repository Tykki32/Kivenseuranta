// ============================================================
// sovitus.hpp - kiven paikan tarkennus: runkokontuuri ja rengaspisteet maskista / saturaatiosta, jaannosvirheet
// 3D-mallia vasten ja Levenberg-Marquardt-sovitus (X, Y, renkaan sade).
// ============================================================
#pragma once
#include "maskit.hpp"

// ------------------------------------------------------------
// RENGASPISTEET: sateittain (n_angles suuntaa) ensimmainen saturaation lasku kynnyksen alle, alipikselitarkasti
// ------------------------------------------------------------

static std::vector<cv::Point2d> detectBoundaryPoints(
    const PaddedView& sat, double cx_px, double cy_px, double r_px_approx,
    int n_angles = BOUNDARY_N_ANGLES,
    double threshold = BOUNDARY_SATURATION_THRESHOLD,
    double r_step = BOUNDARY_RADIAL_STEP_PX,
    double r_factor_min = BOUNDARY_RADIUS_SEARCH_FACTOR_MIN,
    double r_factor_max = BOUNDARY_RADIUS_SEARCH_FACTOR_MAX)
{
    double r_min = r_px_approx * r_factor_min;
    double r_max = r_px_approx * r_factor_max;

    int n_steps = (int)std::floor((r_max - r_min) / r_step) + 1;
    if (n_steps < 2)
        return {};

    std::vector<double> r_vals((size_t)n_steps);
    for (int i = 0; i < n_steps; ++i)
        r_vals[(size_t)i] = r_min + (double)i * r_step;

    std::vector<double> cos_t((size_t)n_angles), sin_t((size_t)n_angles);
    for (int i = 0; i < n_angles; ++i) {
        double theta = 2.0 * M_PI * (double)i / (double)n_angles;
        cos_t[(size_t)i] = std::cos(theta);
        sin_t[(size_t)i] = std::sin(theta);
    }

    std::vector<cv::Point2d> points;
    std::vector<double> vals((size_t)n_steps);
    std::vector<uint8_t> valid((size_t)n_steps);

    for (int i = 0; i < n_angles; ++i) {

        for (int j = 0; j < n_steps; ++j) {
            double x = cx_px + r_vals[(size_t)j] * cos_t[(size_t)i];
            double y = cy_px + r_vals[(size_t)j] * sin_t[(size_t)i];
            double val;
            bool ok = bilinearSample(sat, x, y, val);
            vals[(size_t)j] = val;
            valid[(size_t)j] = ok ? 1 : 0;
        }

        int idx = -1;
        for (int j = 0; j < n_steps - 1; ++j) {
            if (!valid[(size_t)j] || !valid[(size_t)(j + 1)])
                continue;
            bool above = vals[(size_t)j] >= threshold;
            bool below = vals[(size_t)(j + 1)] < threshold;
            if (above && below) {
                idx = j;
                break;
            }
        }

        if (idx < 0)
            continue;

        double v0 = vals[(size_t)idx], v1 = vals[(size_t)(idx + 1)];
        double r0 = r_vals[(size_t)idx], r1 = r_vals[(size_t)(idx + 1)];
        double span = v1 - v0;
        double tt = (std::abs(span) > 1e-9) ? (threshold - v0) / span : 0.0;
        double r_cross = r0 + tt * (r1 - r0);

        points.push_back(cv::Point2d(
            cx_px + r_cross * cos_t[(size_t)i],
            cy_px + r_cross * sin_t[(size_t)i]
        ));
    }

    return points;
}


static double angularSpreadDeg(const std::vector<cv::Point2d>& points, double cx, double cy)
{
    if (points.size() < 2)
        return 0.0;

    std::vector<double> angles(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        double a = std::atan2(points[i].y - cy, points[i].x - cx) * 180.0 / M_PI;
        a = std::fmod(a, 360.0);
        if (a < 0) a += 360.0;
        angles[i] = a;
    }

    std::sort(angles.begin(), angles.end());

    double largest_gap = angles[0] + 360.0 - angles.back();
    for (size_t i = 0; i + 1 < angles.size(); ++i)
        largest_gap = std::max(largest_gap, angles[i + 1] - angles[i]);

    return 360.0 - largest_gap;
}


// ------------------------------------------------------------
// RUNKOKONTUURI: lahin riittavan iso maskikontuuri (painopiste <= max_dist_px). Kontuurin kokoa ei rajoiteta: juuri
// heitetyn / lakaistavan kiven kontuuri voi olla yhtenainen heittajan tai lakaisijan kanssa, ja tarkein on loytaa
// kaikki heitot. coord_shift: maski on pienemmalla reunuksella kuin kutsujan koordinaatisto (siirto ennen laskentaa).
// ------------------------------------------------------------
static bool findContourNear(
    const cv::Mat& mask, cv::Point2d approx_px, std::vector<cv::Point>& best,
    double max_dist_px = BODY_CONTOUR_MAX_SEARCH_DIST_PX,
    double min_area = BODY_CONTOUR_MIN_AREA_PX,
    cv::Point coord_shift = cv::Point(0, 0))
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (coord_shift.x != 0 || coord_shift.y != 0)
        for (auto& c : contours) for (auto& p : c) p += coord_shift;

    bool found = false;
    double best_dist = 0.0;

    for (auto& c : contours) {

        double area = cv::contourArea(c);
        if (area < min_area)
            continue;

        cv::Moments M = cv::moments(c);
        if (M.m00 == 0)
            continue;

        double ccx = M.m10 / M.m00, ccy = M.m01 / M.m00;
        double d = std::hypot(ccx - approx_px.x, ccy - approx_px.y);

        if (d > max_dist_px)
            continue;

        if (!found || d < best_dist) {
            best = c;
            best_dist = d;
            found = true;
        }
    }

    return found;
}


// Runkokontuurin pisteet, joiden ulkopuolella (check_dist_px sateen suuntaan) on matalakylläista jaata: kahvan
// kohdalla (kylläinen) olevat pisteet jaavat pois.
static std::vector<cv::Point2d> filterIceBoundaryPoints(
    const PaddedView& sat, const std::vector<cv::Point>& contour,
    double check_dist_px = BODY_HANDLE_CHECK_DIST_PX,
    double threshold = BOUNDARY_SATURATION_THRESHOLD)
{
    std::vector<cv::Point2d> out;
    if (contour.empty())
        return out;

    double cx = 0.0, cy = 0.0;
    for (auto& p : contour) { cx += p.x; cy += p.y; }
    cx /= (double)contour.size();
    cy /= (double)contour.size();

    for (auto& p : contour) {

        double dx = (double)p.x - cx, dy = (double)p.y - cy;
        double norm = std::hypot(dx, dy);

        if (norm <= 1e-6)
            continue;

        dx /= norm;
        dy /= norm;

        double ox = (double)p.x + dx * check_dist_px;
        double oy = (double)p.y + dy * check_dist_px;

        double sat_val;
        bool ok = bilinearSample(sat, ox, oy, sat_val);

        if (ok && sat_val < threshold)
            out.push_back(cv::Point2d((double)p.x, (double)p.y));
    }

    return out;
}


// ------------------------------------------------------------
// JAANNOSVIRHEET. Kahvan kolo: ympyra (sade handle_r_frac * R_max) kiven ylapinnalla (Z = H_total) keskella;
// handle_r_frac on sovitettu kiviprofiiliin (fit_stone_profile_cpp).
// ------------------------------------------------------------
static std::vector<cv::Point2f> predictedNotchHull(
    double X0, double Y0, double R_max_cm, double H_total_cm, double handle_r_frac,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    int n_theta = 28)
{
    double r_cm = handle_r_frac * R_max_cm;
    std::vector<cv::Point2f> proj_f((size_t)n_theta);
    bool all_finite = true;

    for (int i = 0; i < n_theta; ++i) {
        double theta = 2.0 * M_PI * (double)i / (double)n_theta;
        cv::Vec3d p(X0 + r_cm * std::cos(theta), Y0 + r_cm * std::sin(theta), H_total_cm);
        cv::Vec3d pc = R * p + t;
        cv::Vec3d pi = K * pc;
        double px = pi[0] / pi[2], py = pi[1] / pi[2];

        if (!std::isfinite(px) || !std::isfinite(py))
            all_finite = false;
        proj_f[(size_t)i] = cv::Point2f((float)px, (float)py);
    }

    if (!all_finite)
        return {};

    std::vector<cv::Point2f> hull;
    cv::convexHull(proj_f, hull);

    if (hull.size() < 3)
        return {};

    return hull;
}

// Etumerkillinen etaisyys rungon (hull) miinus kolon reunaan: positiivinen kolollisen muodon sisalla, negatiivinen
// ulkopuolella (hullin ulkopuolella tai kolon sisalla).
static double signedDistWithNotch(
    const std::vector<cv::Point2f>& hull,
    const std::vector<cv::Point2f>& notch_hull,
    const cv::Point2f& pt)
{
    double d_outer = cv::pointPolygonTest(hull, pt, true);

    if (notch_hull.empty() || d_outer <= 0.0)
        return d_outer;

    double d_notch = cv::pointPolygonTest(notch_hull, pt, true);

    if (d_notch > 0.0)
        return -d_notch;

    return std::min(d_outer, -d_notch);
}

// Runkopisteiden etaisyydet ennustettuun (kololliseen) aariviivaan. Linearisoitu LM (alla) mallintaa kolon omalla
// linearisoinnillaan; lopullinen tulos tarkistetaan aina tallä tarkalla funktiolla.
static std::vector<double> profileResiduals(
    const std::vector<cv::Point3d>& local_pts_body, double X0, double Y0,
    double R_max_cm, double H_total_cm, double handle_r_frac,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    const std::vector<cv::Point2f>& pts_f)
{
    auto hull = predictedHull(local_pts_body, X0, Y0, K, R, t);

    std::vector<double> out(pts_f.size());

    if (hull.empty()) {
        std::fill(out.begin(), out.end(), 1000.0);
        return out;
    }

    auto notch_hull = predictedNotchHull(X0, Y0, R_max_cm, H_total_cm, handle_r_frac, K, R, t);

    for (size_t i = 0; i < pts_f.size(); ++i)
        out[i] = signedDistWithNotch(hull, notch_hull, pts_f[i]);

    return out;
}


static std::vector<cv::Point2f> predictedRingPoints(
    double X0, double Y0, double ring_radius_cm, double ring_height_cm,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    int n_theta = 120)
{
    std::vector<cv::Point2f> out((size_t)n_theta);

    for (int i = 0; i < n_theta; ++i) {
        double theta = 2.0 * M_PI * (double)i / (double)n_theta;
        cv::Vec3d p(
            X0 + ring_radius_cm * std::cos(theta),
            Y0 + ring_radius_cm * std::sin(theta),
            ring_height_cm
        );
        cv::Vec3d pc = R * p + t;
        cv::Vec3d pi = K * pc;
        out[(size_t)i] = cv::Point2f((float)(pi[0] / pi[2]), (float)(pi[1] / pi[2]));
    }

    return out;
}


// Rengaspisteiden etaisyydet projisoituun renkaaseen (sade ring_radius_cm, korkeus ring_height_cm)
static std::vector<double> ringPointResiduals(
    double X0, double Y0, double ring_radius_cm, double ring_height_cm,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    const std::vector<cv::Point2f>& observed_f)
{
    auto poly = predictedRingPoints(X0, Y0, ring_radius_cm, ring_height_cm, K, R, t);

    std::vector<double> out(observed_f.size());
    for (size_t i = 0; i < observed_f.size(); ++i)
        out[i] = cv::pointPolygonTest(poly, observed_f[i], true);

    return out;
}


static double median(std::vector<double> v)
{
    if (v.empty())
        return 0.0;

    std::sort(v.begin(), v.end());
    size_t n = v.size();

    if (n % 2 == 1)
        return v[n / 2];

    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}


// ------------------------------------------------------------
// LEVENBERG-MARQUARDT 3 parametrille (X, Y, renkaan sade). Pistejoukot muunnetaan floatiksi kerran.
// ------------------------------------------------------------
static std::vector<cv::Point2f> toPoint2fVec(const std::vector<cv::Point2d>& pts)
{
    std::vector<cv::Point2f> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
        out[i] = cv::Point2f((float)pts[i].x, (float)pts[i].y);
    return out;
}


struct ResidualContext {
    const std::vector<cv::Point3d>* local_pts_body;
    const std::vector<cv::Point2f>* body_pts_f;
    const std::vector<cv::Point2f>* ring_pts_f;
    double ring_height_cm;
    double R_max_cm;
    double handle_r_frac;
    const cv::Matx33d* K;
    const cv::Matx33d* R;
    const cv::Vec3d* t;
};


static std::vector<double> jointResiduals(const ResidualContext& ctx, const cv::Vec3d& params)
{
    double X = params[0], Y = params[1], Rr = params[2];
    std::vector<double> out;

    if (!ctx.body_pts_f->empty()) {
        // ring_height_cm == H_total_cm
        auto r = profileResiduals(*ctx.local_pts_body, X, Y, ctx.R_max_cm, ctx.ring_height_cm,
                                   ctx.handle_r_frac, *ctx.K, *ctx.R, *ctx.t, *ctx.body_pts_f);
        out.insert(out.end(), r.begin(), r.end());
    }

    if (!ctx.ring_pts_f->empty()) {
        auto r = ringPointResiduals(X, Y, Rr, ctx.ring_height_cm, *ctx.K, *ctx.R, *ctx.t, *ctx.ring_pts_f);
        out.insert(out.end(), r.begin(), r.end());
    }

    return out;
}


static bool solve3x3(double A[3][3], const double b[3], double x[3])
{
    double M[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = A[i][j];
        M[i][3] = b[i];
    }

    for (int col = 0; col < 3; ++col) {

        int piv = col;
        double maxval = std::abs(M[col][col]);
        for (int r = col + 1; r < 3; ++r)
            if (std::abs(M[r][col]) > maxval) { maxval = std::abs(M[r][col]); piv = r; }

        if (maxval < 1e-15)
            return false;

        if (piv != col)
            for (int j = 0; j < 4; ++j)
                std::swap(M[col][j], M[piv][j]);

        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            double factor = M[r][col] / M[col][col];
            for (int j = col; j < 4; ++j)
                M[r][j] -= factor * M[col][j];
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (std::abs(M[i][i]) < 1e-15) return false;
        x[i] = M[i][3] / M[i][i];
    }

    return true;
}


// ------------------------------------------------------------
// LINEARISOITU LM: pikseli = K (R (paikallinen + (X, Y, 0)) + t) / z on lineaarinen X:n, Y:n (ja renkaan R:n) suhteen
// ennen jakoa, joten derivaatat saadaan suljetussa muodossa referenssipisteen projektiosta. Yksi taysi projektio
// (+ kupera peite) per LM-kutsu, sen jalkeen kokeilut ensimmaisen kertaluvun approksimaatiolla. Tulos varmistetaan
// tarkalla jaannosfunktiolla (levenbergMarquardt3LinearizedVerified); jos se ei ole vahintaan yhta hyva kuin
// lahtopiste, LM ajetaan tarkasti. MAD-poikkeamien hylkays on epajatkuva, joten linearisoitu ja tarkka LM voivat
// paatya eri (yhta hyviin) paikallisiin minimeihin.
// ------------------------------------------------------------
static const int LM_BODY_STRIDE = 2;    // runkopisteista joka toinen (kun yli 40 pistetta)

struct LinearizedHull {
    bool valid = false;
    std::vector<cv::Point2f> ref_pts;
    std::vector<float> jac_ux, jac_uy;
    std::vector<float> jac_vx, jac_vy;
};

static LinearizedHull buildLinearizedHull(
    const std::vector<cv::Point3d>& local_pts, double X0, double Y0,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t)
{
    LinearizedHull result;

    std::vector<cv::Point2f> proj_f(local_pts.size());
    std::vector<cv::Point3d> pis(local_pts.size());
    bool all_finite = true;

    for (size_t i = 0; i < local_pts.size(); ++i) {
        cv::Vec3d p(local_pts[i].x + X0, local_pts[i].y + Y0, local_pts[i].z);
        cv::Vec3d pc = R * p + t;
        cv::Vec3d pi = K * pc;
        double px = pi[0] / pi[2], py = pi[1] / pi[2];
        if (!std::isfinite(px) || !std::isfinite(py))
            all_finite = false;
        proj_f[i] = cv::Point2f((float)px, (float)py);
        pis[i] = cv::Point3d(pi[0], pi[1], pi[2]);
    }

    if (!all_finite)
        return result;

    std::vector<int> hull_idx;
    cv::convexHull(proj_f, hull_idx, false, false);

    if (hull_idx.size() < 3)
        return result;

    cv::Vec3d kx = K * cv::Vec3d(R(0, 0), R(1, 0), R(2, 0));
    cv::Vec3d ky = K * cv::Vec3d(R(0, 1), R(1, 1), R(2, 1));


    size_t n = hull_idx.size();
    result.ref_pts.resize(n);
    result.jac_ux.resize(n); result.jac_uy.resize(n);
    result.jac_vx.resize(n); result.jac_vy.resize(n);

    for (size_t k = 0; k < n; ++k) {
        size_t idx = (size_t)hull_idx[k];
        result.ref_pts[k] = proj_f[idx];
        double pi0 = pis[idx].x, pi1 = pis[idx].y, pi2 = pis[idx].z;
        double pi2sq = pi2 * pi2;
        result.jac_ux[k] = (float)((kx[0] * pi2 - pi0 * kx[2]) / pi2sq);
        result.jac_uy[k] = (float)((ky[0] * pi2 - pi0 * ky[2]) / pi2sq);
        result.jac_vx[k] = (float)((kx[1] * pi2 - pi1 * kx[2]) / pi2sq);
        result.jac_vy[k] = (float)((ky[1] * pi2 - pi1 * ky[2]) / pi2sq);
    }

    result.valid = true;
    return result;
}

static std::vector<cv::Point2f> evalLinearizedHull(const LinearizedHull& lin, double dX, double dY)
{
    std::vector<cv::Point2f> out(lin.ref_pts.size());
    for (size_t k = 0; k < out.size(); ++k) {
        out[k] = cv::Point2f(
            lin.ref_pts[k].x + (float)(lin.jac_ux[k] * dX + lin.jac_uy[k] * dY),
            lin.ref_pts[k].y + (float)(lin.jac_vx[k] * dX + lin.jac_vy[k] * dY)
        );
    }
    return out;
}


struct LinearizedRing {
    std::vector<cv::Point2f> ref_pts;
    std::vector<float> jac_ux, jac_uy, jac_ur;
    std::vector<float> jac_vx, jac_vy, jac_vr;
};

static LinearizedRing buildLinearizedRing(
    double X0, double Y0, double ring_radius_cm, double ring_height_cm,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    int n_theta = 120)
{
    LinearizedRing result;
    result.ref_pts.resize((size_t)n_theta);
    result.jac_ux.resize((size_t)n_theta); result.jac_uy.resize((size_t)n_theta); result.jac_ur.resize((size_t)n_theta);
    result.jac_vx.resize((size_t)n_theta); result.jac_vy.resize((size_t)n_theta); result.jac_vr.resize((size_t)n_theta);

    cv::Vec3d kx = K * cv::Vec3d(R(0, 0), R(1, 0), R(2, 0));
    cv::Vec3d ky = K * cv::Vec3d(R(0, 1), R(1, 1), R(2, 1));

    for (int i = 0; i < n_theta; ++i) {
        double theta = 2.0 * M_PI * (double)i / (double)n_theta;
        double ct = std::cos(theta), st = std::sin(theta);
        cv::Vec3d p(X0 + ring_radius_cm * ct, Y0 + ring_radius_cm * st, ring_height_cm);
        cv::Vec3d pc = R * p + t;
        cv::Vec3d pi = K * pc;
        double pi0 = pi[0], pi1 = pi[1], pi2 = pi[2];
        double u = pi0 / pi2, v = pi1 / pi2;
        result.ref_pts[(size_t)i] = cv::Point2f((float)u, (float)v);

        double pi2sq = pi2 * pi2;
        double kr0 = ct * kx[0] + st * ky[0];
        double kr1 = ct * kx[1] + st * ky[1];
        double kr2 = ct * kx[2] + st * ky[2];

        result.jac_ux[(size_t)i] = (float)((kx[0] * pi2 - pi0 * kx[2]) / pi2sq);
        result.jac_uy[(size_t)i] = (float)((ky[0] * pi2 - pi0 * ky[2]) / pi2sq);
        result.jac_ur[(size_t)i] = (float)((kr0 * pi2 - pi0 * kr2) / pi2sq);

        result.jac_vx[(size_t)i] = (float)((kx[1] * pi2 - pi1 * kx[2]) / pi2sq);
        result.jac_vy[(size_t)i] = (float)((ky[1] * pi2 - pi1 * ky[2]) / pi2sq);
        result.jac_vr[(size_t)i] = (float)((kr1 * pi2 - pi1 * kr2) / pi2sq);
    }

    return result;
}

static std::vector<cv::Point2f> evalLinearizedRing(const LinearizedRing& lin, double dX, double dY, double dR)
{
    std::vector<cv::Point2f> out(lin.ref_pts.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = cv::Point2f(
            lin.ref_pts[i].x + (float)(lin.jac_ux[i] * dX + lin.jac_uy[i] * dY + lin.jac_ur[i] * dR),
            lin.ref_pts[i].y + (float)(lin.jac_vx[i] * dX + lin.jac_vy[i] * dY + lin.jac_vr[i] * dR)
        );
    }
    return out;
}


struct LinearizedResidualContext {
    const LinearizedHull* hull_lin;
    const LinearizedRing* ring_lin;
    const std::vector<cv::Point2f>* body_pts_f;
    const std::vector<cv::Point2f>* ring_pts_f;
    double X0_ref, Y0_ref, R0_ref;
    const LinearizedRing* notch_lin = nullptr;   // kahvan loveus (signedDistWithNotch), nullptr = ei kaytossa
};

// ------------------------------------------------------------
// ANALYYTTINEN JAKOBIAANI: linearisoidun hullin / renkaan karjet ovat affiineja parametrien suhteen
// (d karki / d param = jac_*), joten pisteen etaisyyden (lahin reuna) derivaatta saadaan ketjusaannolla samassa
// O(N)-lapikaynnissa kuin arvo. Gradientti lasketaan vain hyvaksytylle askeleelle; vaimennuskokeiluille riittaa arvo.
// ------------------------------------------------------------

struct ResidualWithGrad {
    double value = 0.0;
    double dX = 0.0, dY = 0.0, dR = 0.0;
};

// ------------------------------------------------------------------
// NOPEA polygonietaisyys (LM-optimointi, Testi_03_01): sama tulos kuin
// polygonResidualWithGrad, mutta reunojen vakiosuureet (dx,dy,1/len2,
// seuraavan karjen y) lasketaan KERRAN polygonia kohti (ei jokaiselle
// pisteelle), ei modulo-operaatiota eika jakolaskua reunasilmukassa.
// ------------------------------------------------------------------
struct PolyEdges {
    std::vector<double> x1, y1, dx, dy, y2, inv_len2;
    mutable std::vector<double> d2buf, tbuf;
    size_t n = 0;
    void build(const std::vector<cv::Point2f>& pts) {
        n = pts.size();
        x1.resize(n); y1.resize(n); dx.resize(n); dy.resize(n); y2.resize(n); inv_len2.resize(n);
        d2buf.resize(n); tbuf.resize(n);
        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1 == n) ? 0 : i + 1;
            double v1x = pts[i].x, v1y = pts[i].y, v2x = pts[j].x, v2y = pts[j].y;
            x1[i] = v1x; y1[i] = v1y; y2[i] = v2y;
            dx[i] = v2x - v1x; dy[i] = v2y - v1y;
            double len2 = dx[i] * dx[i] + dy[i] * dy[i];
            inv_len2[i] = (len2 < 1e-12) ? 0.0 : 1.0 / len2;
        }
    }
};

static inline ResidualWithGrad polygonResidualFast(
    const PolyEdges& E,
    const float* jac_ux, const float* jac_uy, const float* jac_ur,
    const float* jac_vx, const float* jac_vy, const float* jac_vr,
    double px, double py)
{
    ResidualWithGrad out;
    const size_t n = E.n;
    if (n < 3) { out.value = 1000.0; return out; }

    const double* X1 = E.x1.data(); const double* Y1 = E.y1.data();
    const double* DX = E.dx.data(); const double* DY = E.dy.data();
    const double* Y2 = E.y2.data(); const double* IL = E.inv_len2.data();
    double* D2 = E.d2buf.data(); double* TT = E.tbuf.data();

    // Vaihe 1: etaisyys jokaiseen reunaan - ei haaroja, kaantajan vektoroitavissa.
    for (size_t i = 0; i < n; ++i) {
        const double dx = DX[i], dy = DY[i];
        const double wx = px - X1[i], wy = py - Y1[i];
        double t = (wx * dx + wy * dy) * IL[i];
        t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
        const double ex = wx - t * dx, ey = wy - t * dy;
        D2[i] = ex * ex + ey * ey;
        TT[i] = t;
    }

    // Vaihe 2: lahin reuna + sisapuolitesti (risteykset).
    bool inside = false;
    double best_d2 = std::numeric_limits<double>::max();
    size_t best_i = 0;
    for (size_t i = 0; i < n; ++i) {
        if (D2[i] < best_d2) { best_d2 = D2[i]; best_i = i; }
        const double v1y = Y1[i];
        if ((v1y > py) != (Y2[i] > py)) {
            double x_cross = X1[i] + (py - v1y) / DY[i] * DX[i];
            if (px < x_cross) inside = !inside;
        }
    }
    const double best_t = TT[best_i];

    const double sign = inside ? 1.0 : -1.0;
    const double D = std::sqrt(best_d2);
    out.value = sign * D;
    if (D < 1e-9) return out;

    const size_t i1 = best_i, i2 = (best_i + 1 == n) ? 0 : best_i + 1;
    const double t = best_t;
    const double qx = X1[i1] + t * DX[i1];
    const double qy = Y1[i1] + t * DY[i1];
    const double ux = (qx - px) / D, uy = (qy - py) / D;

    const double dQx_dX = (1.0 - t) * jac_ux[i1] + t * jac_ux[i2];
    const double dQx_dY = (1.0 - t) * jac_uy[i1] + t * jac_uy[i2];
    const double dQy_dX = (1.0 - t) * jac_vx[i1] + t * jac_vx[i2];
    const double dQy_dY = (1.0 - t) * jac_vy[i1] + t * jac_vy[i2];
    out.dX = sign * (ux * dQx_dX + uy * dQy_dX);
    out.dY = sign * (ux * dQx_dY + uy * dQy_dY);
    if (jac_ur != nullptr) {
        const double dQx_dR = (1.0 - t) * jac_ur[i1] + t * jac_ur[i2];
        const double dQy_dR = (1.0 - t) * jac_vr[i1] + t * jac_vr[i2];
        out.dR = sign * (ux * dQx_dR + uy * dQy_dR);
    }
    return out;
}


struct ResidualsAndJacobian {
    std::vector<double> residuals;
    std::vector<std::array<double, 3>> J;
};

static ResidualsAndJacobian jointResidualsAndJacobianLinearized(
    const LinearizedResidualContext& ctx, const cv::Vec3d& params)
{
    ResidualsAndJacobian out;
    double dX = params[0] - ctx.X0_ref;
    double dY = params[1] - ctx.Y0_ref;
    double dR = params[2] - ctx.R0_ref;
    out.residuals.reserve(ctx.body_pts_f->size() + ctx.ring_pts_f->size());
    out.J.reserve(ctx.body_pts_f->size() + ctx.ring_pts_f->size());

    if (!ctx.body_pts_f->empty()) {
        if (ctx.hull_lin->valid) {
            auto hull = evalLinearizedHull(*ctx.hull_lin, dX, dY);
            PolyEdges E; E.build(hull);
            PolyEdges NE; bool notch_built = false;
            std::vector<cv::Point2f> notch_poly;
            for (auto& p : *ctx.body_pts_f) {
                auto rg = polygonResidualFast(
                    E, ctx.hull_lin->jac_ux.data(), ctx.hull_lin->jac_uy.data(), nullptr,
                    ctx.hull_lin->jac_vx.data(), ctx.hull_lin->jac_vy.data(), nullptr,
                    (double)p.x, (double)p.y);
                // signedDistWithNotch: ulkopuolella tai ilman loveusta -> d_outer sellaisenaan
                if (ctx.notch_lin != nullptr && rg.value > 0.0) {
                    if (!notch_built) {
                        notch_poly = evalLinearizedRing(*ctx.notch_lin, dX, dY, 0.0);
                        NE.build(notch_poly);
                        notch_built = true;
                    }
                    auto rn = polygonResidualFast(
                        NE, ctx.notch_lin->jac_ux.data(), ctx.notch_lin->jac_uy.data(), nullptr,
                        ctx.notch_lin->jac_vx.data(), ctx.notch_lin->jac_vy.data(), nullptr,
                        (double)p.x, (double)p.y);
                    if (rn.value > 0.0) {
                        rg.value = -rn.value; rg.dX = -rn.dX; rg.dY = -rn.dY;
                    } else if (-rn.value < rg.value) {
                        rg.value = -rn.value; rg.dX = -rn.dX; rg.dY = -rn.dY;
                    }
                }
                out.residuals.push_back(rg.value);
                out.J.push_back({ rg.dX, rg.dY, 0.0 });
            }
        } else {
            for (size_t i = 0; i < ctx.body_pts_f->size(); ++i) {
                out.residuals.push_back(1000.0);
                out.J.push_back({ 0.0, 0.0, 0.0 });
            }
        }
    }

    if (!ctx.ring_pts_f->empty()) {
        auto ring = evalLinearizedRing(*ctx.ring_lin, dX, dY, dR);
        PolyEdges E; E.build(ring);
        for (auto& p : *ctx.ring_pts_f) {
            auto rg = polygonResidualFast(
                E, ctx.ring_lin->jac_ux.data(), ctx.ring_lin->jac_uy.data(), ctx.ring_lin->jac_ur.data(),
                ctx.ring_lin->jac_vx.data(), ctx.ring_lin->jac_vy.data(), ctx.ring_lin->jac_vr.data(),
                (double)p.x, (double)p.y);
            out.residuals.push_back(rg.value);
            out.J.push_back({ rg.dX, rg.dY, rg.dR });
        }
    }

    return out;
}


static const double LM_REL_TOL = 1e-6;     // linearisoidun LM:n suhteellisen parannuksen pysaytysraja

static cv::Vec3d levenbergMarquardt3Linearized(
    const ResidualContext& ctx, cv::Vec3d params0,
    int max_iterations = 30, double lambda_init = 1e-3, double rel_tol = 1e-10)
{
    PT bpt2;
    LinearizedHull hull_lin = buildLinearizedHull(*ctx.local_pts_body, params0[0], params0[1], *ctx.K, *ctx.R, *ctx.t);
    LinearizedRing ring_lin;
    if (!ctx.ring_pts_f->empty())
        ring_lin = buildLinearizedRing(params0[0], params0[1], params0[2], ctx.ring_height_cm, *ctx.K, *ctx.R, *ctx.t);
    profAdd(P_LM_BUILD, bpt2.lap());

    const bool use_notch = !ctx.body_pts_f->empty() && ctx.handle_r_frac > 0.0;
    LinearizedRing notch_lin;
    if (use_notch)
        notch_lin = buildLinearizedRing(params0[0], params0[1], ctx.handle_r_frac * ctx.R_max_cm, ctx.ring_height_cm, *ctx.K, *ctx.R, *ctx.t, 28);
    LinearizedResidualContext lctx{ &hull_lin, &ring_lin, ctx.body_pts_f, ctx.ring_pts_f, params0[0], params0[1], params0[2],
                                    use_notch ? &notch_lin : nullptr };
    profAdd(P_C_NBODY, (long long)ctx.body_pts_f->size() * 1000000LL);
    profAdd(P_C_NRING, (long long)ctx.ring_pts_f->size() * 1000000LL);
    profAdd(P_C_NHULL, (long long)hull_lin.ref_pts.size() * 1000000LL);
    long long lm_iters = 0, lm_evals = 1;
    rel_tol = std::max(rel_tol, LM_REL_TOL);

    auto sumsq = [](const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return s;
    };

    cv::Vec3d params = params0;
    auto rj = jointResidualsAndJacobianLinearized(lctx, params);
    auto residuals = rj.residuals;
    auto J = rj.J;

    double cost = sumsq(residuals);
    double lam = lambda_init;

    for (int iter = 0; iter < max_iterations; ++iter) {
        ++lm_iters;

        size_t m = residuals.size();

        double JTJ[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        double JTr[3] = {0, 0, 0};

        for (size_t i = 0; i < m; ++i) {
            for (int a = 0; a < 3; ++a) {
                JTr[a] += J[i][(size_t)a] * residuals[i];
                for (int b = 0; b < 3; ++b)
                    JTJ[a][b] += J[i][(size_t)a] * J[i][(size_t)b];
            }
        }

        double diagv[3] = { JTJ[0][0] + 1e-12, JTJ[1][1] + 1e-12, JTJ[2][2] + 1e-12 };

        bool step_taken = false;
        double rel_improvement = 0.0;

        for (int inner = 0; inner < 12; ++inner) {

            double A[3][3];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    A[a][b] = JTJ[a][b];
            A[0][0] += lam * diagv[0];
            A[1][1] += lam * diagv[1];
            A[2][2] += lam * diagv[2];

            double bvec[3] = { -JTr[0], -JTr[1], -JTr[2] };
            double delta[3];

            if (!solve3x3(A, bvec, delta)) {
                lam *= 10.0;
                continue;
            }

            cv::Vec3d trial = params + cv::Vec3d(delta[0], delta[1], delta[2]);
            lm_evals += 1;
            PT ept;
            auto rj_new = jointResidualsAndJacobianLinearized(lctx, trial);
            profAdd(P_LM_EVAL_ONLY, ept.lap());
            double trial_cost = sumsq(rj_new.residuals);

            if (trial_cost < cost) {
                rel_improvement = (cost - trial_cost) / std::max(cost, 1e-12);
                params = trial;
                residuals = std::move(rj_new.residuals);
                J = std::move(rj_new.J);
                cost = trial_cost;
                lam = std::max(lam / 5.0, 1e-12);
                step_taken = true;
                break;
            }

            lam *= 5.0;
        }

        if (!step_taken || rel_improvement < rel_tol)
            break;
    }
    profAdd(P_C_LMITER, lm_iters * 1000000LL);
    profAdd(P_C_LMEVAL, lm_evals * 1000000LL);

    return params;
}


static cv::Vec3d levenbergMarquardt3(
    const ResidualContext& ctx, cv::Vec3d params0,
    int max_iterations = 30, double lambda_init = 1e-3, double rel_tol = 1e-10)
{
    cv::Vec3d params = params0;
    auto residuals = jointResiduals(ctx, params);

    auto sumsq = [](const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return s;
    };

    double cost = sumsq(residuals);
    double lam = lambda_init;
    const double eps = 1e-6;

    for (int iter = 0; iter < max_iterations; ++iter) {

        size_t m = residuals.size();
        std::vector<std::array<double, 3>> J(m);

        // runkopisteiden jaannos ei riipu renkaan sateesta: ilman rengaspisteita R-sarake on tasan nolla
        bool skip_r_column = ctx.ring_pts_f->empty();

        for (int j = 0; j < 3; ++j) {

            if (j == 2 && skip_r_column) {
                for (size_t i = 0; i < m; ++i)
                    J[i][2] = 0.0;
                continue;
            }

            double step = eps * std::max(1.0, std::abs(params[j]));
            cv::Vec3d p_plus = params;
            p_plus[j] += step;
            auto r_plus = jointResiduals(ctx, p_plus);
            for (size_t i = 0; i < m; ++i)
                J[i][(size_t)j] = (r_plus[i] - residuals[i]) / step;
        }

        double JTJ[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        double JTr[3] = {0, 0, 0};

        for (size_t i = 0; i < m; ++i) {
            for (int a = 0; a < 3; ++a) {
                JTr[a] += J[i][(size_t)a] * residuals[i];
                for (int b = 0; b < 3; ++b)
                    JTJ[a][b] += J[i][(size_t)a] * J[i][(size_t)b];
            }
        }

        double diagv[3] = { JTJ[0][0] + 1e-12, JTJ[1][1] + 1e-12, JTJ[2][2] + 1e-12 };

        bool step_taken = false;
        double rel_improvement = 0.0;

        for (int inner = 0; inner < 12; ++inner) {

            double A[3][3];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    A[a][b] = JTJ[a][b];
            A[0][0] += lam * diagv[0];
            A[1][1] += lam * diagv[1];
            A[2][2] += lam * diagv[2];

            double bvec[3] = { -JTr[0], -JTr[1], -JTr[2] };
            double delta[3];

            if (!solve3x3(A, bvec, delta)) {
                lam *= 10.0;
                continue;
            }

            cv::Vec3d trial = params + cv::Vec3d(delta[0], delta[1], delta[2]);
            auto trial_res = jointResiduals(ctx, trial);
            double trial_cost = sumsq(trial_res);

            if (trial_cost < cost) {
                rel_improvement = (cost - trial_cost) / std::max(cost, 1e-12);
                params = trial;
                residuals = trial_res;
                cost = trial_cost;
                lam = std::max(lam / 5.0, 1e-12);
                step_taken = true;
                break;
            }

            lam *= 5.0;
        }

        if (!step_taken || rel_improvement < rel_tol)
            break;
    }

    return params;
}


// Linearisoitu LM + tarkka varmistus: jos loppupisteen tarkka kustannus on suurempi kuin lahtopisteen, ajetaan tarkka LM
static cv::Vec3d levenbergMarquardt3LinearizedVerified(
    const ResidualContext& ctx, cv::Vec3d params0,
    int max_iterations = 30, double lambda_init = 1e-3, double rel_tol = 1e-10)
{
    PT vpt;
    cv::Vec3d params_lin = levenbergMarquardt3Linearized(ctx, params0, max_iterations, lambda_init, rel_tol);
    profAdd(P_LM_LIN, vpt.lap());

    auto sumsq = [](const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return s;
    };

    double exact_cost_start = sumsq(jointResiduals(ctx, params0));
    double exact_cost_lin = sumsq(jointResiduals(ctx, params_lin));

    profAdd(P_LM_VERIFY, vpt.lap());
    if (exact_cost_lin <= exact_cost_start * (1.0 + 1e-9))
        return params_lin;

    PT fpt;
    auto res_exact = levenbergMarquardt3(ctx, params0, max_iterations, lambda_init, rel_tol);
    profAdd(P_LM_FALLBACK, fpt.lap());
    return res_exact;
}


// ============================================================
// YHTEISSOVITUS: runkokontuuri + rengaspisteet -> LM (X, Y, renkaan sade), MAD-poikkeamien poisto ja uusi LM;
// toistetaan uusilla rengaspisteilla kunnes paikka ei enaa muutu (BOUNDARY_CONVERGENCE_CM). mask/sat ovat
// ROI-paikallisia (off_x/off_y), runko- ja rengaspisteet koko kuvan koordinaatistossa. tarkka = sovitus onnistui
// eika siirtynyt yli BOUNDARY_MAX_SHIFT_FROM_APPROX_CM lahtopisteesta.
// ============================================================
// YHTEISSOVITUS (kamera9_04.py:n refine_position_joint_fast) -
// mask_crop/sat_crop ROI-paikallisia, off_x/off_y siirtavat
// koko-framen ja ROI-paikallisen koordinaatiston valilla (body_pts/
// ring_pts pidetaan KOKO FRAMEN koordinaatistossa, kuten Python-
// versiossakin, koska projisointi/residuaalit kayttavat aina
// koko-framen pikselikoordinaatteja).
// ============================================================

struct RefineResult {
    double X_cm = 0.0, Y_cm = 0.0;
    bool tarkka = false;
    int n_body = 0, n_ring = 0;
    double rms_px = 0.0;
    bool has_rms = false;
    double ring_radius_cm = 0.0;
    bool has_ring_radius = false;
};


static RefineResult refinePositionJoint(
    const cv::Mat& mask_crop_roi, const cv::Mat& sat_crop_roi, int off_x_roi, int off_y_roi,
    int frame_w, int frame_h,
    const std::vector<cv::Point3d>& local_pts_body,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    double X0_approx, double Y0_approx)
{
    // Naytteet ROI:n ulkopuolelta (kuvan sisalla) ovat nollia: saturaatio luetaan nollareunustettuna nakymana
    // (200 px, PaddedView) ja maskiin riittaa 1 px:n nollareunus (findContours); kontuurin pisteet siirretaan
    // 200 px:n reunuksen koordinaatteihin.
    PT rpt;
    const int PAD = 200;
    int pad_left = std::max(0, std::min(PAD, off_x_roi));
    int pad_top = std::max(0, std::min(PAD, off_y_roi));
    int pad_right = std::max(0, std::min(PAD, frame_w - (off_x_roi + mask_crop_roi.cols)));
    int pad_bottom = std::max(0, std::min(PAD, frame_h - (off_y_roi + mask_crop_roi.rows)));

    const int mpl = std::min(1, pad_left), mpt = std::min(1, pad_top);
    const int mpr = std::min(1, pad_right), mpb = std::min(1, pad_bottom);
    cv::Mat mask_crop;
    cv::copyMakeBorder(mask_crop_roi, mask_crop, mpt, mpb, mpl, mpr, cv::BORDER_CONSTANT, cv::Scalar(0));
    const int mshift_x = pad_left - mpl, mshift_y = pad_top - mpt;     // maskin koordinaatit -> 200 px:n reunuksen koordinaatit
    PaddedView sat_crop{ &sat_crop_roi, pad_left, pad_top,
                         sat_crop_roi.cols + pad_left + pad_right, sat_crop_roi.rows + pad_top + pad_bottom };
    int off_x = off_x_roi - pad_left;
    int off_y = off_y_roi - pad_top;
    profAdd(P_R_PAD, rpt.lap());

    double ring_height_cm = H_total_cm;

    std::vector<cv::Point3d> approx3d{ cv::Point3d(X0_approx, Y0_approx, H_total_cm / 2.0) };
    auto approx_proj = project3d(K, R, t, approx3d);
    cv::Point2d approx_px_crop(approx_proj[0].x - off_x, approx_proj[0].y - off_y);


    std::vector<cv::Point> raw_contour;
    bool has_contour = findContourNear(mask_crop, approx_px_crop, raw_contour, BODY_CONTOUR_MAX_SEARCH_DIST_PX,
                                       BODY_CONTOUR_MIN_AREA_PX, cv::Point(mshift_x, mshift_y));

    std::vector<cv::Point2d> body_pts;
    if (has_contour) {
        auto body_pts_crop = filterIceBoundaryPoints(sat_crop, raw_contour);
        body_pts.resize(body_pts_crop.size());
        for (size_t i = 0; i < body_pts_crop.size(); ++i)
            body_pts[i] = cv::Point2d(body_pts_crop[i].x + off_x, body_pts_crop[i].y + off_y);
    }
    if (body_pts.size() > 40) {
        std::vector<cv::Point2d> dec;
        for (size_t i = 0; i < body_pts.size(); i += (size_t)LM_BODY_STRIDE) dec.push_back(body_pts[i]);
        body_pts.swap(dec);
    }
    int n_body = (int)body_pts.size();
    profAdd(P_R_CONTOUR, rpt.lap());

    // body_pts ei muutu ulompien iteraatioiden aikana -> float-muunnos kerran
    std::vector<cv::Point2f> body_pts_f = toPoint2fVec(body_pts);

    double X_cur = X0_approx, Y_cur = Y0_approx, R_cur = ring_r_frac_guess * R_max_cm;
    std::vector<cv::Point2d> ring_pts;
    double rms_px = 0.0;
    bool has_rms = false;

    for (int iter = 0; iter < BOUNDARY_MAX_ITERATIONS; ++iter) {

        std::vector<cv::Point3d> centers{
            cv::Point3d(X_cur, Y_cur, ring_height_cm),
            cv::Point3d(X_cur + R_cur, Y_cur, ring_height_cm),
        };
        auto cproj = project3d(K, R, t, centers);
        double cx_px = cproj[0].x, cy_px = cproj[0].y;
        double r_px_approx = std::hypot(cproj[1].x - cx_px, cproj[1].y - cy_px);

        ring_pts.clear();

        if (r_px_approx >= BOUNDARY_MIN_PREDICTED_RADIUS_PX) {
            auto cand = detectBoundaryPoints(sat_crop, cx_px - off_x, cy_px - off_y, r_px_approx);
            if ((int)cand.size() >= BOUNDARY_MIN_VALID_POINTS &&
                angularSpreadDeg(cand, cx_px - off_x, cy_px - off_y) >= BOUNDARY_MIN_ANGULAR_SPREAD_DEG) {
                ring_pts.resize(cand.size());
                for (size_t i = 0; i < cand.size(); ++i)
                    ring_pts[i] = cv::Point2d(cand[i].x + off_x, cand[i].y + off_y);
            }
        }

        int n_ring = (int)ring_pts.size();
        profAdd(P_R_BOUNDARY, rpt.lap());
        profAdd(P_R_OUTER_ITER, 0);

        std::vector<cv::Point2f> ring_pts_f = toPoint2fVec(ring_pts);

        if (n_body < BODY_MIN_VALID_POINTS && n_ring < BOUNDARY_MIN_VALID_POINTS) {
            RefineResult res;
            res.X_cm = X0_approx; res.Y_cm = Y0_approx; res.tarkka = false;
            res.n_body = n_body; res.n_ring = n_ring;
            res.has_rms = false; res.has_ring_radius = false;
            return res;
        }

        ResidualContext ctx{ &local_pts_body, &body_pts_f, &ring_pts_f, ring_height_cm, R_max_cm, handle_r_frac, &K, &R, &t };
        cv::Vec3d params_final = levenbergMarquardt3LinearizedVerified(ctx, cv::Vec3d(X_cur, Y_cur, R_cur), 30);
        profAdd(P_R_LM1, rpt.lap());

        std::vector<cv::Point2d> clean_body = body_pts;
        std::vector<cv::Point2d> clean_ring = ring_pts;

        if (n_body > 0) {
            auto body_resid = profileResiduals(local_pts_body, params_final[0], params_final[1], R_max_cm, H_total_cm, handle_r_frac, K, R, t, body_pts_f);
            double med = median(body_resid);
            std::vector<double> abs_dev(body_resid.size());
            for (size_t i = 0; i < body_resid.size(); ++i) abs_dev[i] = std::abs(body_resid[i] - med);
            double mad_b = median(abs_dev) + 1e-6;
            std::vector<cv::Point2d> inliers;
            for (size_t i = 0; i < body_resid.size(); ++i)
                if (std::abs(body_resid[i] - med) < 3.0 * 1.4826 * mad_b)
                    inliers.push_back(body_pts[i]);
            if ((int)inliers.size() >= BODY_MIN_VALID_POINTS)
                clean_body = inliers;
        }

        if (n_ring > 0) {
            auto ring_resid = ringPointResiduals(
                params_final[0], params_final[1], params_final[2], ring_height_cm, K, R, t, ring_pts_f
            );
            double med = median(ring_resid);
            std::vector<double> abs_dev(ring_resid.size());
            for (size_t i = 0; i < ring_resid.size(); ++i) abs_dev[i] = std::abs(ring_resid[i] - med);
            double mad_r = median(abs_dev) + 1e-6;
            std::vector<cv::Point2d> inliers;
            for (size_t i = 0; i < ring_resid.size(); ++i)
                if (std::abs(ring_resid[i] - med) < 3.0 * 1.4826 * mad_r)
                    inliers.push_back(ring_pts[i]);
            if ((int)inliers.size() >= BOUNDARY_MIN_VALID_POINTS)
                clean_ring = inliers;
        }

        std::vector<cv::Point2f> clean_body_f =
            (clean_body.size() == body_pts.size()) ? body_pts_f : toPoint2fVec(clean_body);
        std::vector<cv::Point2f> clean_ring_f =
            (clean_ring.size() == ring_pts.size()) ? ring_pts_f : toPoint2fVec(clean_ring);

        profAdd(P_R_MAD, rpt.lap());
        if (clean_body.size() != body_pts.size() || clean_ring.size() != ring_pts.size()) {
            ResidualContext ctx2{ &local_pts_body, &clean_body_f, &clean_ring_f, ring_height_cm, R_max_cm, handle_r_frac, &K, &R, &t };
            params_final = levenbergMarquardt3LinearizedVerified(ctx2, params_final, 30);
            profAdd(P_R_LM2, rpt.lap());
        }

        ResidualContext ctx_final{ &local_pts_body, &clean_body_f, &clean_ring_f, ring_height_cm, R_max_cm, handle_r_frac, &K, &R, &t };
        auto resid_final = jointResiduals(ctx_final, params_final);

        if (!resid_final.empty()) {
            double s = 0.0;
            for (double v : resid_final) s += v * v;
            rms_px = std::sqrt(s / (double)resid_final.size());
            has_rms = true;
        } else {
            has_rms = false;
        }

        profAdd(P_R_FINAL, rpt.lap());
        double X_new = params_final[0], Y_new = params_final[1], R_new = std::abs(params_final[2]);
        double moved = std::hypot(X_new - X_cur, Y_new - Y_cur);
        X_cur = X_new; Y_cur = Y_new; R_cur = R_new;

        if (moved < BOUNDARY_CONVERGENCE_CM)
            break;
    }

    double total_shift = std::hypot(X_cur - X0_approx, Y_cur - Y0_approx);

    RefineResult res;

    if (total_shift > BOUNDARY_MAX_SHIFT_FROM_APPROX_CM) {
        res.X_cm = X0_approx; res.Y_cm = Y0_approx; res.tarkka = false;
        res.n_body = n_body; res.n_ring = (int)ring_pts.size();
        res.rms_px = rms_px; res.has_rms = has_rms;
        res.has_ring_radius = false;
        return res;
    }

    res.X_cm = X_cur; res.Y_cm = Y_cur; res.tarkka = true;
    res.n_body = n_body; res.n_ring = (int)ring_pts.size();
    res.rms_px = rms_px; res.has_rms = has_rms;
    res.ring_radius_cm = R_cur; res.has_ring_radius = true;

    return res;
}


