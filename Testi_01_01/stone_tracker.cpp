// M_PI (kaytetaan mm. kulmalaskennassa alempana) EI ole standardin
// C++ osa - MSVC:n <cmath> paljastaa sen VAIN jos _USE_MATH_DEFINES
// on maaritelty ENNEN ensimmaista <cmath>-includea (myos transitiivisia,
// esim. OpenCV/pybind11:n omien headereiden kautta) - siksi tama on
// TIEDOSTON ENSIMMAINEN rivi. GCC/Linux paljastaa M_PI:n oletuksena
// ilman tata, minka vuoksi tama puuttui alkuperaisesta Linux-
// kehitysversiosta.
#define _USE_MATH_DEFINES

// ============================================================
// stone_tracker.cpp - Testi_01_01
//
// C++-porttaus SEURANTA-vaiheen kuumasta polusta (Task 5, kayttajan
// pyynnosta): per-kiven paivitys (ROI-maski+saturaatio, ristikkohaku,
// LM-yhteissovitus) on profiloitu Python-versiossa n. 77ms/kivi -
// nelja samanaikaista kivea = n. 310ms/frame, kaukana 40ms budjetista
// 25fps:lle. Tama tiedosto EI MUUTA MATEMATIIKKAA/ALGORITMIA -
// se ratkaisee TASMALLEEN saman ristikkohaun ja LM-sovituksen samoilla
// pisteilla ja samoilla iteraatioilla kuin kamera9_04.py:n
// locate_by_grid_search_fast + refine_position_joint_fast (jotka
// itse ovat kamera9_02/03.py:n TASMALLEEN sama matematiikka, vain
// nopeutettuina - katso naiden tiedostojen omat kommentit), vain
// C++:lla laskettuna JA rinnakkaistettuna std::thread:eilla usealle
// samanaikaiselle kivelle.
//
// EI KOSKETA mode_engine.cpp:aa (kalibroinnin/moodikuvan jo validoitu
// koodi) - tama on OMA, erillinen pybind11-moduuli.
//
// Portattu (katso Python-vastineet):
//   kamera9_01.py: create_granite_mask, _project_3d
//   kamera9_02.py: _hull_overlap_score
//   kamera9_03.py: _find_contour_near, _ring_point_residuals,
//                  _angular_spread_deg
//   kamera9_04.py: predicted_stone_hull_fast, profile_residuals_fast,
//                  _score_candidate_fast, _grid_search_best_fast,
//                  locate_by_grid_search_fast, _bilinear_sample_vec,
//                  detect_boundary_points_fast,
//                  filter_ice_boundary_points_fast,
//                  refine_position_joint_fast, _track_roi_bounds,
//                  create_granite_mask_roi, compute_sat_roi
//   kamera8_01.py: _levenberg_marquardt (tassa erikoistettu 3
//                  parametrille - sama numeerinen Jacobian+LM-
//                  periaate, ei mitaan matemaattista eroa)
//
// local_pts_body/local_pts_search (kiven OMA, profiilista riippuva
// paikallinen rengasgeometria) EI porattu tanne - ne lasketaan
// Python-puolella KERRAN (kamera9_04.py:n build_local_stone_rings,
// koskematon) ja annetaan tanne valmiina pistetaulukkoina, koska
// niiden Catmull-Rom-splini-rakennus ei ole per-frame-kuuma polku.
//
// TUNNETTU, ALGORITMIN OMA (EI PORTTAUSVIRHE) NUMEERINEN HERKKYYS:
// numeerisessa validoinnissa (400 SEURANTA-framea oikealta videolta,
// Python-tulos vs. tama C++-versio TASMALLEEN samoilla syotteilla)
// ristikkohaku+peitto-osuus TASMAA AINA (0 poikkeamaa pisteytyksessa,
// koska hullOverlapScore:n int32-typistys - EI pyoristys, katso alla -
// ja maskin laskenta ovat bittitarkasti Pythonin kanssa samat). refine_
// position_joint:in LM-sovitus kuitenkin paatyy n. 40%:ssa framejsta
// LAHELLA MUTTA EI TASMALLEEN samaan (X,Y)-pisteeseen kuin Python -
// molemmat sovitukset ovat YHTA HYVIA (rms_px lahes identtinen,
// esim. 0.53 vs 0.57px pahimmassakin havaitussa tapauksessa, vaikka
// (X,Y)-ero oli 24cm) koska refine_position_joint_fast:in MAD-pohjainen
// poikkeavien hylkays (3.0*1.4826*MAD kova raja-arvo) on ITSESSAAN
// EPAJATKUVA funktio residuaaleista - jos yksi piste on TASMALLEEN
// rajalla, koneepsilonin tasoinen ero (esim. numpy/LAPACK:in
// np.linalg.solve vs. taman tiedoston oma Gaussin eliminaatio
// osittaisella pivotoinnilla solve3x3:ssa, tai cv::pointPolygonTest:in
// liukulukusummauksen jarjestys) voi kaantaa taman pisteen sisa-/
// ulkopuolelle-paatoksen, mika johtaa LM:n konvergoitumiseen ERI
// (mutta yhta patevaan) lahialueen minimiin. Tama EI ole korjattavissa
// muuttamatta itse algoritmia (esim. pehmentamalla kynnysarvoa), mika
// on nimenomaisesti KIELLETTY (katso ylla) - tama on Pythonin OMAN
// algoritmin (kamera8_01.py:n _levenberg_marquardt + refine_position_
// joint_fast:in MAD-hylkays) sisaanrakennettu herkkyys, joka nayttaytyy
// vasta eri kielella/kirjastolla lasketun liukulukuaritmetiikan kautta.
// ============================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

namespace py = pybind11;


// ============================================================
// VAKIOT - tasmalleen kamera9_01/02/03.py:sta (katso niiden omat
// kommentit perusteluista - ei muutettu tassa).
// ============================================================

static const int STONE_MAX_SATURATION = 60;
static const double STONE_MIN_DARKNESS = 15.0;
static const double STONE_DARKNESS_SIGMA = 25.0;

static const int BOUNDARY_N_ANGLES = 72;
static const double BOUNDARY_SATURATION_THRESHOLD = 70.0;
static const double BOUNDARY_RADIAL_STEP_PX = 0.25;
static const double BOUNDARY_RADIUS_SEARCH_FACTOR_MIN = 0.2;
static const double BOUNDARY_RADIUS_SEARCH_FACTOR_MAX = 2.6;
static const double BOUNDARY_MIN_PREDICTED_RADIUS_PX = 4.0;
static const int BOUNDARY_MIN_VALID_POINTS = 12;
static const double BOUNDARY_MIN_ANGULAR_SPREAD_DEG = 180.0;
static const int BOUNDARY_MAX_ITERATIONS = 5;
static const double BOUNDARY_CONVERGENCE_CM = 0.05;
static const double BOUNDARY_MAX_SHIFT_FROM_APPROX_CM = 25.0;

static const double BODY_CONTOUR_MIN_AREA_PX = 15.0;
static const double BODY_CONTOUR_MAX_SEARCH_DIST_PX = 60.0;
static const double BODY_HANDLE_CHECK_DIST_PX = 2.5;
static const int BODY_MIN_VALID_POINTS = 8;

static const int MASK_ROI_MARGIN_PX = 150;


// ============================================================
// KAMERAPROJEKTIO (kamera9_01.py:n _project_3d)
// ============================================================

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


// ============================================================
// KIVEN ENNUSTETTU AARIVIIVA (kamera9_04.py:n
// predicted_stone_hull_fast) - siirto + projisointi + kupera peite
// valmiiksi lasketulle paikalliselle pistejoukolle.
// ============================================================

static std::vector<cv::Point2f> predictedHull(
    const std::vector<cv::Point3d>& local_pts, double X0, double Y0,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t)
{
    std::vector<cv::Point3d> pts(local_pts.size());

    for (size_t i = 0; i < local_pts.size(); ++i)
        pts[i] = cv::Point3d(local_pts[i].x + X0, local_pts[i].y + Y0, local_pts[i].z);

    auto proj = project3d(K, R, t, pts);

    std::vector<cv::Point2f> proj_f(proj.size());
    bool all_finite = true;

    for (size_t i = 0; i < proj.size(); ++i) {
        if (!std::isfinite(proj[i].x) || !std::isfinite(proj[i].y))
            all_finite = false;
        proj_f[i] = cv::Point2f((float)proj[i].x, (float)proj[i].y);
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
// PEITTO-OSUUS (kamera9_02.py:n _hull_overlap_score) - hull-pisteet
// annetaan KOKO FRAMEN koordinaatistossa, off_x/off_y siirtaa ne
// ROI-rajatun maskin paikalliseen koordinaatistoon.
// ============================================================

static double hullOverlapScore(
    const cv::Mat& mask_crop, const std::vector<cv::Point2f>& hull,
    int off_x, int off_y)
{
    if (hull.size() < 3)
        return 0.0;

    // Python: hull_int = hull.astype(np.int32) - TRUNKOI kohti nollaa
    // (EI PYORISTA lahimpaan, kuten C:n (int)-cast) KOKO FRAMEN
    // koordinaatistossa ENNEN mitaan ROI-siirtoa (Pythonissa mask on
    // taman funktion nakokulmasta aina koko-framen kokoinen, nolla-
    // taytetty taulukko - katso create_granite_mask_roi/compute_sat_roi).
    // off_x/off_y ovat kokonaislukuja, mutta trunc(x-k) != trunc(x)-k
    // yleisesti kun x-k voi ylittaa nollan (esim. x=0.3,k=1: trunc(x-k)=
    // trunc(-0.7)=0, mutta trunc(x)-k=0-1=-1) - siksi TRUNKOINTI ON
    // TEHTAVA ENNEN off_x/off_y-vahennysta, ei jalkeen, jotta tama
    // tasmaa tasmalleen Pythonin tulokseen.
    std::vector<cv::Point> hull_int(hull.size());
    for (size_t i = 0; i < hull.size(); ++i)
        hull_int[i] = cv::Point((int)(hull[i].x), (int)(hull[i].y));
    for (auto& p : hull_int) { p.x -= off_x; p.y -= off_y; }

    cv::Rect bbox = cv::boundingRect(hull_int);

    int x0 = std::max(bbox.x, 0);
    int y0 = std::max(bbox.y, 0);
    int x1 = std::min(bbox.x + bbox.width, mask_crop.cols);
    int y1 = std::min(bbox.y + bbox.height, mask_crop.rows);

    if (x1 <= x0 || y1 <= y0)
        return 0.0;

    std::vector<cv::Point> shifted(hull_int.size());
    for (size_t i = 0; i < hull_int.size(); ++i)
        shifted[i] = cv::Point(hull_int[i].x - x0, hull_int[i].y - y0);

    cv::Mat canvas = cv::Mat::zeros(y1 - y0, x1 - x0, CV_8UC1);
    std::vector<std::vector<cv::Point>> polys{shifted};
    cv::fillPoly(canvas, polys, cv::Scalar(255));

    int hull_area = cv::countNonZero(canvas);
    if (hull_area == 0)
        return 0.0;

    cv::Mat mask_sub = mask_crop(cv::Rect(x0, y0, x1 - x0, y1 - y0));

    int overlap = 0;
    for (int r = 0; r < canvas.rows; ++r) {
        const uchar* cptr = canvas.ptr<uchar>(r);
        const uchar* mptr = mask_sub.ptr<uchar>(r);
        for (int c = 0; c < canvas.cols; ++c)
            if (cptr[c] > 0 && mptr[c] > 0)
                ++overlap;
    }

    return (double)overlap / (double)hull_area;
}


// ============================================================
// RISTIKKOHAKU (kamera9_04.py:n _grid_search_best_fast +
// locate_by_grid_search_fast) - karkea, sitten hieno.
// ============================================================

static std::vector<double> arangeVec(double start, double stop, double step)
{
    // Vastaa numpy.arange:a: pituus = ceil((stop-start)/step),
    // arvot start + i*step.
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


static std::pair<cv::Point2d, double> gridSearchBest(
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Mat& mask_crop, int off_x, int off_y,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    const std::vector<double>& x_vals, const std::vector<double>& y_vals)
{
    double best_score = -1.0;
    cv::Point2d best_xy(x_vals.empty() ? 0.0 : x_vals[0], y_vals.empty() ? 0.0 : y_vals[0]);

    for (double X : x_vals) {
        for (double Y : y_vals) {
            auto hull = predictedHull(local_pts_search, X, Y, K, R, t);
            double score = hullOverlapScore(mask_crop, hull, off_x, off_y);
            if (score > best_score) {
                best_score = score;
                best_xy = cv::Point2d(X, Y);
            }
        }
    }

    return {best_xy, best_score};
}


static std::pair<cv::Point2d, double> locateByGridSearchFast(
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Mat& mask_crop, int off_x, int off_y,
    double x_center, double x_half_range, double y_center, double y_half_range,
    double coarse_step, double fine_step,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t)
{
    auto x_vals = arangeVec(x_center - x_half_range, x_center + x_half_range + 1e-6, coarse_step);
    auto y_vals = arangeVec(y_center - y_half_range, y_center + y_half_range + 1e-6, coarse_step);
    auto best1 = gridSearchBest(local_pts_search, mask_crop, off_x, off_y, K, R, t, x_vals, y_vals);

    auto x_vals2 = arangeVec(best1.first.x - coarse_step, best1.first.x + coarse_step + 1e-6, fine_step);
    auto y_vals2 = arangeVec(best1.first.y - coarse_step, best1.first.y + coarse_step + 1e-6, fine_step);
    auto best2 = gridSearchBest(local_pts_search, mask_crop, off_x, off_y, K, R, t, x_vals2, y_vals2);

    return best2;
}


// ============================================================
// GRANIITTIMASKI + SATURAATIO (kamera9_01.py:n create_granite_mask,
// kamera9_04.py:n compute_sat_roi:n vastine) - laskettuna suoraan
// PIENELLE crop:lle (ei koko-frame-kokoista nollattua taulukkoa,
// toisin kuin Python-versio - katso kamera9_04.py:n oma kommentti
// MASK_ROI_MARGIN_PX:sta: talla ei ole vaikutusta TULOKSEEN koska
// kaikki kaytto tapahtuu joka tapauksessa ROI:n sisalla, mutta
// saastaa turhan koko-framen nollaus/varauksen).
//
// HUOM KOKO FRAMEN (HAKU) SUORITUSKYVYSTA: taman funktion sisalta
// kutsuttavat cv::GaussianBlur/morphologyEx KOKO 1920x1080-framelle
// (search_new_stone, ei-ROI-rajattu HAKU) osoittautuivat Linux-
// kehitysymparistossa YLLATTAEN HITAAMMIKSI kuin Pythonin cv2-
// vastine (n. 180ms vs. Pythonin 89ms) - EI porttausvirhe, vaan
// tama kehitysympariston oma libopencv4.6 (apt/pkg-config) puuttuu
// Intel IPP -kiihdytyksen, kun taas Pythonin pip-asennettu cv2-paketti
// tuo OMAN, IPP-kiihdytetyn OpenCV 5.0 -kirjastonsa mukana (katso
// cv2.getBuildInformation() molemmin puolin - eri versio, eri IPP-tila).
// TARKISTA tama sama asia Windows-kaannoksessa (linkitetaanko IPP-
// kiihdytettya OpenCV:ta, esim. vcpkg:n tai virallisen prebuilt-SDK:n
// versio - nama yleensa SISALTAVAT IPP:n oletuksena) ennen kuin
// vertailet HAKU:n C++/Python-nopeuksia omalla koneella - Linux-
// kehitysymparistossa mitattu ~1.3x nopeutus HAKU:lle EI todennakoisesti
// vastaa oikeaa kannettavuutta jos linkitetty OpenCV eroaa nain.
// ============================================================

static cv::Mat createGraniteMask(const cv::Mat& frame_bgr)
{
    cv::Mat hsv, gray, gray_f, bg, darkness;

    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray_f, CV_32F);
    cv::GaussianBlur(gray_f, bg, cv::Size(0, 0), STONE_DARKNESS_SIGMA);
    darkness = bg - gray_f;

    std::vector<cv::Mat> hsv_ch;
    cv::split(hsv, hsv_ch);
    const cv::Mat& sat_ch = hsv_ch[1];

    cv::Mat mask = cv::Mat::zeros(frame_bgr.size(), CV_8UC1);

    for (int r = 0; r < mask.rows; ++r) {
        const uchar* satp = sat_ch.ptr<uchar>(r);
        const float* darkp = darkness.ptr<float>(r);
        uchar* mp = mask.ptr<uchar>(r);
        for (int c = 0; c < mask.cols; ++c) {
            bool low_saturation = satp[c] < STONE_MAX_SATURATION;
            bool dark_enough = darkp[c] > (float)STONE_MIN_DARKNESS;
            mp[c] = (low_saturation && dark_enough) ? 255 : 0;
        }
    }

    cv::Mat kernel_open = cv::Mat::ones(5, 5, CV_8U);
    cv::Mat kernel_close = cv::Mat::ones(3, 3, CV_8U);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_close);

    return mask;
}


static cv::Mat computeSat(const cv::Mat& frame_bgr)
{
    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    cv::Mat sat_f;
    ch[1].convertTo(sat_f, CV_32F);
    return sat_f;
}


// ============================================================
// TAUSTAN VAIMENNUS (main.py:n suppress_static_background, Task 6
// - HAKU:n C++-porttaus). Peittaa (valkoisella) alueet jotka
// vastaavat kalibroinnin puhdasta staattista referenssikuvaa,
// jottei painettu/staattinen sisalto (sponsoritekstit, viivat)
// voi tulla virhetunnistetuksi kiveksi HAKU-vaiheessa - katso
// main.py:n oma kommentti taman alkuperaisesta motivaatiosta.
// ============================================================

static cv::Mat suppressStaticBackground(
    const cv::Mat& frame_bgr, const cv::Mat& reference_bgr, double diff_threshold)
{
    if (reference_bgr.empty() ||
        frame_bgr.rows != reference_bgr.rows ||
        frame_bgr.cols != reference_bgr.cols ||
        frame_bgr.type() != reference_bgr.type())
        return frame_bgr.clone();

    cv::Mat diff, diff_gray;
    cv::absdiff(frame_bgr, reference_bgr, diff);
    cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);

    cv::Mat out = frame_bgr.clone();

    for (int r = 0; r < out.rows; ++r) {
        const uchar* dptr = diff_gray.ptr<uchar>(r);
        cv::Vec3b* optr = out.ptr<cv::Vec3b>(r);
        for (int c = 0; c < out.cols; ++c) {
            if ((double)dptr[c] < diff_threshold)
                optr[c] = cv::Vec3b(255, 255, 255);
        }
    }

    return out;
}


// ============================================================
// ROI-RAJAUS (kamera9_04.py:n _track_roi_bounds)
// ============================================================

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

    int x0 = (int)std::floor(minu) - margin_px;
    int x1 = (int)std::ceil(maxu) + margin_px;
    int y0 = (int)std::floor(minv) - margin_px;
    int y1 = (int)std::ceil(maxv) + margin_px;

    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    x1 = std::min(x1, frame_w);
    y1 = std::min(y1, frame_h);

    if (x1 < x0) x1 = x0;
    if (y1 < y0) y1 = y0;

    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}


// ============================================================
// BILINEAARINEN NAYTTO (kamera9_04.py:n _bilinear_sample_vec, tassa
// yksi piste kerrallaan - C++-silmukka on jo tarpeeksi nopea, ei
// tarvitse numpy-tyylista vektorointia).
// ============================================================

static bool bilinearSample(const cv::Mat& channel, double x, double y, double& out)
{
    int w = channel.cols, h = channel.rows;
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    int x1 = x0 + 1, y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h)
        return false;

    double fx = x - x0, fy = y - y0;
    double v00 = channel.at<float>(y0, x0);
    double v10 = channel.at<float>(y0, x1);
    double v01 = channel.at<float>(y1, x0);
    double v11 = channel.at<float>(y1, x1);

    out = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
          v01 * (1 - fx) * fy + v11 * fx * fy;

    return true;
}


// ============================================================
// RENGASPISTEIDEN ALIPIKSELIHAVAINNOT (kamera9_04.py:n
// detect_boundary_points_fast)
// ============================================================

static std::vector<cv::Point2d> detectBoundaryPoints(
    const cv::Mat& sat, double cx_px, double cy_px, double r_px_approx,
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


// ============================================================
// LAHIN KONTUURI (kamera9_03.py:n _find_contour_near) +
// KAHVASUODATUS (kamera9_04.py:n filter_ice_boundary_points_fast)
// ============================================================

static bool findContourNear(
    const cv::Mat& mask, cv::Point2d approx_px, std::vector<cv::Point>& best,
    double max_dist_px = BODY_CONTOUR_MAX_SEARCH_DIST_PX,
    double min_area = BODY_CONTOUR_MIN_AREA_PX)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

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

        if (d <= max_dist_px && (!found || d < best_dist)) {
            best = c;
            best_dist = d;
            found = true;
        }
    }

    return found;
}


static std::vector<cv::Point2d> filterIceBoundaryPoints(
    const cv::Mat& sat, const std::vector<cv::Point>& contour,
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


// ============================================================
// JAANNOSVIRHEET (kamera9_04.py:n profile_residuals_fast,
// kamera9_03.py:n _ring_point_residuals)
// ============================================================

static std::vector<double> profileResiduals(
    const std::vector<cv::Point3d>& local_pts_body, double X0, double Y0,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    const std::vector<cv::Point2d>& pts)
{
    auto hull = predictedHull(local_pts_body, X0, Y0, K, R, t);

    std::vector<double> out(pts.size());

    if (hull.empty()) {
        std::fill(out.begin(), out.end(), 1000.0);
        return out;
    }

    for (size_t i = 0; i < pts.size(); ++i)
        out[i] = cv::pointPolygonTest(hull, cv::Point2f((float)pts[i].x, (float)pts[i].y), true);

    return out;
}


static std::vector<cv::Point2f> predictedRingPoints(
    double X0, double Y0, double ring_radius_cm, double ring_height_cm,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    int n_theta = 120)
{
    std::vector<cv::Point3d> pts3d((size_t)n_theta);
    for (int i = 0; i < n_theta; ++i) {
        double theta = 2.0 * M_PI * (double)i / (double)n_theta;
        pts3d[(size_t)i] = cv::Point3d(
            X0 + ring_radius_cm * std::cos(theta),
            Y0 + ring_radius_cm * std::sin(theta),
            ring_height_cm
        );
    }

    auto proj = project3d(K, R, t, pts3d);

    std::vector<cv::Point2f> out((size_t)n_theta);
    for (int i = 0; i < n_theta; ++i)
        out[(size_t)i] = cv::Point2f((float)proj[(size_t)i].x, (float)proj[(size_t)i].y);

    return out;
}


static std::vector<double> ringPointResiduals(
    double X0, double Y0, double ring_radius_cm, double ring_height_cm,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    const std::vector<cv::Point2d>& observed)
{
    auto poly = predictedRingPoints(X0, Y0, ring_radius_cm, ring_height_cm, K, R, t);

    std::vector<double> out(observed.size());
    for (size_t i = 0; i < observed.size(); ++i)
        out[i] = cv::pointPolygonTest(poly, cv::Point2f((float)observed[i].x, (float)observed[i].y), true);

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


// ============================================================
// LEVENBERG-MARQUARDT (kamera8_01.py:n _levenberg_marquardt,
// erikoistettu 3 parametrille X,Y,R - sama numeerinen Jacobian +
// vaimennettu pienimman nelion periaate, ei matemaattista eroa).
// ============================================================

struct ResidualContext {
    const std::vector<cv::Point3d>* local_pts_body;
    const std::vector<cv::Point2d>* body_pts;
    const std::vector<cv::Point2d>* ring_pts;
    double ring_height_cm;
    const cv::Matx33d* K;
    const cv::Matx33d* R;
    const cv::Vec3d* t;
};


static std::vector<double> jointResiduals(const ResidualContext& ctx, const cv::Vec3d& params)
{
    double X = params[0], Y = params[1], Rr = params[2];
    std::vector<double> out;

    if (!ctx.body_pts->empty()) {
        auto r = profileResiduals(*ctx.local_pts_body, X, Y, *ctx.K, *ctx.R, *ctx.t, *ctx.body_pts);
        out.insert(out.end(), r.begin(), r.end());
    }

    if (!ctx.ring_pts->empty()) {
        auto r = ringPointResiduals(X, Y, Rr, ctx.ring_height_cm, *ctx.K, *ctx.R, *ctx.t, *ctx.ring_pts);
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

        for (int j = 0; j < 3; ++j) {
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
    double R_max_cm, double H_total_cm, double ring_r_frac_guess,
    double X0_approx, double Y0_approx)
{
    // Python: mask/sat ANNETAAN AINA koko-framen kokoisina taulukkoina
    // (nollataytettyja ROI:n ulkopuolella - katso create_granite_mask_roi/
    // compute_sat_roi), joten mika tahansa piste TODELLISEN FRAMEN
    // sisalla on aina laillinen naytepiste Pythonissa (arvo 0 jos ROI:n
    // ulkopuolella), vaikka se olisi taman ROI-rajatun crop:in OMAN
    // pienen taulukon ulkopuolella. Jaljitellaan tama tarkalleen
    // tayttamalla crop nollareunuksella (leikattuna todellisiin frame-
    // rajoihin asti, aivan kuten Pythonin oikea taulukko olisi) ennen
    // kontuuri-/rengashakua - muuten bilineaarinaytto hylkaisi Pythonissa
    // KELVOLLISIA (ROI:n reunan lahella olevia) pisteita vain koska ne
    // ovat taman PIENEN crop:in omien pikselirajojen ulkopuolella.
    const int PAD = 200;
    int pad_left = std::max(0, std::min(PAD, off_x_roi));
    int pad_top = std::max(0, std::min(PAD, off_y_roi));
    int pad_right = std::max(0, std::min(PAD, frame_w - (off_x_roi + mask_crop_roi.cols)));
    int pad_bottom = std::max(0, std::min(PAD, frame_h - (off_y_roi + mask_crop_roi.rows)));

    cv::Mat mask_crop, sat_crop;
    cv::copyMakeBorder(mask_crop_roi, mask_crop, pad_top, pad_bottom, pad_left, pad_right,
                        cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::copyMakeBorder(sat_crop_roi, sat_crop, pad_top, pad_bottom, pad_left, pad_right,
                        cv::BORDER_CONSTANT, cv::Scalar(0));
    int off_x = off_x_roi - pad_left;
    int off_y = off_y_roi - pad_top;

    double ring_height_cm = H_total_cm;

    std::vector<cv::Point3d> approx3d{ cv::Point3d(X0_approx, Y0_approx, H_total_cm / 2.0) };
    auto approx_proj = project3d(K, R, t, approx3d);
    cv::Point2d approx_px_crop(approx_proj[0].x - off_x, approx_proj[0].y - off_y);

    std::vector<cv::Point> raw_contour;
    bool has_contour = findContourNear(mask_crop, approx_px_crop, raw_contour);

    std::vector<cv::Point2d> body_pts;
    if (has_contour) {
        auto body_pts_crop = filterIceBoundaryPoints(sat_crop, raw_contour);
        body_pts.resize(body_pts_crop.size());
        for (size_t i = 0; i < body_pts_crop.size(); ++i)
            body_pts[i] = cv::Point2d(body_pts_crop[i].x + off_x, body_pts_crop[i].y + off_y);
    }
    int n_body = (int)body_pts.size();

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

        if (n_body < BODY_MIN_VALID_POINTS && n_ring < BOUNDARY_MIN_VALID_POINTS) {
            RefineResult res;
            res.X_cm = X0_approx; res.Y_cm = Y0_approx; res.tarkka = false;
            res.n_body = n_body; res.n_ring = n_ring;
            res.has_rms = false; res.has_ring_radius = false;
            return res;
        }

        ResidualContext ctx{ &local_pts_body, &body_pts, &ring_pts, ring_height_cm, &K, &R, &t };
        cv::Vec3d params_final = levenbergMarquardt3(ctx, cv::Vec3d(X_cur, Y_cur, R_cur), 30);

        std::vector<cv::Point2d> clean_body = body_pts;
        std::vector<cv::Point2d> clean_ring = ring_pts;

        if (n_body > 0) {
            auto body_resid = profileResiduals(local_pts_body, params_final[0], params_final[1], K, R, t, body_pts);
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
                params_final[0], params_final[1], params_final[2], ring_height_cm, K, R, t, ring_pts
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

        if (clean_body.size() != body_pts.size() || clean_ring.size() != ring_pts.size()) {
            ResidualContext ctx2{ &local_pts_body, &clean_body, &clean_ring, ring_height_cm, &K, &R, &t };
            params_final = levenbergMarquardt3(ctx2, params_final, 30);
        }

        ResidualContext ctx_final{ &local_pts_body, &clean_body, &clean_ring, ring_height_cm, &K, &R, &t };
        auto resid_final = jointResiduals(ctx_final, params_final);

        if (!resid_final.empty()) {
            double s = 0.0;
            for (double v : resid_final) s += v * v;
            rms_px = std::sqrt(s / (double)resid_final.size());
            has_rms = true;
        } else {
            has_rms = false;
        }

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


// ============================================================
// PYBIND11-RAJAPINTA
// ============================================================

static std::vector<cv::Point3d> parsePts3(py::array_t<double, py::array::c_style | py::array::forcecast>& arr)
{
    auto b = arr.unchecked<2>();
    std::vector<cv::Point3d> out((size_t)b.shape(0));
    for (py::ssize_t i = 0; i < b.shape(0); ++i)
        out[(size_t)i] = cv::Point3d(b(i, 0), b(i, 1), b(i, 2));
    return out;
}


static cv::Matx33d parseMat33(py::array_t<double, py::array::c_style | py::array::forcecast>& arr)
{
    auto b = arr.unchecked<2>();
    cv::Matx33d M;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j) = b(i, j);
    return M;
}


static cv::Vec3d parseVec3(py::array_t<double, py::array::c_style | py::array::forcecast>& arr)
{
    auto b = arr.unchecked<1>();
    return cv::Vec3d(b(0), b(1), b(2));
}


struct StoneUpdateResult {
    double score = 0.0;
    bool has_position = false;
    RefineResult refined;
};


static StoneUpdateResult trackStoneUpdateOne(
    const cv::Mat& frame_mat,
    const std::vector<cv::Point3d>& local_pts_body,
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double X0, double Y0,
    double track_half_range_cm, double coarse_step_cm, double fine_step_cm,
    double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess)
{
    int frame_w = frame_mat.cols, frame_h = frame_mat.rows;

    double roi_half_range = track_half_range_cm + BOUNDARY_MAX_SHIFT_FROM_APPROX_CM;
    cv::Rect roi = trackRoiBounds(X0, Y0, roi_half_range, R_max_cm, H_total_cm, frame_w, frame_h, K, R, t);

    StoneUpdateResult out;

    if (roi.width <= 0 || roi.height <= 0)
        return out;

    cv::Mat crop = frame_mat(roi);
    cv::Mat mask_crop = createGraniteMask(crop);
    cv::Mat sat_crop = computeSat(crop);

    auto best = locateByGridSearchFast(
        local_pts_search, mask_crop, roi.x, roi.y,
        X0, track_half_range_cm, Y0, track_half_range_cm,
        coarse_step_cm, fine_step_cm, K, R, t
    );

    out.score = best.second;

    if (out.score < score_threshold)
        return out;

    out.has_position = true;
    out.refined = refinePositionJoint(
        mask_crop, sat_crop, roi.x, roi.y, frame_w, frame_h, local_pts_body, K, R, t,
        R_max_cm, H_total_cm, ring_r_frac_guess, best.first.x, best.first.y
    );

    return out;
}


static py::dict resultToDict(const StoneUpdateResult& r)
{
    py::dict d;
    d["score"] = r.score;
    d["found"] = r.has_position;

    if (r.has_position) {
        d["X_cm"] = r.refined.X_cm;
        d["Y_cm"] = r.refined.Y_cm;
        d["tarkka"] = r.refined.tarkka;
        d["n_body"] = r.refined.n_body;
        d["n_ring"] = r.refined.n_ring;
        d["rms_px"] = r.refined.has_rms ? py::cast(r.refined.rms_px) : py::cast(nullptr);
        d["ring_radius_cm"] = r.refined.has_ring_radius ? py::cast(r.refined.ring_radius_cm) : py::cast(nullptr);
    }

    return d;
}


static py::dict track_stone_update(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_u,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_body_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_search_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr,
    double X0, double Y0,
    double track_half_range_cm, double coarse_step_cm, double fine_step_cm,
    double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess)
{
    auto buf = frame_u.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("frame_u must be HxWx3 uint8 BGR");

    cv::Mat frame_mat((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, (void*)buf.ptr);

    auto local_pts_body = parsePts3(local_pts_body_arr);
    auto local_pts_search = parsePts3(local_pts_search_arr);
    auto K = parseMat33(K_arr);
    auto R = parseMat33(R_arr);
    auto t = parseVec3(t_arr);

    StoneUpdateResult result;
    {
        py::gil_scoped_release release;
        result = trackStoneUpdateOne(
            frame_mat, local_pts_body, local_pts_search, K, R, t,
            X0, Y0, track_half_range_cm, coarse_step_cm, fine_step_cm,
            score_threshold, R_max_cm, H_total_cm, ring_r_frac_guess
        );
    }

    return resultToDict(result);
}


static py::list track_stones_batch(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_u,
    py::array_t<double, py::array::c_style | py::array::forcecast> X0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> Y0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_body_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_search_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr,
    double track_half_range_cm, double coarse_step_cm, double fine_step_cm,
    double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess)
{
    auto buf = frame_u.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("frame_u must be HxWx3 uint8 BGR");

    cv::Mat frame_mat((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, (void*)buf.ptr);

    auto local_pts_body = parsePts3(local_pts_body_arr);
    auto local_pts_search = parsePts3(local_pts_search_arr);
    auto K = parseMat33(K_arr);
    auto R = parseMat33(R_arr);
    auto t = parseVec3(t_arr);

    auto X0b = X0_arr.unchecked<1>();
    auto Y0b = Y0_arr.unchecked<1>();
    int n_stones = (int)X0b.shape(0);

    std::vector<StoneUpdateResult> results((size_t)n_stones);

    {
        py::gil_scoped_release release;

        // KRIITTINEN: OpenCV:n OMA sisainen rinnakkaistus (GaussianBlur,
        // morphologyEx, cvtColor jne. - "Parallel framework: pthreads")
        // yrittaa muuten kayttaa KAIKKIA CPU-ytimia JOKAISESSA alla
        // olevassa per-kiven worker-saikeessa SAMANAIKAISESTI - eli N
        // kiven kanssa N kertaa OpenCV:n oma sisainen saiemaara ytimista
        // kilpailemassa (havaittu kaytannossa: 1 kivi ~33ms, mutta jo
        // 2 kiveä ~107ms - PAHEMPI kuin sarjallisesti, koska ylikuormitus/
        // kontekstinvaihto syo koko hyodyn taman tiedoston OMASTA,
        // ULOMMASTA per-kivi-rinnakkaistuksesta). setNumThreads(1)
        // pakottaa jokaisen sisaisen OpenCV-kutsun sarjalliseksi TAMAN
        // kutsun ajaksi, jolloin AINOA rinnakkaistus on tama tiedoston
        // oma per-kivi std::thread-jako - palautetaan alkuperainen arvo
        // heti kutsun jalkeen etta muu prosessi (mode_engine.cpp,
        // Python-puolen paneiliseuranta jne.) ei karsi tasta.
        int prev_num_threads = cv::getNumThreads();
        cv::setNumThreads(1);

        std::atomic<int> next_idx(0);
        unsigned hw = std::thread::hardware_concurrency();
        int worker_count = std::max(1, std::min(n_stones, (int)(hw == 0 ? 4u : hw)));

        auto worker = [&]() {
            while (true) {
                int i = next_idx.fetch_add(1);
                if (i >= n_stones)
                    break;
                results[(size_t)i] = trackStoneUpdateOne(
                    frame_mat, local_pts_body, local_pts_search, K, R, t,
                    X0b(i), Y0b(i), track_half_range_cm, coarse_step_cm, fine_step_cm,
                    score_threshold, R_max_cm, H_total_cm, ring_r_frac_guess
                );
            }
        };

        std::vector<std::thread> workers;
        workers.reserve((size_t)worker_count);
        for (int i = 0; i < worker_count; ++i)
            workers.emplace_back(worker);
        for (auto& w : workers)
            w.join();

        cv::setNumThreads(prev_num_threads);
    }

    py::list out;
    for (auto& r : results)
        out.append(resultToDict(r));

    return out;
}


// ============================================================
// HAKU (main.py:n elavan seurannan HAKU-lohko, Task 6): uuden
// kiven etsinta kiinteältä paata-rajatulta vyohykkeelta KOKO
// framen kokoisella maskilla/saturaatiolla - EI ROI-rajattu
// (toisin kuin SEURANTA), koska hakuvyohyke itse voi jo kattaa
// ison osan framesta eika ajeta joka framella (katso main.py:n
// oma kommentti). Uudelleenkayttaa TASMALLEEN samoja funktioita
// (createGraniteMask, computeSat, locateByGridSearchFast,
// refinePositionJoint) off_x=off_y=0:lla - EI omaa, erillista
// logiikkaa naille.
//
// Portattu (Python-vastine main.py:ssa):
//   suppress_static_background, ja elavan HAKU-lohkon oma
//   kutsujarjestys (k9.create_granite_mask taustavaimennetulle
//   framelle, saturaatio ALKUPERAISESTA (ei-vaimennetusta)
//   framesta, k94.locate_by_grid_search_fast + k94.refine_
//   position_joint_fast).
// ============================================================

static StoneUpdateResult searchNewStoneOne(
    const cv::Mat& frame_mat, const cv::Mat& background_reference, double diff_threshold,
    const std::vector<cv::Point3d>& local_pts_body,
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double x_center, double x_half_width, double y_center, double y_half_range,
    double coarse_step_cm, double fine_step_cm, double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess)
{
    int frame_w = frame_mat.cols, frame_h = frame_mat.rows;

#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t0 = std::chrono::steady_clock::now();
#endif
    cv::Mat frame_filtered = suppressStaticBackground(frame_mat, background_reference, diff_threshold);
#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t1 = std::chrono::steady_clock::now();
#endif
    cv::Mat mask_search = createGraniteMask(frame_filtered);
#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t2 = std::chrono::steady_clock::now();
#endif
    cv::Mat sat_search = computeSat(frame_mat);
#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t3 = std::chrono::steady_clock::now();
#endif

    StoneUpdateResult out;

    auto best = locateByGridSearchFast(
        local_pts_search, mask_search, 0, 0,
        x_center, x_half_width, y_center, y_half_range,
        coarse_step_cm, fine_step_cm, K, R, t
    );

#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t4 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    fprintf(stderr, "[HAKU timing] suppress=%.2fms mask=%.2fms sat=%.2fms grid=%.2fms",
            ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4));
#endif

    out.score = best.second;

    if (out.score < score_threshold) {
#ifdef STONE_TRACKER_DEBUG_TIMING
        fprintf(stderr, " (no refine, score=%.3f)\n", out.score);
#endif
        return out;
    }

    out.has_position = true;
    out.refined = refinePositionJoint(
        mask_search, sat_search, 0, 0, frame_w, frame_h, local_pts_body, K, R, t,
        R_max_cm, H_total_cm, ring_r_frac_guess, best.first.x, best.first.y
    );

#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t5 = std::chrono::steady_clock::now();
    fprintf(stderr, " refine=%.2fms\n", ms(t4, t5));
#endif

    return out;
}


static py::dict search_new_stone(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_u,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> background_reference,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_body_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_search_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr,
    double x_center, double x_half_width, double y_center, double y_half_range,
    double coarse_step_cm, double fine_step_cm, double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess,
    double diff_threshold = 30.0)
{
    auto buf = frame_u.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("frame_u must be HxWx3 uint8 BGR");

    cv::Mat frame_mat((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, (void*)buf.ptr);

    cv::Mat ref_mat;
    auto ref_buf = background_reference.request();
    if (ref_buf.ndim == 3 && ref_buf.shape[2] == 3)
        ref_mat = cv::Mat((int)ref_buf.shape[0], (int)ref_buf.shape[1], CV_8UC3, (void*)ref_buf.ptr);

    auto local_pts_body = parsePts3(local_pts_body_arr);
    auto local_pts_search = parsePts3(local_pts_search_arr);
    auto K = parseMat33(K_arr);
    auto R = parseMat33(R_arr);
    auto t = parseVec3(t_arr);

    StoneUpdateResult result;
    {
        py::gil_scoped_release release;
        result = searchNewStoneOne(
            frame_mat, ref_mat, diff_threshold, local_pts_body, local_pts_search, K, R, t,
            x_center, x_half_width, y_center, y_half_range,
            coarse_step_cm, fine_step_cm, score_threshold,
            R_max_cm, H_total_cm, ring_r_frac_guess
        );
    }

    return resultToDict(result);
}


PYBIND11_MODULE(stone_tracker, m)
{
    m.doc() = "C++-porttaus SEURANTA- ja HAKU-vaiheiden kuumasta polusta (Task 5+6)";

    m.def("track_stone_update", &track_stone_update,
          "Yhden kiven ristikkohaku+yhteissovitus (SEURANTA-paivitys)");

    m.def("track_stones_batch", &track_stones_batch,
          "Usean kiven ristikkohaku+yhteissovitus rinnakkain std::thread:eilla");

    m.def("search_new_stone", &search_new_stone,
          "Uuden kiven haku kiinteältä vyohykkeelta (HAKU), koko frame",
          py::arg("frame_u"), py::arg("background_reference"),
          py::arg("local_pts_body"), py::arg("local_pts_search"),
          py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("x_center"), py::arg("x_half_width"),
          py::arg("y_center"), py::arg("y_half_range"),
          py::arg("coarse_step_cm"), py::arg("fine_step_cm"),
          py::arg("score_threshold"),
          py::arg("R_max_cm"), py::arg("H_total_cm"), py::arg("ring_r_frac_guess"),
          py::arg("diff_threshold") = 30.0);
}
