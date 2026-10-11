// ============================================================
// haku.hpp - HAKU: uusien kivien etsinta kiintealta hakualueelta (kaukohogin ymparilla) koko kuvan maskilla.
//
// Ristikkohaku (locateByGridSearchFast) loytaa alueen parhaan kohteen, jonka paikka tarkennetaan (refinePositionJoint).
// Ehdokas hyvaksytaan, jos tarkennetun paikan peitto-osuus >= HAKU_ACCEPT_SCORE ja spawn-suodatin hyvaksyy sen.
// Lahempana ja isompana nakyva pelaaja voittaa ristikkohaun helposti, joten jokaisen yrityksen jalkeen kohteen koko
// yhtenainen maskialue poistetaan (floodFill) ja haetaan uudelleen: nain myos kauempana oleva kivi paasee vuoroon.
// Palauttaa kaikki hyvaksytyt ehdokkaat (enintaan max_results, yrityksia enintaan max_attempts).
// ============================================================
#pragma once
#include "seuranta.hpp"

// Tarkennetun paikan peitto-osuuden alaraja. Matalampi kuin SEURANNAN kynnys: radan vahvistus (Python) suodattaa
// pelaajat myohemmin usean ruudun perusteella.
static const double HAKU_ACCEPT_SCORE = 0.20;

// ------------------------------------------------------------
// SPAWN-SUODATIN: roskaehdokkaiden (pelaajat, lakaisijat, kohina) hylkays jo rekisteroinnissa: sovituksen rms_px <=
// rms_max, runkopisteita nb_min..nb_max, |X| <= abs_x_max, ristikkopistemaara <= score_max. Hylatty ehdokas
// kasitellaan kuten hyvaksymatta jaanyt. Python: set_spawn_filter(A.HAKU_UUSI_*); rms_max <= 0 = pois.
// ------------------------------------------------------------
struct SpawnFilterCfg {
    double rms_max = 0.0;
    int nb_min = 0, nb_max = 1000000;
    double abs_x_max = 1e9;
    double score_max = 1e9;
};
static SpawnFilterCfg g_spawn_cfg;

static bool spawnFilterPass(const RefineResult& r, double grid_score)
{
    if (g_spawn_cfg.rms_max <= 0.0)
        return true;
    bool ok = r.has_rms && r.rms_px <= g_spawn_cfg.rms_max
        && r.n_body >= g_spawn_cfg.nb_min && r.n_body <= g_spawn_cfg.nb_max
        && std::abs(r.X_cm) <= g_spawn_cfg.abs_x_max
        && grid_score <= g_spawn_cfg.score_max;
    profAdd(ok ? P_SPAWN_ACCEPT : P_SPAWN_REJECT, 0);
    return ok;
}

static void searchNewStones(
    const cv::Mat& frame_mat, const cv::Mat& background_reference, double diff_threshold,
    const std::vector<cv::Point3d>& local_pts_body,
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double x_center, double x_half_width, double y_center, double y_half_range,
    double coarse_step_cm, double fine_step_cm, double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    std::vector<StoneUpdateResult>& found, int max_results, int max_attempts)
{
    int frame_w = frame_mat.cols, frame_h = frame_mat.rows;
    PT hpt_total, hpt;
    cv::Mat frame_filtered = suppressStaticBackground(frame_mat, background_reference, diff_threshold);
    profAdd(P_HAKU_SUPPRESS, hpt.lap());
    cv::Mat mask_search = createGraniteMask(frame_filtered);
    profAdd(P_HAKU_MASK, hpt.lap());
    cv::Mat sat_search = computeSat(frame_mat);
    profAdd(P_HAKU_SAT, hpt.lap());

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        auto best = locateByGridSearchFast(local_pts_search, mask_search, 0, 0, x_center, x_half_width, y_center,
                                           y_half_range, coarse_step_cm, fine_step_cm, K, R, t);
        profAdd(P_HAKU_LOCATE, hpt.lap());
        if (best.second < score_threshold)
            return;

        RefineResult refined = refinePositionJoint(
            mask_search, sat_search, 0, 0, frame_w, frame_h, local_pts_body, K, R, t,
            R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, best.first.x, best.first.y);
        profAdd(P_HAKU_REFINE, hpt.lap());

        // pyorea kivi: tumma alue loppuu mallin reunaan (korkea pistemaara); jalka tms. jatkuu reunan yli
        auto refined_hull = predictedHull(local_pts_body, refined.X_cm, refined.Y_cm, K, R, t);
        double refined_score = hullOverlapScore(mask_search, refined_hull, 0, 0);
        if (refined_score >= HAKU_ACCEPT_SCORE && spawnFilterPass(refined, best.second)) {
            StoneUpdateResult r;
            r.score = best.second;
            r.refined = refined;
            r.has_position = true;
            found.push_back(r);
            if ((int)found.size() >= max_results)
                return;
        }

        // Poistetaan kohteen yhtenainen maskialue. Siemen: lahin maskipikseli mallin keskipisteen projektion
        // ymparilta (projektio ei valttamatta osu tasan kohteen pikseliin).
        std::vector<cv::Point3d> best_pos3d{ cv::Point3d(best.first.x, best.first.y, H_total_cm / 2.0) };
        auto best_proj = project3d(K, R, t, best_pos3d);
        int px0 = (int)std::lround(best_proj[0].x);
        int py0 = (int)std::lround(best_proj[0].y);
        int seed_px = -1, seed_py = -1;
        const int SEED_SEARCH_RADIUS_PX = 25;
        for (int rad = 0; rad <= SEED_SEARCH_RADIUS_PX && seed_px < 0; ++rad) {
            int x_lo = std::max(0, px0 - rad), x_hi = std::min(mask_search.cols - 1, px0 + rad);
            int y_lo = std::max(0, py0 - rad), y_hi = std::min(mask_search.rows - 1, py0 + rad);
            for (int yy = y_lo; yy <= y_hi && seed_px < 0; ++yy) {
                bool on_border_row = (yy == py0 - rad || yy == py0 + rad);
                int step = on_border_row ? 1 : std::max(1, x_hi - x_lo);
                for (int xx = x_lo; xx <= x_hi; xx += step) {
                    if (mask_search.at<uchar>(yy, xx) != 0) {
                        seed_px = xx; seed_py = yy;
                        break;
                    }
                }
            }
        }
        if (seed_px < 0)
            return;       // ei poistettavaa aluetta -> sama tulos toistuisi

        cv::Mat flood_mask = cv::Mat::zeros(mask_search.rows + 2, mask_search.cols + 2, CV_8UC1);
        cv::floodFill(mask_search, flood_mask, cv::Point(seed_px, seed_py), cv::Scalar(0),
                      nullptr, cv::Scalar(0), cv::Scalar(0), 4 | cv::FLOODFILL_MASK_ONLY | (255 << 8));
        cv::Mat region = flood_mask(cv::Rect(1, 1, mask_search.cols, mask_search.rows));
        mask_search.setTo(cv::Scalar(0), region);
        profAdd(P_HAKU_FLOOD, hpt.lap());
    }
    profAdd(P_HAKU_TOTAL, hpt_total.lap());
}

// Python: seuranta.py (Seuranta._run_haku, taustasaie). Lista dict:eja kuten track_stones_batch (ilman "sil").
static py::list search_new_stones(
    ArrU8 frame_u, ArrU8 background_reference,
    ArrD local_pts_body_arr, ArrD local_pts_search_arr,
    ArrD K_arr, ArrD R_arr, ArrD t_arr,
    double x_center, double x_half_width, double y_center, double y_half_range,
    double coarse_step_cm, double fine_step_cm, double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    double diff_threshold, int max_results, int max_attempts)
{
    cv::Mat frame_mat = bgrView(frame_u);
    if (frame_mat.empty())
        throw std::runtime_error("frame_u must be HxWx3 uint8 BGR");
    cv::Mat ref_mat = bgrView(background_reference);
    auto local_pts_body = parsePts3(local_pts_body_arr);
    auto local_pts_search = parsePts3(local_pts_search_arr);
    auto K = parseMat33(K_arr);
    auto R = parseMat33(R_arr);
    auto t = parseVec3(t_arr);

    std::vector<StoneUpdateResult> results;
    {
        py::gil_scoped_release release;
        ScopedSingleThreadedOpenCV single_threaded_opencv_guard;
        searchNewStones(frame_mat, ref_mat, diff_threshold, local_pts_body, local_pts_search, K, R, t,
                        x_center, x_half_width, y_center, y_half_range, coarse_step_cm, fine_step_cm, score_threshold,
                        R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, results, max_results, max_attempts);
    }
    py::list out;
    for (auto& r : results)
        out.append(resultToDict(r));
    return out;
}
