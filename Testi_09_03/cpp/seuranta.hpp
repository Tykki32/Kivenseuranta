// ============================================================
// seuranta.hpp - SEURANTA: liikkuvien kivien paikan paivitys (track_stones_batch, kivet rinnan omissa saikeissaan).
//
// Yhden kiven paivitys:
//   1) ROI hakualueen ymparilta; taustanvaimennus (jos kynnys > 0; muuten kuva on vaimennettu jo Pythonissa),
//      saturaatio alkuperaisesta kuvasta, graniittimaski vaimennetusta + etualamaski (OR).
//   2) Yhdistelmahaku: ristikkohaku ennustetusta paikasta + mean-shift viimeisesta ja ennustetusta paikasta (rinnan).
//      Valinta: ehdokkaista, joiden pistemaara >= kynnys, suurin (pistemaara - taaksepain-rangaistus -
//      ennusteesta poikkeamisen rangaistus); jos yksikaan ei ylita kynnysta, suurin pistemaara.
//   3) LM-yhteissovitus (refinePositionJoint); havainto hylataan jos kivi siirtyi taaksepain yli max_backward_cm.
//   4) Siluettitarkennus samassa saikeessa (set_seuranta_silhouette).
// ============================================================
#pragma once
#include "meanshift.hpp"
#include "siluetti.hpp"

struct StoneUpdateResult {
    double score = 0.0;
    bool has_position = false;
    RefineResult refined;
};

static StoneUpdateResult trackStoneUpdateOne(
    const cv::Mat& frame_mat, const cv::Mat& background_reference, double diff_threshold,
    const std::vector<cv::Point3d>& local_pts_body,
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double X0, double Y0,
    double track_half_range_x_cm, double track_half_range_y_cm,
    double coarse_step_cm, double fine_step_cm,
    double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    double max_backward_cm,
    const MeanShiftParams& ms_params, double pred_dX, double pred_dY)
{
    auto msSince = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    PT pt_call;
    PT pt;
    int frame_w = frame_mat.cols, frame_h = frame_mat.rows;
    StoneUpdateResult out;

    // ROI mitoitetaan suuremmalla hakualueen puolikkaalla + tarkennuksen sallima siirtyma
    double roi_half_range = std::max(track_half_range_x_cm, track_half_range_y_cm) + BOUNDARY_MAX_SHIFT_FROM_APPROX_CM;
    cv::Rect roi = trackRoiBounds(X0, Y0, roi_half_range, R_max_cm, H_total_cm, frame_w, frame_h, K, R, t);
    if (roi.width <= 0 || roi.height <= 0) {
        profAdd(P_CALL_TOTAL, pt_call.lap());
        return out;
    }

    // --- 1) maskit ---
    cv::Mat crop = frame_mat(roi);
    pt.lap();
    cv::Mat crop_filtered = crop;
    bool have_bg_reference = false;
    if (!background_reference.empty() && background_reference.size() == frame_mat.size()
        && background_reference.type() == frame_mat.type()) {
        if (diff_threshold > 0.0)
            crop_filtered = suppressStaticBackground(crop, background_reference(roi), diff_threshold);
        have_bg_reference = true;
    }
    profAdd(P_PREP_CROP_SUPPRESS, pt.lap());

    // etualamaski omassa saikeessaan (g_prep_parallel) saturaation ja graniittimaskin rinnalla
    cv::Mat fg_mask_par;
    long long fg_ns = 0;
    std::thread fg_thread;
    const bool fg_par = have_bg_reference && g_prep_parallel.load();
    if (fg_par)
        fg_thread = std::thread([&]() { PT p; fg_mask_par = createForegroundFromWhitened(crop_filtered); fg_ns = p.lap(); });

    // saturaatio alkuperaisesta kuvasta (valkaisu loisi keinotekoisen saturaatiorajan kiven reunalle)
    cv::Mat sat_u8_crop, sat_crop, mask_crop;
    const bool same_image = (crop_filtered.data == crop.data);
    sat_u8_crop = satU8FromBgr(crop);
    sat_u8_crop.convertTo(sat_crop, CV_32F);
    profAdd(P_PREP_SAT, pt.lap());
    mask_crop = createGraniteMask(crop_filtered, same_image ? &sat_u8_crop : nullptr);
    profAdd(P_PREP_GRANITE, pt.lap());

    cv::Mat mask_for_track = mask_crop;
    if (fg_par) {
        fg_thread.join();
        mask_for_track = mask_crop | fg_mask_par;
        pt.lap();
        profAdd(P_PREP_FOREGROUND, fg_ns);
    } else {
        if (have_bg_reference)
            mask_for_track = mask_crop | createForegroundFromWhitened(crop_filtered);
        profAdd(P_PREP_FOREGROUND, pt.lap());
    }

    // --- 2) yhdistelmahaku ---
    auto tl0 = std::chrono::steady_clock::now();
    pt.lap();
    struct Cand { cv::Point2d xy; double score; double adj; };
    std::vector<Cand> cands;
    const bool have_pred = std::hypot(pred_dX, pred_dY) > 0.5;
    auto adjust = [&](const cv::Point2d& xy, double score) {
        double back = std::max(0.0, (xy.y - Y0) - ms_params.ens_back_tol_cm);
        double pdev = have_pred ? std::hypot(xy.x - (X0 + pred_dX), xy.y - (Y0 + pred_dY)) : 0.0;
        double adj = score - ms_params.ens_back_pen * back / 10.0 - ms_params.ens_pred_pen * pdev / 10.0;
        cands.push_back({ xy, score, adj });
    };

    std::pair<cv::Point2d, double> g;
    MeanShiftResult m1, m2;
    {
        long long ns1 = 0, ns2 = 0;
        std::thread th1([&]() {
            PT p;
            m1 = meanShiftLocate(local_pts_search, mask_for_track, roi.x, roi.y,
                X0, Y0, track_half_range_x_cm, track_half_range_y_cm, K, R, t, ms_params);
            ns1 = p.lap();
        });
        std::thread th2;
        if (have_pred)
            th2 = std::thread([&]() {
                PT p;
                m2 = meanShiftLocate(local_pts_search, mask_for_track, roi.x, roi.y,
                    X0, Y0, track_half_range_x_cm, track_half_range_y_cm, K, R, t, ms_params,
                    X0 + pred_dX, Y0 + pred_dY);
                ns2 = p.lap();
            });
        g = locateByGridSearchTrackingFast(
            local_pts_search, mask_for_track, roi.x, roi.y,
            X0 + pred_dX, track_half_range_x_cm, Y0 + pred_dY, track_half_range_y_cm,
            coarse_step_cm, fine_step_cm, K, R, t);
        profAdd(P_LOC_ENS_GRID, pt.lap());
        th1.join();
        if (have_pred) th2.join();
        profAdd(P_LOC_MS1, ns1);
        if (have_pred) profAdd(P_LOC_MS2, ns2);
    }
    adjust(g.first, g.second);
    adjust(m1.xy, m1.score);
    if (have_pred)
        adjust(m2.xy, m2.score);

    const Cand* pick = nullptr;
    for (const auto& c : cands)
        if (c.score >= score_threshold && (!pick || c.adj > pick->adj))
            pick = &c;
    if (!pick)
        for (const auto& c : cands)
            if (!pick || c.score > pick->score)
                pick = &c;
    profAdd(P_LOC_TOTAL, (long long)(msSince(tl0) * 1e6));

    out.score = pick->score;
    if (out.score < score_threshold) {
        profAdd(P_CALL_TOTAL, pt_call.lap());
        return out;
    }

    // --- 3) LM-yhteissovitus ---
    auto tr0 = std::chrono::steady_clock::now();
    out.refined = refinePositionJoint(
        mask_for_track, sat_crop, roi.x, roi.y, frame_w, frame_h, local_pts_body, K, R, t,
        R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, pick->xy.x, pick->xy.y);
    profAdd(P_REFINE_TOTAL, (long long)(msSince(tr0) * 1e6));

    // kivi ei peruuta: Y kasvaa (kohti heittopaata) yli max_backward_cm -> vaara kohde (viereinen kivi / pelaaja)
    out.has_position = !((out.refined.Y_cm - Y0) > max_backward_cm);
    profAdd(P_CALL_TOTAL, pt_call.lap());
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

// Python: seuranta.py (Seuranta._seuranta). Palauttaa listan dict:eja (score, found, X_cm, Y_cm, tarkka, n_body, n_ring,
// rms_px, ring_radius_cm, sil).
static py::list track_stones_batch(
    ArrU8 frame_u, ArrU8 background_reference,
    ArrD X0_arr, ArrD Y0_arr, ArrD half_range_x_arr, ArrD half_range_y_arr,
    ArrD local_pts_body_arr, ArrD local_pts_search_arr,
    ArrD K_arr, ArrD R_arr, ArrD t_arr,
    double coarse_step_cm, double fine_step_cm, double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    double max_backward_cm, double diff_threshold,
    double ms_gain, int ms_max_iter, double ms_tol_px, double ms_inner_weight, double ms_margin_scale,
    double ms_tau, double ms_polish_step_cm,
    ArrD pred_dx_arr, ArrD pred_dy_arr,
    double ens_back_tol_cm, double ens_back_pen, double ens_pred_pen)
{
    PT bpt_wall, bpt;
    cv::Mat frame_mat = bgrView(frame_u);
    if (frame_mat.empty())
        throw std::runtime_error("frame_u must be HxWx3 uint8 BGR");
    cv::Mat ref_mat = bgrView(background_reference);

    auto local_pts_body = parsePts3(local_pts_body_arr);
    auto local_pts_search = parsePts3(local_pts_search_arr);
    auto K = parseMat33(K_arr);
    auto R = parseMat33(R_arr);
    auto t = parseVec3(t_arr);

    MeanShiftParams msp;
    msp.gain = ms_gain;
    msp.max_iter = ms_max_iter;
    msp.tol_px = ms_tol_px;
    msp.inner_weight = ms_inner_weight;
    msp.margin_scale = ms_margin_scale;
    msp.tau = ms_tau;
    msp.polish_step_cm = ms_polish_step_cm;
    msp.ens_back_tol_cm = ens_back_tol_cm;
    msp.ens_back_pen = ens_back_pen;
    msp.ens_pred_pen = ens_pred_pen;

    auto X0b = X0_arr.unchecked<1>();
    auto Y0b = Y0_arr.unchecked<1>();
    auto HXb = half_range_x_arr.unchecked<1>();
    auto HYb = half_range_y_arr.unchecked<1>();
    auto PDXb = pred_dx_arr.unchecked<1>();
    auto PDYb = pred_dy_arr.unchecked<1>();
    int n_stones = (int)X0b.shape(0);

    std::vector<StoneUpdateResult> results((size_t)n_stones);
    std::vector<SilResult> sil_results((size_t)n_stones);
    std::vector<char> has_sil((size_t)n_stones, 0);
    SeurantaSilCfg sil_cfg;
    { std::lock_guard<std::mutex> lk(g_sil_cfg_mx); sil_cfg = g_sil_cfg; }
    profAdd(P_BATCH_SETUP, bpt.lap());

    {
        py::gil_scoped_release release;
        ScopedSingleThreadedOpenCV single_threaded_opencv_guard;

        std::atomic<int> next_idx(0);
        unsigned hw = std::thread::hardware_concurrency();
        int worker_count = std::max(1, std::min(n_stones, (int)(hw == 0 ? 4u : hw)));
        auto worker = [&]() {
            while (true) {
                int i = next_idx.fetch_add(1);
                if (i >= n_stones)
                    break;
                results[(size_t)i] = trackStoneUpdateOne(
                    frame_mat, ref_mat, diff_threshold, local_pts_body, local_pts_search, K, R, t,
                    X0b(i), Y0b(i), HXb(i), HYb(i), coarse_step_cm, fine_step_cm,
                    score_threshold, R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, max_backward_cm,
                    msp, PDXb(i), PDYb(i));
                if (sil_cfg.on && results[(size_t)i].has_position) {
                    const auto& rr = results[(size_t)i].refined;
                    sil_results[(size_t)i] = silhouetteRefineCore(frame_mat, sil_cfg.body, sil_cfg.K, sil_cfg.R, sil_cfg.t,
                        sil_cfg.R_max, sil_cfg.H_total, sil_cfg.lovi, rr.X_cm, rr.Y_cm, sil_cfg.w, sil_cfg.lam, sil_cfg.sigfrac,
                        sil_cfg.max_shift, sil_cfg.half, sil_cfg.margin, sil_cfg.open_size, sil_cfg.band);
                    has_sil[(size_t)i] = 1;
                }
            }
        };
        std::vector<std::thread> workers;
        workers.reserve((size_t)worker_count);
        for (int i = 0; i < worker_count; ++i)
            workers.emplace_back(worker);
        for (auto& w : workers)
            w.join();
    }

    bpt.lap();
    py::list out;
    for (size_t i = 0; i < results.size(); ++i) {
        py::dict d = resultToDict(results[i]);
        if (has_sil[i]) d["sil"] = silResultToTuple(sil_results[i]);
        out.append(d);
    }
    profAdd(P_BATCH_RESULT, bpt.lap());
    profAdd(P_BATCH_WALL, bpt_wall.lap());
    return out;
}
