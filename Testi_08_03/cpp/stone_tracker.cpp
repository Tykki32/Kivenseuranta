// ============================================================
// stone_tracker.cpp - C++-laajennusmoduuli (pybind11): kivien HAKU ja SEURANTA, siluettitarkennus, ruudun
// esikasittely (taustanvaimennus, vaihekorrelaatio, GPU-vaihe B) ja kiviprofiilin opettelu.
//
// Osat (kaannetaan yhtena kaannosyksikkona; jarjestys = riippuvuudet):
//   yhteiset.hpp     includet, aikamittaus, OpenCV:n yksisaikeisuusvartija, vakiot, numpy-muunnokset
//   geometria.hpp    projektio, kiven hulli, peitto-osuus, ristikkohaut
//   maskit.hpp       saturaatio, graniittimaski, variportti, taustanvaimennus, ROI, bilineaarinen nayte
//   sovitus.hpp      runkokontuuri, rengaspisteet, jaannosvirheet, Levenberg-Marquardt, refinePositionJoint
//   meanshift.hpp    SEURANNAN mean-shift-paikannus
//   siluetti.hpp     siluettitarkennus
//   seuranta.hpp     track_stones_batch
//   haku.hpp         search_new_stones
//   esikasittely.hpp suppress_shadow_background, phase_window / phase_mulnorm / phase_peak
//   gpu.hpp          GPU-vaihe B (OpenCL)
//   profiili.hpp     scan_stone_candidates, fit_stone_profile_cpp, alpha_observation_cpp / alpha_contour_cpp
// ============================================================
#include "yhteiset.hpp"
#include "geometria.hpp"
#include "maskit.hpp"
#include "sovitus.hpp"
#include "meanshift.hpp"
#include "siluetti.hpp"
#include "seuranta.hpp"
#include "haku.hpp"
#include "esikasittely.hpp"
#include "gpu.hpp"
#include "profiili.hpp"

PYBIND11_MODULE(stone_tracker, m)
{
    m.doc() = "Kivien HAKU ja SEURANTA, siluettitarkennus, esikasittely ja kiviprofiilin opettelu";

    // --- tiedot ja asetukset ---
    m.def("build_info", []() { return std::string("stone_tracker kaannetty ") + __DATE__ + " " + __TIME__; });
    m.def("opencv_build_info", []() { return std::string(cv::getBuildInformation()); });
    m.def("prof_snapshot", &prof_snapshot, "Aikamittaukset: lista (nimi, ms yhteensa, kutsuja)");
    m.def("sat_simd_info", []() {
        const bool simd = satSimdOk();
        return py::make_tuple(simd, g_sat_us[0], g_sat_us[1]);
    }, "Saturaatio cvtColor:lla (SIMD)? -> (kaytossa, skalaari_us, cvtColor_us)");
    m.def("set_color_gate", [](int s_max, int h_tol) { g_gate_s_max = s_max; g_gate_h_tol = h_tol; },
          py::arg("s_max"), py::arg("h_tol"));
    m.def("set_spawn_filter", [](double rms_max, int nb_min, int nb_max, double abs_x_max, double score_max) {
        g_spawn_cfg.rms_max = rms_max; g_spawn_cfg.nb_min = nb_min; g_spawn_cfg.nb_max = nb_max;
        g_spawn_cfg.abs_x_max = abs_x_max; g_spawn_cfg.score_max = score_max;
    }, py::arg("rms_max"), py::arg("nb_min"), py::arg("nb_max"), py::arg("abs_x_max"), py::arg("score_max"));
    m.def("set_prep_parallel", [](int on) { g_prep_parallel = on ? 1 : 0; }, py::arg("on"));
    m.def("set_cv_single_thread_persistent", [](int on) { setCvSingleThreadPersistent(on != 0); }, py::arg("on"));
    m.def("set_seuranta_silhouette", [](ArrD body_arr, ArrD K_arr, ArrD R_arr, ArrD t_arr,
                                        double R_max, double H_total, double lovi, double w, double lam, double sigfrac,
                                        int max_shift, int half, int margin, int open_size, int band) {
        SeurantaSilCfg c;
        c.body = parsePts3(body_arr);
        c.K = parseMat33(K_arr); c.R = parseMat33(R_arr); c.t = parseVec3(t_arr);
        c.R_max = R_max; c.H_total = H_total; c.lovi = lovi; c.w = w; c.lam = lam; c.sigfrac = sigfrac;
        c.max_shift = max_shift; c.half = half; c.margin = margin; c.open_size = open_size; c.band = band;
        c.on = true;
        std::lock_guard<std::mutex> lk(g_sil_cfg_mx);
        g_sil_cfg = c;
    });

    // --- SEURANTA ja HAKU ---
    m.def("track_stones_batch", &track_stones_batch,
          "SEURANTA: kivien paikan paivitys rinnakkain (yhdistelmahaku + LM + siluettitarkennus)",
          py::arg("frame_u"), py::arg("background_reference"),
          py::arg("X0_arr"), py::arg("Y0_arr"), py::arg("half_range_x_arr"), py::arg("half_range_y_arr"),
          py::arg("local_pts_body"), py::arg("local_pts_search"), py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("coarse_step_cm"), py::arg("fine_step_cm"), py::arg("score_threshold"),
          py::arg("R_max_cm"), py::arg("H_total_cm"), py::arg("ring_r_frac_guess"), py::arg("handle_r_frac"),
          py::arg("max_backward_cm"), py::arg("diff_threshold"),
          py::arg("ms_gain"), py::arg("ms_max_iter"), py::arg("ms_tol_px"), py::arg("ms_inner_weight"),
          py::arg("ms_margin_scale"), py::arg("ms_tau"), py::arg("ms_polish_step_cm"),
          py::arg("pred_dx"), py::arg("pred_dy"),
          py::arg("ens_back_tol_cm"), py::arg("ens_back_pen"), py::arg("ens_pred_pen"));
    m.def("search_new_stones", &search_new_stones,
          "HAKU: uudet kivet hakualueelta (kaikki kelvolliset ehdokkaat, enintaan max_results)",
          py::arg("frame_u"), py::arg("background_reference"),
          py::arg("local_pts_body"), py::arg("local_pts_search"), py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("x_center"), py::arg("x_half_width"), py::arg("y_center"), py::arg("y_half_range"),
          py::arg("coarse_step_cm"), py::arg("fine_step_cm"), py::arg("score_threshold"),
          py::arg("R_max_cm"), py::arg("H_total_cm"), py::arg("ring_r_frac_guess"), py::arg("handle_r_frac"),
          py::arg("diff_threshold"), py::arg("max_results"), py::arg("max_attempts"));
    m.def("silhouette_refine_cpp", &silhouette_refine_cpp,
          py::arg("frame"), py::arg("local_pts_body"), py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("R_max_cm"), py::arg("H_total_cm"), py::arg("handle_r_frac"), py::arg("X0"), py::arg("Y0"),
          py::arg("w_leak"), py::arg("lam"), py::arg("sigma_frac"), py::arg("max_shift_px"), py::arg("half"),
          py::arg("margin"), py::arg("open_size"), py::arg("band_px"));

    // --- esikasittely ---
    m.def("suppress_shadow_background", &suppress_shadow_background,
          py::arg("frame"), py::arg("reference"), py::arg("gains"), py::arg("biases"),
          py::arg("diff_threshold"), py::arg("v_drop_min"), py::arg("v_drop_max"), py::arg("ice_s_max"), py::arg("ice_v_min"));
    m.def("phase_window", &phase_window);
    m.def("phase_mulnorm", &phase_mulnorm);
    m.def("phase_peak", &phase_peak);
    m.def("gpu_b_init", &gpu_b_init);
    m.def("gpu_b_warp", &gpu_b_warp);
    m.def("gpu_b_suppress", &gpu_b_suppress);

    // --- kiviprofiili ---
    m.def("scan_stone_candidates", &scan_stone_candidates,
          "Kivikandidaatit yhdesta ruudusta (kiviprofiilin opettelu)",
          py::arg("frame_bgr"), py::arg("background_reference"),
          py::arg("H_final"), py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("min_area"), py::arg("max_area"), py::arg("min_fill_ratio"), py::arg("min_aspect_ratio"),
          py::arg("x_min"), py::arg("x_max"), py::arg("y_min"), py::arg("y_max"),
          py::arg("pixels_per_cm"), py::arg("output_x_min_cm"), py::arg("output_y_max_cm"),
          py::arg("diff_threshold") = 30.0,
          py::arg("roi_x0") = -1, py::arg("roi_y0") = -1, py::arg("roi_x1") = -1, py::arg("roi_y1") = -1);
    m.def("fit_stone_profile_cpp", &fit_stone_profile_cpp,
          py::arg("K"), py::arg("R"), py::arg("t"), py::arg("contours"), py::arg("positions0"),
          py::arg("n_sample") = 40, py::arg("initial_radius") = 14.55, py::arg("height_min") = 11.43, py::arg("height_max") = 15.0,
          py::arg("handle_min") = 0.30, py::arg("handle_max") = 0.95, py::arg("handle_init") = 0.70,
          py::arg("reg_weight") = 60.0, py::arg("max_iterations") = 100, py::arg("r_fixed") = -1.0);
    m.def("alpha_observation_cpp", &alpha_observation_cpp,
          py::arg("frame"), py::arg("reference"), py::arg("gains"), py::arg("biases"), py::arg("cx"), py::arg("cy"),
          py::arg("diff_threshold"), py::arg("v_drop_min"), py::arg("v_drop_max"), py::arg("ice_s_max"), py::arg("ice_v_min"),
          py::arg("crop_half"), py::arg("a_dilate"), py::arg("dark_v_max"));
    m.def("alpha_contour_cpp", &alpha_contour_cpp,
          py::arg("alpha"), py::arg("adil"), py::arg("center_x"), py::arg("center_y"), py::arg("origin_x"), py::arg("origin_y"),
          py::arg("level"), py::arg("up"));
}
