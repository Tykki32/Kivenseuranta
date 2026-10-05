// ============================================================
// yhteiset.hpp - stone_tracker-moduulin yhteiset osat: includet, aikamittaus, saikeistyksen ohjaus,
// OpenCV:n yksisaikeisuusvartija, algoritmin vakiot ja numpy-muunnokset.
// ============================================================
#pragma once

// M_PI: MSVC paljastaa sen vain jos _USE_MATH_DEFINES on maaritelty ennen ensimmaista <cmath>-includea.
#define _USE_MATH_DEFINES

// GPU-vaihe B lataa OpenCL-kirjaston ajonaikaisesti (ei kaannosaikaista riippuvuutta)
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace py = pybind11;

// numpy-taulukot (C-jarjestys, tyyppimuunnos tarvittaessa)
using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrF = py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrU8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// ============================================================
// AIKAMITTAUS: jokainen vaihe kirjaa kokonaisajan ja kutsumaaran globaaleihin laskureihin; Python lukee ne
// prof_snapshot()-funktiolla ajon lopun raporttiin. Saikeissa ajat summautuvat (CPU-aika).
// ============================================================
enum ProfId {
    P_CALL_TOTAL, P_PREP_CROP_SUPPRESS, P_PREP_SAT, P_PREP_GRANITE, P_PREP_FOREGROUND,
    P_LOC_MS1, P_LOC_MS2, P_LOC_ENS_GRID, P_LOC_TOTAL,
    P_REFINE_TOTAL, P_R_PAD, P_R_CONTOUR, P_R_BOUNDARY, P_R_LM1, P_R_MAD, P_R_LM2, P_R_FINAL, P_R_OUTER_ITER,
    P_HAKU_TOTAL, P_HAKU_SUPPRESS, P_HAKU_MASK, P_HAKU_SAT, P_HAKU_LOCATE, P_HAKU_REFINE, P_HAKU_FLOOD,
    P_BATCH_WALL, P_BATCH_SETUP, P_BATCH_RESULT,
    P_SPAWN_REJECT, P_SPAWN_ACCEPT,
    P_GM_CVT, P_GM_BLUR, P_GM_LOOP, P_GM_MORPH, P_GS_HULL, P_GS_SCORE,
    P_LM_FALLBACK, P_LM_LIN, P_LM_VERIFY, P_LM_BUILD, P_LM_EVAL_ONLY,
    P_C_NBODY, P_C_NRING, P_C_NHULL, P_C_LMITER, P_C_LMEVAL,
    P_COUNT
};
static const char* PROF_NAMES[P_COUNT] = {
    "seuranta/kivi-paivitys yhteensa (CPU)", "  prep: rajaus + taustanvaimennus", "  prep: saturaatio", "  prep: graniittimaski (GaussianBlur)",
    "  prep: etualamaski (whitened)", "  haku: mean-shift (viimeisesta)", "  haku: mean-shift (ennustetusta)",
    "  haku: ristikko (yhdistelma, ennustettu keskipiste)", "  haku yhteensa", "  LM-tarkennus yhteensa", "    LM: reunustus (copyMakeBorder)",
    "    LM: runkokontuuri + jaasuodatus", "    LM: reunapisteet (detectBoundaryPoints)", "    LM: sovitus #1", "    LM: MAD-poikkeamat",
    "    LM: sovitus #2 (uusinta)", "    LM: lopullinen residuaali", "    LM: ulkoiteraatioita (lkm)",
    "HAKU yhteensa", "  HAKU: taustanvaimennus (koko frame)", "  HAKU: graniittimaski (koko frame)", "  HAKU: saturaatio (koko frame)",
    "  HAKU: ristikkohaku", "  HAKU: LM-tarkennus", "  HAKU: floodFill-poisto",
    "SEURANTA-kutsu seinakello (track_stones_batch)", "  kutsun alustus (numpy->cv, GIL)", "  kutsun tulokset (dict)",
    "HAKU: spawn-suodatin HYLKASI ehdokkaan (kpl)", "HAKU: spawn-suodatin hyvaksyi ehdokkaan (kpl)",
    "    granite: cvtColor x2 + convert", "    granite: GaussianBlur", "    granite: kynnys-silmukka", "    granite: morfologia", "    ristikko: predictedHull", "    ristikko: hullOverlapScore",
    "    LM: TARKKA VARAKEINO-LM (linearisointi hylatty)", "    LM#1: lineaarinen LM", "    LM#1: tarkka varmistus", "    LM: linearisoinnin rakennus", "    LM: residuaali+jacobi evaluaatiot",
    "LASKURI keskiarvo: runkopisteita/LM", "LASKURI keskiarvo: rengaspisteita/LM", "LASKURI keskiarvo: hull-kulmia/LM", "LASKURI keskiarvo: LM-iteraatioita/LM", "LASKURI keskiarvo: residuaalievaluaatioita/LM",
};
static std::atomic<long long> g_prof_ns[P_COUNT];
static std::atomic<long long> g_prof_n[P_COUNT];

static inline void profAdd(ProfId id, long long ns) { g_prof_ns[id] += ns; g_prof_n[id] += 1; }

static py::list prof_snapshot()
{
    py::list L;
    for (int i = 0; i < P_COUNT; ++i)
        L.append(py::make_tuple(std::string(PROF_NAMES[i]), (double)g_prof_ns[i].load() / 1e6, (long long)g_prof_n[i].load()));
    return L;
}

// Valiaikakello: lap() palauttaa edellisesta lap():sta kuluneen ajan (ns)
struct PT {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    long long lap() {
        auto t1 = std::chrono::steady_clock::now();
        long long d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        t0 = t1;
        return d;
    }
};

// Kiven valmistelussa etualamaski omassa saikeessaan saturaation + graniittimaskin rinnalla (sama tulos).
// Python: set_prep_parallel (tiedostoajo 1, live 0).
static std::atomic<int> g_prep_parallel{0};

// ============================================================
// OpenCV:n sisaisen rinnakkaistuksen poiskytkenta C++-saikeiden ajaksi (viitelaskuri).
// SEURANTA ajaa kivet omissa saikeissaan ja HAKU taustasaikeessa samaan aikaan; OpenCV:n oma rinnakkaistus kilpailisi
// niiden kanssa. cv::setNumThreads on prosessinlaajuinen, joten vain ensimmainen sisaanmenija asettaa 1:n ja vain
// viimeinen poistuja palauttaa alkuperaisen arvon. Elavassa vaiheessa (set_cv_single_thread_persistent(1)) arvo 1
// asetetaan kerran eika palauteta: Windowsin ConcRT-saiekehys loisi muuten saiepoolin uudelleen joka vaihdolla.
// ============================================================
namespace {
std::mutex g_cv_single_thread_mutex;
int g_cv_single_thread_refcount = 0;
int g_cv_single_thread_saved = 1;
bool g_cv_single_thread_persistent = false;
}

class ScopedSingleThreadedOpenCV {
public:
    ScopedSingleThreadedOpenCV() {
        std::lock_guard<std::mutex> lock(g_cv_single_thread_mutex);
        if (g_cv_single_thread_persistent) {
            g_cv_single_thread_refcount++;
            return;
        }
        if (g_cv_single_thread_refcount == 0) {
            g_cv_single_thread_saved = cv::getNumThreads();
            cv::setNumThreads(1);
        }
        g_cv_single_thread_refcount++;
    }
    ~ScopedSingleThreadedOpenCV() {
        std::lock_guard<std::mutex> lock(g_cv_single_thread_mutex);
        g_cv_single_thread_refcount--;
        if (g_cv_single_thread_persistent)
            return;
        if (g_cv_single_thread_refcount == 0)
            cv::setNumThreads(g_cv_single_thread_saved);
    }
    ScopedSingleThreadedOpenCV(const ScopedSingleThreadedOpenCV&) = delete;
    ScopedSingleThreadedOpenCV& operator=(const ScopedSingleThreadedOpenCV&) = delete;
};

static void setCvSingleThreadPersistent(bool on)
{
    std::lock_guard<std::mutex> lock(g_cv_single_thread_mutex);
    if (on && !g_cv_single_thread_persistent) {
        if (g_cv_single_thread_refcount == 0) g_cv_single_thread_saved = cv::getNumThreads();
        cv::setNumThreads(1);
        g_cv_single_thread_persistent = true;
    } else if (!on && g_cv_single_thread_persistent) {
        g_cv_single_thread_persistent = false;
        if (g_cv_single_thread_refcount == 0) cv::setNumThreads(g_cv_single_thread_saved);
    }
}

// ============================================================
// ALGORITMIN VAKIOT
// ============================================================

// graniittimaski: kiven pikseli on matalakylläinen ja paikallista taustaa tummempi
static const int STONE_MAX_SATURATION = 60;
static const double STONE_MIN_DARKNESS = 15.0;
static const double STONE_DARKNESS_SIGMA = 25.0;     // taustan arvion sumennus (px)
static const int GRANITE_DOWN = 4;                    // taustan arvio 4x pienennetylla kuvalla
static const int GRANITE_OPEN = 5;                    // maskin avaus (siluettitarkennus kayttaa 3)
static const int GRANITE_CLOSE = 3;                   // maskin sulkeminen

// rengaspisteet (kiven reuna saturaatiosta sateittain)
static const int BOUNDARY_N_ANGLES = 72;
static const double BOUNDARY_SATURATION_THRESHOLD = 70.0;
static const double BOUNDARY_RADIAL_STEP_PX = 0.25;
static const double BOUNDARY_RADIUS_SEARCH_FACTOR_MIN = 0.2;
static const double BOUNDARY_RADIUS_SEARCH_FACTOR_MAX = 2.6;
static const double BOUNDARY_MIN_PREDICTED_RADIUS_PX = 4.0;
static const int BOUNDARY_MIN_VALID_POINTS = 12;
static const double BOUNDARY_MIN_ANGULAR_SPREAD_DEG = 180.0;
static const int BOUNDARY_MAX_ITERATIONS = 5;
static const double BOUNDARY_CONVERGENCE_CM = 0.5;
static const double BOUNDARY_MAX_SHIFT_FROM_APPROX_CM = 25.0;

// runkokontuuri: lahin maskikontuuri kandidaattipaikan lahella. Hakuetaisyys 40 px (ei 60): muuten lahella oleva
// erillinen tumma kohde (pelaaja, vaatteen poimu) voisi tulla valituksi. Loydetyn kontuurin kokoa ei rajoiteta, koska
// juuri heitetyn / lakaistavan kiven maskikontuuri voi olla yhtenainen heittajan tai lakaisijan kanssa.
static const double BODY_CONTOUR_MIN_AREA_PX = 15.0;
static const double BODY_CONTOUR_MAX_SEARCH_DIST_PX = 40.0;
static const double BODY_HANDLE_CHECK_DIST_PX = 2.5;
static const int BODY_MIN_VALID_POINTS = 8;

static const int MASK_ROI_MARGIN_PX = 150;

// SEURANNAN laajenevan ristikkohaun pysaytyskynnys (peitto-osuus 95 %)
static const double TRACK_EARLY_STOP_SCORE = 0.95;

// Peitto-osuuden turvavyohyke: hulli 1.1-kertaisena omasta painopisteestaan. Vyohykkeella (hullin ulkopuolella)
// olevat maskipikselit vahennetaan pistemaarasta (pelaaja/harja tai viereinen kivi ei saa tayttä pistemaaraa).
static const double HULL_MARGIN_SCALE = 1.10;
static const double HULL_OUTSIDE_PENALTY_WEIGHT = 1.0;

// ============================================================
// numpy -> OpenCV
// ============================================================
static std::vector<cv::Point3d> parsePts3(ArrD& arr)
{
    auto b = arr.unchecked<2>();
    std::vector<cv::Point3d> out((size_t)b.shape(0));
    for (py::ssize_t i = 0; i < b.shape(0); ++i)
        out[(size_t)i] = cv::Point3d(b(i, 0), b(i, 1), b(i, 2));
    return out;
}

static cv::Matx33d parseMat33(ArrD& arr)
{
    auto b = arr.unchecked<2>();
    cv::Matx33d M;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j) = b(i, j);
    return M;
}

static cv::Vec3d parseVec3(ArrD& arr)
{
    auto b = arr.unchecked<1>();
    return cv::Vec3d(b(0), b(1), b(2));
}

// HxWx3 uint8 -> cv::Mat (ei kopiota); tyhja Mat jos muoto ei ole BGR-kuva
static cv::Mat bgrView(ArrU8& arr)
{
    auto buf = arr.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        return cv::Mat();
    return cv::Mat((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, (void*)buf.ptr);
}
