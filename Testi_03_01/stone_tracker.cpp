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

// ============================================================
// AIKAMITTAUKSET (Testi_03_01): jokainen seurannan/HAKU:n vaihe kirjaa kokonaisajan ja kutsumaaran
// globaaleihin (atomic) laskureihin. prof_reset()/prof_snapshot() Pythonista. Ajastimen oma kustannus
// on ~50 ns / mittaus, eli mitätön verrattuna vaiheisiin (ms). Säikeissä ajoaika summataan
// (CPU-aika), joten P_BATCH_WALL vs. summa kertoo rinnakkaistuksen hyodyn.
// ============================================================
enum ProfId {
    P_CALL_TOTAL, P_PREP_CROP_SUPPRESS, P_PREP_SAT, P_PREP_GRANITE, P_PREP_FOREGROUND,
    P_LOC_GRID, P_LOC_MS1, P_LOC_MS2, P_LOC_ENS_GRID, P_LOC_TOTAL,
    P_REFINE_TOTAL, P_R_PAD, P_R_CONTOUR, P_R_BOUNDARY, P_R_LM1, P_R_MAD, P_R_LM2, P_R_FINAL, P_R_OUTER_ITER,
    P_HAKU_TOTAL, P_HAKU_SUPPRESS, P_HAKU_MASK, P_HAKU_SAT, P_HAKU_LOCATE, P_HAKU_REFINE, P_HAKU_FLOOD,
    P_BATCH_WALL, P_BATCH_SETUP, P_BATCH_RESULT,
    P_LM_FALLBACK, P_LM_LIN, P_LM_VERIFY, P_LM_BUILD, P_LM_EVAL_ONLY,
    P_C_NBODY, P_C_NRING, P_C_NHULL, P_C_LMITER, P_C_LMEVAL,
    P_COUNT
};
static const char* PROF_NAMES[P_COUNT] = {
    "seuranta/kivi-paivitys yhteensa (CPU)", "  prep: rajaus + taustanvaimennus", "  prep: saturaatio", "  prep: graniittimaski (GaussianBlur)",
    "  prep: etualamaski (whitened)", "  haku: ristikkohaku", "  haku: mean-shift (viimeisesta)", "  haku: mean-shift (ennustetusta)",
    "  haku: ristikko (yhdistelma, ennustettu keskipiste)", "  haku yhteensa", "  LM-tarkennus yhteensa", "    LM: reunustus (copyMakeBorder)",
    "    LM: runkokontuuri + jaasuodatus", "    LM: reunapisteet (detectBoundaryPoints)", "    LM: sovitus #1", "    LM: MAD-poikkeamat",
    "    LM: sovitus #2 (uusinta)", "    LM: lopullinen residuaali", "    LM: ulkoiteraatioita (lkm)",
    "HAKU yhteensa", "  HAKU: taustanvaimennus (koko frame)", "  HAKU: graniittimaski (koko frame)", "  HAKU: saturaatio (koko frame)",
    "  HAKU: ristikkohaku", "  HAKU: LM-tarkennus", "  HAKU: floodFill-poisto",
    "SEURANTA-kutsu seinakello (track_stones_batch)", "  kutsun alustus (numpy->cv, GIL)", "  kutsun tulokset (dict)",
    "    LM: TARKKA VARAKEINO-LM (linearisointi hylatty)", "    LM#1: lineaarinen LM", "    LM#1: tarkka varmistus", "    LM: linearisoinnin rakennus", "    LM: residuaali+jacobi evaluaatiot",
    "LASKURI keskiarvo: runkopisteita/LM", "LASKURI keskiarvo: rengaspisteita/LM", "LASKURI keskiarvo: hull-kulmia/LM", "LASKURI keskiarvo: LM-iteraatioita/LM", "LASKURI keskiarvo: residuaalievaluaatioita/LM"
};
static std::atomic<long long> g_prof_ns[P_COUNT];
static std::atomic<long long> g_prof_n[P_COUNT];

static inline void profAdd(ProfId id, long long ns) { g_prof_ns[id] += ns; g_prof_n[id] += 1; }

struct PT {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    long long lap() {
        auto t1 = std::chrono::steady_clock::now();
        long long d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        t0 = t1;
        return d;
    }
};



// ============================================================
// OpenCV:n sisaisen rinnakkaistuksen turvallinen, VIITELASKURILLINEN
// poiskytkenta (kayttajan pyynnosta: HAKU ja SEURANTA ajetaan nyt
// samanaikaisesti kahdesta eri saikeesta - main.py:n elava seuranta).
//
// TAUSTA: cv::setNumThreads on PROSESSINLAAJUINEN (globaali) tila,
// EI saiekohtainen. track_stones_batch (SEURANTA) kytkee sen pois
// PAALTA omaa per-kivi-saiejakoaan varten (muuten OpenCV:n oma
// sisainen rinnakkaistus kilpailisi jokaisen per-kivi-saikeen SISALLA
// - mitattu: 1 kivi ~33ms, mutta 2 kiveä ~107ms, PAHEMPI kuin
// sarjallisesti). Ennen taman kayttoonottoa search_new_stone (HAKU)
// EI kytkenyt sita pois lainkaan, koska se ajettiin AINA sarjallisesti
// - koskaan SAMAAN AIKAAN track_stones_batch:in kanssa.
//
// Nyt kun HAKU ja SEURANTA voivat olla kaynnissa SAMANAIKAISESTI ERI
// SAIKEISTA, naiivi per-kutsu "tallenna alkuperainen -> aseta 1 ->
// palauta alkuperainen" (kummallakin funktiolla OMA, TOISISTAAN
// TIETAMATON versio) EI OLE TURVALLINEN: jos molemmat ovat kaynnissa
// yhtaaikaa, jalkimmainen sisaanmenija voi lukea jo ykkoseksi
// asetetun arvon "alkuperaiseksi" arvokseen, ja palauttaessaan
// TAMAN saastaa OpenCV:n rinnakkaistuksen PYSYVASTI 1 saikeeseen
// LOPUKSI ASTI - HILJAA, ilman virhetta, huomattavasti hidastaen
// KAIKKEA myohempaa (warpAffine+remap, paneiliseuranta, jne.).
//
// Korjaus: viitelaskuri mutexin takana - VAIN ensimmainen sisaan-
// menija tallentaa alkuperaisen arvon ja asettaa 1:n, VAIN viimeinen
// poistuja palauttaa sen - valissa olevat samanaikaiset kutsujat
// vain kasvattavat/vahentavat laskuria eivatka koske itse arvoon.
// ============================================================

namespace {
std::mutex g_cv_single_thread_mutex;
int g_cv_single_thread_refcount = 0;
int g_cv_single_thread_saved = 1;
}

class ScopedSingleThreadedOpenCV {
public:
    ScopedSingleThreadedOpenCV() {
        std::lock_guard<std::mutex> lock(g_cv_single_thread_mutex);
        if (g_cv_single_thread_refcount == 0) {
            g_cv_single_thread_saved = cv::getNumThreads();
            cv::setNumThreads(1);
        }
        g_cv_single_thread_refcount++;
    }
    ~ScopedSingleThreadedOpenCV() {
        std::lock_guard<std::mutex> lock(g_cv_single_thread_mutex);
        g_cv_single_thread_refcount--;
        if (g_cv_single_thread_refcount == 0) {
            cv::setNumThreads(g_cv_single_thread_saved);
        }
    }
    ScopedSingleThreadedOpenCV(const ScopedSingleThreadedOpenCV&) = delete;
    ScopedSingleThreadedOpenCV& operator=(const ScopedSingleThreadedOpenCV&) = delete;
};


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
// Pienennetty 60:sta 40:een (kayttajan pyynnosta, katso git-historia):
// vahentaa toistuvasti havaittua virhetta, jossa ristikkohaku osuu
// pelaajan vaatetukseen ja findContourNear nappaa TAYSIN ERILLISEN,
// vain sattumalta lahella olevan tumman kontuurinpalan (esim. toinen
// pelaaja tai vaatteen poimu) sen sijaan etta hylkaisi liian isona.
// HUOM: refinePositionJoint:issa (alempana) ollut ERI suoja (LOYDETYN
// kontuurin maksimikoko, BODY_CONTOUR_MAX_AREA_MULTIPLIER) on sittemmin
// POISTETTU kokonaan (kayttajan pyynnosta, katso sen oma kommentti) -
// katso talta samalta periaatteelta: juuri heitetyn/lakaistavan kiven
// oma maskikontuuri voi olla YHTENAINEN heittajan/lakaisijan kanssa
// (kayttaja: "heittäjä näkyy kiven yli alussa") - EI haluta hylata
// oikeaa kivea vain siksi etta jotain muutakin nakyy sen vieressa/
// paalla, kunhan LOYTYNYT kontuuri ITSE on lahella kandidaattipaikkaa.
static const double BODY_CONTOUR_MAX_SEARCH_DIST_PX = 40.0;
static const double BODY_HANDLE_CHECK_DIST_PX = 2.5;
static const int BODY_MIN_VALID_POINTS = 8;

static const int MASK_ROI_MARGIN_PX = 150;

// SEURANTA:n laajenevan ristikkohaun pysaytyskynnys (UUSI, EI
// Python-porttaus - katso locateByGridSearchTrackingFast:in oma
// kommentti). Tunnettu vakio jota voi saataa mitatun datan
// perusteella - 0.95 tarkoittaa "peitto-osuus lahella tayttaa
// (95%) testihullon".
static const double TRACK_EARLY_STOP_SCORE = 0.95;

// HULLIN ULKOPUOLISTEN LOYDOSTEN MIINUSPISTEET (UUSI, EI Python-
// porttaus - kayttajan pyynnosta): hullOverlapScore oli aiemmin
// YKSISUUNTAINEN - se mittasi VAIN kuinka suuri osa ennustetun
// hullin SISALLA olevista pikseleista loytyy maskista, muttei
// mitenkaan rangaissut siita etta maski jatkuu reilusti hullin
// ULKOPUOLELLE (esim. pelaaja/harja tai kaksi toisiinsa koskettavaa
// kivea, jotka voisivat silti antaa taydellisen 1.0-pistemaaran
// koska KAIKKI hullin pikselit sattuvat olemaan maskissa). Lisatty
// "turvavyohyke" (hulli laajennettuna HULL_MARGIN_SCALE:lla omasta
// painopisteestaan) - vyohykkeen SISALLA (mutta alkuperaisen hullin
// ULKOPUOLELLA) loydetyt maskipikselit vahennetaan pistemaarasta
// painotettuna HULL_OUTSIDE_PENALTY_WEIGHT:lla. Hullin skaalaus
// 2D:ssa omasta painopisteestaan (ei 3D-sateen kautta) on tietoinen
// yksinkertaistus - riittavan tarkka pienelle (10%) marginaalille,
// eika vaadi K/R/t-projektiota tai 3D-pisteita tahan funktioon.
static const double HULL_MARGIN_SCALE = 1.10;
static const double HULL_OUTSIDE_PENALTY_WEIGHT = 1.0;


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
    // OPTIMOINTI (EI numeerista eroa): alkuperainen versio rakensi
    // ensin VALIAIKAISEN pts-taulukon (Point3d, X0/Y0-siirrolla) ja
    // kutsui sitten project3d:ta joka rakensi OMAN valiaikaisen
    // out-taulukkonsa (Point2d) - eli kaksi ylimaarasta heap-varausta
    // JOKAISELLA kutsulla, vaikka lopputulos on aina vain proj_f
    // (Point2f). Tama funktio on JOKAISEN LM-residuaalilaskennan
    // (profileResiduals) sisalla, siis satoja kertoja per kiven
    // paivitys (katso jointResiduals) - siirretty tassa suoraan
    // paikallisesta pisteesta lopulliseen projisoituun Point2f:aan
    // ILMAN valivaiheiden materialisointia. Laskentajarjestys/
    // liukulukuoperaatiot tarkalleen samat kuin ennen (X0/Y0-lisays
    // ENNEN R*p+t:ta, K*pc JÄLKEEN, jako pi[2]:lla, sitten float-
    // kasti VASTA aivan lopuksi, kuten alkuperaisessakin) - siis
    // BITTITARKASTI sama tulos, vain vahemman valiaikaista muistin-
    // varausta.
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

    // Turvavyohyke (UUSI, EI Python-porttaus): hulli skaalattuna
    // HULL_MARGIN_SCALE:lla OMASTA PAINOPISTEESTAAN - suhteellinen
    // (pikseleina hullin omaan kokoon nahden, EI kiintea cm-etaisyys),
    // katso TAMAN FUNKTION ylla oleva HULL_MARGIN_SCALE-kommentti.
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

    std::vector<cv::Point> shifted(hull_int.size());
    for (size_t i = 0; i < hull_int.size(); ++i)
        shifted[i] = cv::Point(hull_int[i].x - x0, hull_int[i].y - y0);

    std::vector<cv::Point> margin_shifted(margin_int.size());
    for (size_t i = 0; i < margin_int.size(); ++i)
        margin_shifted[i] = cv::Point(margin_int[i].x - x0, margin_int[i].y - y0);

    cv::Mat canvas = cv::Mat::zeros(y1 - y0, x1 - x0, CV_8UC1);
    std::vector<std::vector<cv::Point>> polys{shifted};
    cv::fillPoly(canvas, polys, cv::Scalar(255));

    int hull_area = cv::countNonZero(canvas);
    if (hull_area == 0)
        return 0.0;

    cv::Mat margin_canvas = cv::Mat::zeros(y1 - y0, x1 - x0, CV_8UC1);
    std::vector<std::vector<cv::Point>> margin_polys{margin_shifted};
    cv::fillPoly(margin_canvas, margin_polys, cv::Scalar(255));

    cv::Mat mask_sub = mask_crop(cv::Rect(x0, y0, x1 - x0, y1 - y0));

    int overlap = 0;
    int outside_overlap = 0;
    int band_area = 0;

    for (int r = 0; r < canvas.rows; ++r) {
        const uchar* cptr = canvas.ptr<uchar>(r);
        const uchar* mptr = mask_sub.ptr<uchar>(r);
        const uchar* bptr = margin_canvas.ptr<uchar>(r);
        for (int c = 0; c < canvas.cols; ++c) {
            bool inside = cptr[c] > 0;
            bool in_margin = bptr[c] > 0;
            bool in_band = in_margin && !inside;

            if (inside && mptr[c] > 0)
                ++overlap;

            if (in_band) {
                ++band_area;
                if (mptr[c] > 0)
                    ++outside_overlap;
            }
        }
    }

    double inside_score = (double)overlap / (double)hull_area;
    double outside_score = band_area > 0 ? (double)outside_overlap / (double)band_area : 0.0;

    return inside_score - HULL_OUTSIDE_PENALTY_WEIGHT * outside_score;
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


// HAKU:n ristikkohaun nopeutus (kayttajan ehdotus, katso keskustelu-
// historia): HAKU:n hakuvyohyke on kiintea, KAPEA (x_half_width=70cm)
// ja KAUKANA kamerasta (lahella kaukaista hogline:a, katso kamera9_02.
// py:n SEARCH_Y_MIN/MAX_CM) - talla etaisyydella kiven projisoitu KOKO
// JA MUOTO eivat muutu merkittavasti koko hakuvyohykkeen sisalla, joten
// jokaiselle ristikkopisteelle EI TARVITSE laskea TAYTTA 3D->2D-
// projisiota (predictedHull, joka projisioi KAIKKI local_pts_search-
// pisteet erikseen) uudelleen - riittaa laskea taydellinen hulli VAIN
// KERRAN hakuvyohykkeen keskella (x_center,y_center - kaukaisen hog-
// linen kohdalla) ja SIIRTAA (translatoida) tata samaa hullia muille
// ristikkopisteille paikallisen lineaarisen approksimaation (Jacobin
// matriisi px/cm, laskettu differenssilla samasta keskipisteesta)
// mukaan. HUOM: tama approksimaatio EI PADE koko radalle (lahella
// kameraa/taloa projisio muuttuu paljon nopeammin) - siksi SEURANNAN
// oma ristikkohaku (locateByGridSearchTrackingFast, jonka kohde voi
// olla missa tahansa radalla) EI kayta tata, vain gridSearchBest:ia
// suoraan taydella projisiolla joka pisteessa.
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


static std::pair<cv::Point2d, double> locateByGridSearchFast(
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Mat& mask_crop, int off_x, int off_y,
    double x_center, double x_half_range, double y_center, double y_half_range,
    double coarse_step, double fine_step,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t)
{
    // Referenssihulli + paikallinen Jacobi (px per cm) hakuvyohykkeen
    // keskella - katso gridSearchBestLinearized:in oma kommentti.
    // JAC_DELTA_CM: pieni siirtyma differenssille - ei vaikuta tulokseen
    // (lineaarinen approksimaatio joka tapauksessa), vain numeeriseen
    // tarkkuuteen.
    static const double JAC_DELTA_CM = 10.0;

    auto ref_hull = predictedHull(local_pts_search, x_center, y_center, K, R, t);

    std::vector<cv::Point3d> jac_pts{
        cv::Point3d(x_center, y_center, 0.0),
        cv::Point3d(x_center + JAC_DELTA_CM, y_center, 0.0),
        cv::Point3d(x_center, y_center + JAC_DELTA_CM, 0.0),
    };
    // Referenssikorkeus ei ole taalla merkityksellinen (vain siirtymän
    // suunta/mittakaava), mutta kaytetaan samaa Z=0-tasoa kuin
    // rayPlaneIntersectionZ0 - riittava tarkkuus Jacobille.
    auto jac_proj = project3d(K, R, t, jac_pts);
    cv::Point2d jac_col_x(
        (jac_proj[1].x - jac_proj[0].x) / JAC_DELTA_CM,
        (jac_proj[1].y - jac_proj[0].y) / JAC_DELTA_CM
    );
    cv::Point2d jac_col_y(
        (jac_proj[2].x - jac_proj[0].x) / JAC_DELTA_CM,
        (jac_proj[2].y - jac_proj[0].y) / JAC_DELTA_CM
    );

    auto x_vals = arangeVec(x_center - x_half_range, x_center + x_half_range + 1e-6, coarse_step);
    auto y_vals = arangeVec(y_center - y_half_range, y_center + y_half_range + 1e-6, coarse_step);
    auto best1 = gridSearchBestLinearized(
        mask_crop, off_x, off_y, ref_hull, x_center, y_center, jac_col_x, jac_col_y, x_vals, y_vals
    );

    auto x_vals2 = arangeVec(best1.first.x - coarse_step, best1.first.x + coarse_step + 1e-6, fine_step);
    auto y_vals2 = arangeVec(best1.first.y - coarse_step, best1.first.y + coarse_step + 1e-6, fine_step);
    auto best2 = gridSearchBestLinearized(
        mask_crop, off_x, off_y, ref_hull, x_center, y_center, jac_col_x, jac_col_y, x_vals2, y_vals2
    );

    return best2;
}


// ============================================================
// SEURANTA:N LAAJENEVA KARKEA HAKU (UUSI, EI Python-porttaus -
// kayttajan pyynnosta): Pythonin/HAKU:n locateByGridSearchFast
// kay AINA koko TRACK_HALF_RANGE_CM:n laatikon (esim. 11x11=121
// pistetta) lapi, vaikka kivi useimmiten liikkuu vain vahan
// framejen valilla ja oikea osuma loytyy jo aivan edellisen
// sijainnin lahelta - jatkuvuus on jo vahva prior (katso
// TRACK_SCORE_THRESHOLD:in oma kommentti kamera9_02.py:ssa).
//
// Tama funktio laajenee EDELLISESTA sijainnista (x_center,
// y_center) ULOSPAIN rengas kerrallaan (Tsebysevin etaisyys
// ruudukkoindekseissa) ja PYSAHTYY heti kun loytyy tarpeeksi hyva
// osuma (score >= TRACK_EARLY_STOP_SCORE), minka jalkeen viela
// tarkentaa PAIKALLISELLA kukkulankiipeilylla (tarkistaa muutaman
// pisteen - 8-naapurusto samalla ruudukkoresoluutiolla - loydetyn
// pisteen ymparilta, siirtyy sinne jos parempi loytyy, ja toistaa
// kunnes mikaan naapuri ei ole parempi). Jos tarpeeksi hyvaa osumaa
// EI loydy, koko sama laatikko (samat ehdokaspisteet kuin
// locateByGridSearchFast:issa) kaydaan silti kokonaan lapi ennen
// luovuttamista - eli PAHIMMASSA tapauksessa (esim. kivi tormaa
// toiseen tai katoaa hetkeksi peittoon) kattavuus on TASMALLEEN
// sama kuin ennen, ei regressiota.
//
// HUOM: koska pisteet kaydaan lapi ERI JARJESTYKSESSA (rengas
// keskelta ulospain, ei rivi kerrallaan) kuin Pythonin/alkuperaisen
// exhaustive-haun, se MIKA piste voittaa TASAPELISSA (kaksi pistetta
// tasan sama pistemaara) voi silloin harvinaisissa tapauksissa
// erota - tama ei ole regressio vaan tarkoituksellinen, hyvaksytty
// ero (katso myos tiedoston alun kommentti). Kaytetaan VAIN SEURANTA:
// ssa (trackStoneUpdateOne) - HAKU:ssa (searchNewStoneOne) EI ole
// vastaavaa "edellinen sijainti" -prioria, joten se kayttaa edelleen
// muuttamatonta, tasmalleen Pythonia vastaavaa locateByGridSearchFast:
// ia.
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

        auto evalPoint = [&](int i, int j) -> double {
            visited[idx(i, j)] = 1;
            auto hull = predictedHull(local_pts_search, x_vals[(size_t)i], y_vals[(size_t)j], K, R, t);
            return hullOverlapScore(mask_crop, hull, off_x, off_y);
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
// ETUALAMASKI VALMIIKSI VAIMENNETUSTA KUVASTA (UUSI, EI Python-
// porttaus - kayttajan pyynnosta, katso keskusteluhistoria):
// suppressStaticBackground (ja Pythonin ENABLE_SHADOW_TOLERANT_
// STABILIZATION-polun oma, varjonsietoisempi vastine main.py:ssa,
// jonka tulos ANNETAAN TASSA jo valmiiksi crop_filtered:ina) VALKAISEE
// (asettaa TASMALLEEN (255,255,255):ksi) kaikki taustaa vastaavat
// pikselit - siis "ei-valkoinen" crop_filtered:issa ON JO taman
// olemassaolevan, jo validoidun vaimennuslogiikan oma etuala-paatos,
// riippumatta kumpi (Python-taso vai tama C++-taso, tai molemmat)
// sen teki. Kaytetaan SEURANNASSA (trackStoneUpdateOne) YHDISTETTYNA
// (OR) rakeisuusmaskiin (createGraniteMask) - juurisyyanalyysi (katso
// keskusteluhistoria) osoitti etta pelkka rakeisuusmaski voi olla
// HYVIN HARVA pitkalla etaisyydella (esim. ohuen sivultapain
// nakyvan kiven reunalla, jossa paikallinen tummuuskontrasti
// darkness-testille on heikko) - talloin refinePositionJoint:in
// runkopisteiden (n_body) maara putoaa niin alas etta sen oma
// MAD-pohjainen poikkeavien hylkays (katso taman tiedoston alun
// kommentti - se ITSESSAAN on epajatkuva funktio pienella
// pistemaaralla) alkaa hypahdella framesta toiseen VAIKKA kivi
// liikkuisi tasaisesti - mitattu oikealla datalla: X-suunnan
// sijainti hyppii jopa n. 5-6cm yhden framen valilla kiven kulkiessa
// yksin, ilman mitaan naapurikohdetta lahella (varmistettu
// vertaamalla molempia maskeja silmamaarin taman jakson framista).
// Taustanvaimennus loytaa saman kiven riippumatta paikallisesta
// varikontrastista, joten yhdistetty (rakeisuus OR etuala) maski
// antaa TIHEAMMAN, siten VAKAAMMAN pistejoukon findContourNear:lle -
// juurisyy (harva pistejoukko + jyrkka MAD-kynnys) korjataan siis
// TIHENTAMALLA syotemaski, ei muuttamalla itse MAD-logiikkaa.
//
// HUOM (tarkeaa): TATA EI SAA laskea suoraan absdiff(crop, ref):sta
// omalla erillisella kynnyksellaan, koska kutsuja saattaa antaa
// diff_threshold=0.0 (main.py:n haku_seuranta_diff_threshold, kun
// ENABLE_SHADOW_TOLERANT_STABILIZATION=True - katso trackStoneUpdate-
// One:n omaa kutsua alla) TARKOITUKSELLA ohittaakseen taman C++:n
// SISAISEN, YKSINKERTAISEMMAN vaimennuksen (koska Python on jo tehnyt
// paremman, varjonsietoisen vaimennuksen FRAME:LLE ITSELLEEN ennen
// kutsua) - jos tama funktio laskisi oman diff-testinsa uudelleen
// diff_threshold:lla, 0.0 tekisi JOKAISESTA pikselista "etualaa"
// (koska absdiff >= 0.0 patee aina), mika hajottaisi koko haun
// (havaittu talla TASMALLEEN talla tavalla ensimmaisessa
// A/B-testauksessa - katso keskusteluhistoria). Lukemalla etuala
// SUORAAN jo-vaimennetun crop_filtered:in valkoisuudesta valtetaan
// tama kokonaan, koska crop_filtered ITSE on jo oikein laskettu
// (kummalla tahansa vaimennuspolulla) ENNEN taman funktion kutsua.
// ============================================================

static cv::Mat createForegroundFromWhitened(const cv::Mat& crop_filtered_bgr)
{
    cv::Mat mask(crop_filtered_bgr.size(), CV_8UC1);
    for (int r = 0; r < mask.rows; ++r) {
        const cv::Vec3b* pptr = crop_filtered_bgr.ptr<cv::Vec3b>(r);
        uchar* mp = mask.ptr<uchar>(r);
        for (int c = 0; c < mask.cols; ++c) {
            const cv::Vec3b& px = pptr[c];
            bool is_whitened = (px[0] == 255 && px[1] == 255 && px[2] == 255);
            mp[c] = is_whitened ? 0 : 255;
        }
    }
    return mask;
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

// KOKORAJA (kayttajan pyynnosta, ei kamera9_03.py:ssa - taman
// tiedoston OMA lisays, EI porttaus): _find_contour_near hyvaksyi
// aiemmin LAHIMMAN riittavan ISON kontuurin ilMAN ylarajaa - jos
// lahin kontuuri sattui olemaan esim. pelaajan/lakaisijan siluetti
// (havaittu kayttajan omasta datasta: n_body hyppasi 27:sta yli
// 250:een, RMS 1.9px:sta yli 15px:iin, kun SEURANTA harhautui
// kiveltä kyykistyneeseen pelaajaan), se hyvaksyttiin siina missa
// oikea kivikin. max_area rajaa taman - laskettu KUTSUJASSA kiven
// oman, taman kandidaattipaikan projisoidun rungon pinta-alasta
// (etaisyys kamerasta huomioitu automaattisesti, koska tama on
// sama projisointi jota ristikkohaku/sovitus jo kayttavat), joten
// sama absoluuttinen kerroin toimii kiven koko liu'un matkalla.
static bool findContourNear(
    const cv::Mat& mask, cv::Point2d approx_px, std::vector<cv::Point>& best,
    double max_dist_px = BODY_CONTOUR_MAX_SEARCH_DIST_PX,
    double min_area = BODY_CONTOUR_MIN_AREA_PX,
    double max_area = -1.0,
    bool* rejected_for_size = nullptr)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    bool found = false;
    double best_dist = 0.0;

    // Erikseen seurataan lahin kontuuri joka olisi muuten kelvannut
    // (min_area+etaisyys tayttyy) mutta hylattiin YKSINOMAAN liian
    // suuren pinta-alan takia - talla erotetaan "ei loytynyt mitaan
    // lahella" (rejected_for_size=false, ennallaan) tapauksesta
    // "loytyi jotain lahella mutta se on liian iso" (rejected_for_
    // size=true) - kutsuja voi kayttaa jalkimmaista pakottamaan koko
    // havainnon hylkays (eika vain "epatarkka") sen sijaan etta
    // palataan takaisin arvattuun/hilahaun sijaintiin (joka tassa
    // tapauksessa todennakoisesti ON se liian iso kohde, esim.
    // pelaaja).
    bool oversized_found = false;
    double oversized_best_dist = 0.0;

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

        if (max_area > 0.0 && area > max_area) {
            if (!oversized_found || d < oversized_best_dist) {
                oversized_found = true;
                oversized_best_dist = d;
            }
            continue;
        }

        if (!found || d < best_dist) {
            best = c;
            best_dist = d;
            found = true;
        }
    }

    if (rejected_for_size != nullptr)
        *rejected_for_size = (!found && oversized_found);

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

// Kahvan aiheuttaman kolon sade suhteessa R_max:iin - kolon SIJAINTI on
// MAARITELTY (keskitetty kiven pystyakselille (X0,Y0) korkeudella
// z=H_total), mutta SADE (handle_r_frac) on kayttajan pyynnosta
// AJONAIKAINEN PARAMETRI, sovitettu Pythonin fit_stone_profile:ssa
// (kamera9_01.py:n HANDLE_NOTCH_R_FRAC-kommentti) ja annettu tanne
// jokaisen kutsun mukana (R_max_cm/H_total_cm:n tapaan) - EI enaa
// kiintea C++-vakio.
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

// kamera9_01.py:n _signed_dist_with_notch:in porttaus - etumerkillinen
// etaisyys graniittirungon (hull) MIINUS kahvan kolon (notch_hull)
// reunaan. Positiivinen SISALLA todellisessa (kolollisessa) muodossa,
// negatiivinen ULKOPUOLELLA (joko kokonaan hullin ulkopuolella TAI
// kolon SISALLA, koska kolo on POIS LEIKATTU alue).
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

// OPTIMOINTI (EI numeerista eroa): pts otetaan nyt VALMIIKSI Point2f:
// ksi kasteltuna (katso toPoint2fVec) - profileResiduals on jointResi
// duals:in kautta LM:n Jacobian/damping-silmukan sisalla, kutsuttuna
// satoja kertoja per refinePositionJoint-kutsu (katso jointResiduals:in
// oma kommentti), mutta pts (body_pts/clean_body) EI MUUTU LAINKAAN
// naiden kutsujen valilla - vain X0/Y0 muuttuu. Alkuperainen versio
// teki double->float-kastin JOKAISELLE pisteelle JOKAISELLA kutsulla
// UUDELLEEN, vaikka tulos on aina sama - kutsuja (refinePositionJoint)
// kastaa nyt kerran ja uudelleenkayttaa. cv::Point2f((float)x,(float)y)
// on deterministinen - sama kutsu tuottaa AINA saman bittitarkan
// tuloksen laskettiinpa se kerran tai monta kertaa, joten tama EI
// muuta yhtaan lukua.
//
// HUOM (kayttajan pyynnosta lisatty kahvan kolo): TAMA on TARKKA
// residuaalifunktio, ja on NYT kolotietoinen (predictedNotchHull +
// signedDistWithNotch). levenbergMarquardt3Linearized:in OMA,
// linearisoitu nopea polku (buildLinearizedHull/evalLinearizedHull)
// EI ole paivitetty kolotietoiseksi - se mallintaa vain konveksin
// ulkorungon, ei koloa. Tama on TIETOINEN rajaus: levenbergMarquardt3
// LinearizedVerified vertaa linearisoidun tuloksen TARKKAA (tata
// funktiota kayttavaa) kustannusta lahtopisteen tarkkaan kustannukseen
// ja hylkaa linearisoidun tuloksen taydeksi tarkaksi LM:ksi jos se ei
// ole vahintaan yhta hyva - kolon aiheuttama poikkeama nakyy siis AINA
// oikein lopullisessa hyvaksytyssa tuloksessa, vaikka linearisoitu
// nopea polku itse ei sita "nae" hakiessaan askeltaan.
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
    // OPTIMOINTI (EI numeerista eroa): sama periaate kuin predicted
    // Hull:issa - vaivaiset pts3d/proj-valitaulukot poistettu, suora
    // pisteesta Point2f:aan -projisointi tasmalleen samassa jarjes-
    // tyksessa kuin ennen.
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


// OPTIMOINTI (EI numeerista eroa): sama periaate kuin profileResi
// duals:issa - observed otetaan valmiiksi Point2f:ksi kasteltuna.
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


// ============================================================
// LEVENBERG-MARQUARDT (kamera8_01.py:n _levenberg_marquardt,
// erikoistettu 3 parametrille X,Y,R - sama numeerinen Jacobian +
// vaimennettu pienimman nelion periaate, ei matemaattista eroa).
// ============================================================

// Yksi double->float-kasti pistejoukolle - katso profileResiduals/
// ringPointResiduals:in oma kommentti: kutsuja kastaa TASAN kerran
// per body_pts/ring_pts/clean_body/clean_ring -joukko (nama pysyvat
// muuttumattomina koko LM-ajon ajan), ei jokaisella residuaali-
// kutsulla uudelleen.
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
        // ring_height_cm == H_total_cm (katso refinePositionJoint:in
        // "double ring_height_cm = H_total_cm;") - uudelleenkaytetty
        // tassa H_total_cm:na, ei oma erillinen kenttansa.
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


// ============================================================
// LINEARISOITU LM (UUSI, EI Python-porttaus - kayttajan pyynnosta,
// mitattu ja visuaalisesti tarkistettu): projisointi pikseli =
// K*(R_kam*(paikallinen+X,Y,0)+t), jako z:lla - koska R_kam*(X,Y,0)
// = X*R_kam:n 1. sarake + Y*R_kam:n 2. sarake, tama on LINEAARINEN
// X:n/Y:n suhteen ENNEN jakoa, joten derivaatta d(pikseli)/dX ja
// d(pikseli)/dY on SULJETUSSA MUODOSSA (osamaaran derivaatta) - ei
// vaadi YHTAAN ylimaaraista projektiota, vain referenssipisteen omat
// pi0,pi1,pi2-arvot (jotka joka tapauksessa lasketaan taysessa
// projektiossa). Sama patee rengaspisteiden R-parametrille (rengas-
// piste = R*cos(theta),R*sin(theta) - myos lineaarinen R:n suhteen).
//
// Kaytannossa: yksi TAYSI projektio (+convexHull runko-osalle) per
// LM-kutsu (levenbergMarquardt3Linearized:in alussa, params0:lla), ja
// SEN JALKEEN jokainen myohempi (X,Y,R)-kokeilu (Jacobian-sarakkeet +
// damping-yritykset - satoja per refinePositionJoint-kutsu, katso
// mitattu jakauma alempana) approksimoidaan HALVALLA lineaarikaavalla
// taysen projektion+convexHull:in sijaan.
//
// Tama ON approksimaatio (ensimmaisen kertaluvun Taylor-kehitelma),
// jota TARKKA VARMISTUS suojaa (katso levenbergMarquardt3Linearized
// Verified): koska referenssipisteessa (dX=dY=dR=0) approksimaatio on
// TASMALLEEN tarkka, lahtokustannus saadaan "ilmaiseksi" TARKALLA
// residuaalifunktiolla - jos linearisoidun LM:n lopputulos ei ole
// vahintaan yhta hyva TARKASTI mitattuna kuin lahtopiste, approksi-
// maatio hylataan ja koko LM ajetaan uudestaan TARKASTI (varakeino).
// Tama takaa etta linearisoitu polku ei voi koskaan tuottaa TARKKAA
// LM:aa huonompaa tulosta.
//
// HUOM (validoitu mittaamalla + visuaalisesti tarkistamalla oikeista
// videoframeista): koska refinePositionJoint:in MAD-poikkeavien-
// hylkays (3.0*1.4826*MAD) on itsessaan EPAJATKUVA funktio residuaa-
// leista (katso tiedoston alun kommentti - sama ilmio kuin Python vs.
// tarkka C++ -erossa), jopa VARMISTETTU linearisoitu askel voi
// paatya ERI (mutta yhta patevaan) paikalliseen minimiin kuin tarkka
// LM - tama EI ole approksimaatiovirhe vaan koko algoritmin sisaan-
// rakennettu, jo ennestaan tunnettu numeerinen herkkyys. Molemmat
// ratkaisut tarkistettu visuaalisesti (3D-mallin projisointi oikealle
// videoframelle) yhta valideiksi naissa tapauksissa.
// ============================================================

static const bool g_lm_notch = getenv("LM_NOTCH") ? atoi(getenv("LM_NOTCH")) != 0 : true;
static const int g_body_stride = getenv("LM_BODY_STRIDE") ? atoi(getenv("LM_BODY_STRIDE")) : 2;
static const double g_hull_eps = getenv("LM_HULL_EPS") ? atof(getenv("LM_HULL_EPS")) : 0.0;

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

    if (g_hull_eps > 0.0) {
        // Poistetaan karjet joiden etaisyys naapurien valiseen janaan < eps px (kupera kayra
        // ylitarkasti diskretoitu) - LM:n reunatestin hinta on suoraan verrannollinen karkien maaraan.
        std::vector<int> idx(hull_idx.begin(), hull_idx.end());
        bool removed = true;
        while (removed && idx.size() > 12) {
            removed = false;
            for (size_t k = 0; k < idx.size() && idx.size() > 12; ) {
                const cv::Point2f& a = proj_f[(size_t)idx[(k + idx.size() - 1) % idx.size()]];
                const cv::Point2f& b = proj_f[(size_t)idx[k]];
                const cv::Point2f& c = proj_f[(size_t)idx[(k + 1) % idx.size()]];
                double abx = c.x - a.x, aby = c.y - a.y;
                double L = std::sqrt(abx * abx + aby * aby);
                double dist = L < 1e-9 ? 0.0 : std::abs((b.x - a.x) * aby - (b.y - a.y) * abx) / L;
                if (dist < g_hull_eps) { idx.erase(idx.begin() + (long)k); removed = true; k += 1; }
                else ++k;
            }
        }
        hull_idx.assign(idx.begin(), idx.end());
    }

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

static std::vector<double> jointResidualsLinearized(const LinearizedResidualContext& ctx, const cv::Vec3d& params)
{
    double dX = params[0] - ctx.X0_ref;
    double dY = params[1] - ctx.Y0_ref;
    double dR = params[2] - ctx.R0_ref;

    std::vector<double> out;

    if (!ctx.body_pts_f->empty()) {
        if (ctx.hull_lin->valid) {
            auto hull = evalLinearizedHull(*ctx.hull_lin, dX, dY);
            std::vector<double> r(ctx.body_pts_f->size());
            for (size_t i = 0; i < ctx.body_pts_f->size(); ++i)
                r[i] = cv::pointPolygonTest(hull, (*ctx.body_pts_f)[i], true);
            out.insert(out.end(), r.begin(), r.end());
        } else {
            std::vector<double> r(ctx.body_pts_f->size(), 1000.0);
            out.insert(out.end(), r.begin(), r.end());
        }
    }

    if (!ctx.ring_pts_f->empty()) {
        auto ring = evalLinearizedRing(*ctx.ring_lin, dX, dY, dR);
        std::vector<double> r(ctx.ring_pts_f->size());
        for (size_t i = 0; i < ctx.ring_pts_f->size(); ++i)
            r[i] = cv::pointPolygonTest(ring, (*ctx.ring_pts_f)[i], true);
        out.insert(out.end(), r.begin(), r.end());
    }

    return out;
}


// ============================================================
// ANALYYTTINEN JACOBIAN (UUSI, EI Python-porttaus - kayttajan
// pyynnosta: "koska se on nyt lineaarinen"): evalLinearizedHull/Ring
// tuottaa karkipisteet AFFIININA parametrien funktiona (d(karkipiste)/
// d(param) = jac_*[k], VAKIO - ei riipu siita missa arvioidaan). Piste-
// aariviiva-etaisyys (lahimman reunan etaisyys) on derivoituva funktio
// reunan kahdesta paatepisteesta (poissa kulmista) - ketjusaannolla
// d(residuaali)/d(param) = d(etaisyys)/d(karkipiste) * jac_*[karki].
// Tama antaa TARKAN Jacobianin YHDESSA O(N) lahin-reuna-haussa - EI
// tarvitse finite-difference:in 2-3 YLIMAARAISTA koko residuaalijoukon
// uudelleenlaskentaa jokaisella ulommalla LM-iteraatiolla.
//
// polygonResidualWithGrad laskee residuaalin ARVON ja gradientin
// SAMASSA O(N)-lapikaynnissa (seka sisa-/ulkopuolella-testi etta lahin-
// reuna-haku - HUOM: taman on oltava YKSI lapikaynti eika kaksi, muuten
// nopeushyoty katoaa). Gradienttia ei kuitenkaan lasketa jokaiselle
// LM:n damping-kokeilulle (kalliimpaa kuin pelkka arvo) - vain kerran
// per HYVAKSYTTY askel (jointResidualsLinearized/cv::pointPolygonTest
// riittaa halvaksi arvo-vain-tarkistukseksi kokeiluille).
// ============================================================

struct ResidualWithGrad {
    double value = 0.0;
    double dX = 0.0, dY = 0.0, dR = 0.0;
};

static ResidualWithGrad polygonResidualWithGrad(
    const std::vector<cv::Point2f>& pts,
    const float* jac_ux, const float* jac_uy, const float* jac_ur,
    const float* jac_vx, const float* jac_vy, const float* jac_vr,
    const cv::Point2f& p)
{
    ResidualWithGrad out;

    size_t n = pts.size();
    if (n < 3) {
        out.value = 1000.0;
        return out;
    }

    bool inside = false;
    double best_d2 = std::numeric_limits<double>::max();
    size_t best_i1 = 0, best_i2 = 0;
    double best_t = 0.0;
    double px = (double)p.x, py = (double)p.y;

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        double v1x = pts[i].x, v1y = pts[i].y;
        double v2x = pts[j].x, v2y = pts[j].y;

        if ((v1y > py) != (v2y > py)) {
            double x_cross = v1x + (py - v1y) / (v2y - v1y) * (v2x - v1x);
            if (px < x_cross)
                inside = !inside;
        }

        double dx = v2x - v1x, dy = v2y - v1y;
        double len2 = dx * dx + dy * dy;
        double t;
        if (len2 < 1e-12) {
            t = 0.0;
        } else {
            double wx = px - v1x, wy = py - v1y;
            t = (wx * dx + wy * dy) / len2;
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;
        }
        double qx = v1x + t * dx, qy = v1y + t * dy;
        double ex = px - qx, ey = py - qy;
        double d2 = ex * ex + ey * ey;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_i1 = i; best_i2 = j; best_t = t;
        }
    }

    double sign = inside ? 1.0 : -1.0;
    double D = std::sqrt(best_d2);
    out.value = sign * D;

    if (D < 1e-9)
        return out;

    size_t i1 = best_i1, i2 = best_i2;
    double t = best_t;
    double qx = pts[i1].x + t * (pts[i2].x - pts[i1].x);
    double qy = pts[i1].y + t * (pts[i2].y - pts[i1].y);
    double ux = (qx - px) / D;
    double uy = (qy - py) / D;

    double dQx_dX = (1.0 - t) * jac_ux[i1] + t * jac_ux[i2];
    double dQx_dY = (1.0 - t) * jac_uy[i1] + t * jac_uy[i2];
    double dQy_dX = (1.0 - t) * jac_vx[i1] + t * jac_vx[i2];
    double dQy_dY = (1.0 - t) * jac_vy[i1] + t * jac_vy[i2];

    out.dX = sign * (ux * dQx_dX + uy * dQy_dX);
    out.dY = sign * (ux * dQx_dY + uy * dQy_dY);

    if (jac_ur != nullptr) {
        double dQx_dR = (1.0 - t) * jac_ur[i1] + t * jac_ur[i2];
        double dQy_dR = (1.0 - t) * jac_vr[i1] + t * jac_vr[i2];
        out.dR = sign * (ux * dQx_dR + uy * dQy_dR);
    }

    return out;
}


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


static const double g_lm_rel_tol = getenv("LM_REL_TOL") ? atof(getenv("LM_REL_TOL")) : 1e-6;
static const double g_lm_step_tol = getenv("LM_STEP_TOL") ? atof(getenv("LM_STEP_TOL")) : 0.0;

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

    LinearizedRing notch_lin;
    if (g_lm_notch && !ctx.body_pts_f->empty() && ctx.handle_r_frac > 0.0)
        notch_lin = buildLinearizedRing(params0[0], params0[1], ctx.handle_r_frac * ctx.R_max_cm, ctx.ring_height_cm, *ctx.K, *ctx.R, *ctx.t, 28);
    LinearizedResidualContext lctx{ &hull_lin, &ring_lin, ctx.body_pts_f, ctx.ring_pts_f, params0[0], params0[1], params0[2],
                                    (g_lm_notch && !ctx.body_pts_f->empty() && ctx.handle_r_frac > 0.0) ? &notch_lin : nullptr };
    profAdd(P_C_NBODY, (long long)ctx.body_pts_f->size() * 1000000LL);
    profAdd(P_C_NRING, (long long)ctx.ring_pts_f->size() * 1000000LL);
    profAdd(P_C_NHULL, (long long)hull_lin.ref_pts.size() * 1000000LL);
    long long lm_iters = 0, lm_evals = 1;
    double last_step_max = 1e9;
    rel_tol = std::max(rel_tol, g_lm_rel_tol);

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

            // Kokeiluaskeleen hyvaksynta/hylkays tarvitsee VAIN
            // kustannuksen (arvon) - kaytetaan halpaa arvo-vain-
            // funktiota (cv::pointPolygonTest). Gradientti lasketaan
            // vain KERRAN hyvaksytylle askeleelle (alla) - katso
            // taman lohkon alun kommentti.
            cv::Vec3d trial = params + cv::Vec3d(delta[0], delta[1], delta[2]);
            lm_evals += 1;
            PT ept;
            auto rj_new = jointResidualsAndJacobianLinearized(lctx, trial);
            profAdd(P_LM_EVAL_ONLY, ept.lap());
            double trial_cost = sumsq(rj_new.residuals);

            if (trial_cost < cost) {
                rel_improvement = (cost - trial_cost) / std::max(cost, 1e-12);
                params = trial;
                last_step_max = std::max(std::abs(delta[0]), std::max(std::abs(delta[1]), std::abs(delta[2])));
                residuals = std::move(rj_new.residuals);
                J = std::move(rj_new.J);
                cost = trial_cost;
                lam = std::max(lam / 5.0, 1e-12);
                step_taken = true;
                break;
            }

            lam *= 5.0;
        }

        if (!step_taken || rel_improvement < rel_tol || last_step_max < g_lm_step_tol)
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

        // OPTIMOINTI (EI numeerista eroa): profile_residuals_fast/
        // profileResiduals (runko-pisteiden jaannos) ei KOSKAAN kayta
        // R-parametria (vain X0,Y0 - katso funktion oma allekirjoitus) -
        // kun rengaspisteita ei ole (ring_pts tyhja), jaannosvektori EI
        // MUUTU LAINKAAN kun R:aa hairitaan, joten Jacobian R-sarake on
        // AINA TASMALLEEN nolla (residuals - residuals = 0, ei vain
        // likimaarin nolla) - taysin sama seka Pythonin etta taman
        // tiedoston OMASSA aiemmassa, aina jokaisen sarakkeen erikseen
        // laskevassa versiossa. Ohitetaan siis turha jointResiduals-
        // kutsu (koko runko-pisteiden pointPolygonTest-silmukka) talle
        // sarakkeelle kun se on jo etukateen tiedossa nollaksi - EI
        // vaikuta lopputulokseen, vain sailyttaa turhan laskennan.
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


// Linearisoitu LM + TARKKA VARMISTUS (katso levenbergMarquardt3Line
// arized:in oma kommentti). Referenssipisteessa (dX=dY=dR=0) linea
// risointi on TASMALLEEN tarkka, joten lahtokustannus saadaan
// "ilmaiseksi" tarkalla residuaalifunktiolla. Jos linearisoidun LM:n
// loppupiste ei OIKEASTI (tarkalla funktiolla mitattuna) ole vahin
// taan yhta hyva kuin lahtopiste, approksimaatio on pettanyt -
// hylataan tulos ja ajetaan TARKKA LM alusta lahtien varakeinona.
// Tama takaa etta linearisoitu polku EI VOI koskaan tuottaa huonom
// paa TULOSTA (pienimman nelion kustannusta) kuin tarkka LM - pahim
// millaan vain menetetaan nopeushyoty yksittaisessa kutsussa.
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
    static const bool g_lm_debug = getenv("LM_DEBUG") != nullptr;
    if (g_lm_debug) {
        static std::atomic<int> dbg(0);
        if (dbg++ < 60)
            fprintf(stderr, "[LMDBG] %s start=(%.1f,%.1f,%.2f) lin=(%.1f,%.1f,%.2f) cost0=%.1f costLin=%.1f nb=%zu nr=%zu\n",
                exact_cost_lin <= exact_cost_start * (1.0 + 1e-9) ? "OK  " : "FAIL",
                params0[0], params0[1], params0[2], params_lin[0], params_lin[1], params_lin[2],
                exact_cost_start, exact_cost_lin, ctx.body_pts_f->size(), ctx.ring_pts_f->size());
    }
    if (exact_cost_lin <= exact_cost_start * (1.0 + 1e-9))
        return params_lin;

    PT fpt;
    auto res_exact = levenbergMarquardt3(ctx, params0, max_iterations, lambda_init, rel_tol);
    profAdd(P_LM_FALLBACK, fpt.lap());
    return res_exact;
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
    // true jos lahin runkokontuuri kandidaattipaikan lahella hylattiin
    // YKSINOMAAN liian suuren pinta-alan takia (katso findContourNear)
    // - kutsuja (trackStoneUpdateOne/searchNewStoneOne) kayttaa tata
    // pakottamaan KOKO havainnon hylkays (has_position=false), koska
    // liian iso kohde talla kandidaattipaikalla ei todennakoisesti ole
    // kivi (esim. pelaaja).
    bool oversized_reject = false;
};


static RefineResult refinePositionJoint(
    const cv::Mat& mask_crop_roi, const cv::Mat& sat_crop_roi, int off_x_roi, int off_y_roi,
    int frame_w, int frame_h,
    const std::vector<cv::Point3d>& local_pts_body,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
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
    PT rpt;
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
    profAdd(P_R_PAD, rpt.lap());

    double ring_height_cm = H_total_cm;

    std::vector<cv::Point3d> approx3d{ cv::Point3d(X0_approx, Y0_approx, H_total_cm / 2.0) };
    auto approx_proj = project3d(K, R, t, approx3d);
    cv::Point2d approx_px_crop(approx_proj[0].x - off_x, approx_proj[0].y - off_y);

    // POISTETTU (kayttajan pyynnosta, katso keskusteluhistoria - empiiri-
    // sesti A/B-testattu koko Testivideo-julkaisua vasten): kokoraja
    // esti aiemmin findContourNear:ia hyvaksymasta kontuuria joka oli
    // yli BODY_CONTOUR_MAX_AREA_MULTIPLIER=4.0x odotettua kiven kokoa -
    // tama esti KIVI+PELAAJA-yhdistyneen kontuurin hyvaksymisen (turval-
    // linen), mutta sivuvaikutuksena myos AIDON kiven havainnon jos se
    // sattui olemaan hetkellisesti kosketuksissa/lahella pelaajaa (esim.
    // heittajaa) HAKU/SEURANTA-hetkella - havaittu oikealla datalla:
    // yksi aito heitto sai tasta VAARAN sijainnin/ajoituksen (7s virhe
    // ylitysajassa) koska koko havainto hylattiin puoliksi kesken.
    //
    // KAYTTAJAN PAATOS (kokoraja pois): "olisi tarkeinta loytaa KAIKKI
    // heitot" - hyvaksytaan tietoisesti riski etta joskus TAYSIN VAARA,
    // pelaajan siluettiin lukkiutuva havainto voi paasta lapi (havaittu
    // A/B-testissa: yksi ~17s "haamuheitto" jonka RMS oli 13-42px, ei
    // koskaan 1-3px:n tasolla kuin aidot kivet) - PIENEMPI RISKI kuin
    // aitojen heittojen menettaminen, kayttajan prioriteetin mukaan.
    auto approx_hull = predictedHull(local_pts_body, X0_approx, Y0_approx, K, R, t);
    double max_contour_area = -1.0;

    std::vector<cv::Point> raw_contour;
    bool oversized_reject = false;
    bool has_contour = findContourNear(
        mask_crop, approx_px_crop, raw_contour,
        BODY_CONTOUR_MAX_SEARCH_DIST_PX, BODY_CONTOUR_MIN_AREA_PX, max_contour_area,
        &oversized_reject
    );

    std::vector<cv::Point2d> body_pts;
    if (has_contour) {
        auto body_pts_crop = filterIceBoundaryPoints(sat_crop, raw_contour);
        body_pts.resize(body_pts_crop.size());
        for (size_t i = 0; i < body_pts_crop.size(); ++i)
            body_pts[i] = cv::Point2d(body_pts_crop[i].x + off_x, body_pts_crop[i].y + off_y);
    }
    if (g_body_stride > 1 && body_pts.size() > 40) {
        std::vector<cv::Point2d> dec;
        for (size_t i = 0; i < body_pts.size(); i += (size_t)g_body_stride) dec.push_back(body_pts[i]);
        body_pts.swap(dec);
    }
    int n_body = (int)body_pts.size();
    profAdd(P_R_CONTOUR, rpt.lap());

    // body_pts EI MUUTU koko BOUNDARY_MAX_ITERATIONS-silmukan/LM-ajon
    // aikana (vain ring_pts vaihtuu per ulompi iteraatio) - kastetaan
    // Point2f:ksi TASAN kerran tassa, katso profileResiduals:in oma
    // kommentti.
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

        // ring_pts VAIHTUU joka ulommalla iteraatiolla, joten kastetaan
        // uudelleen tassa (mutta silti vain KERRAN per ulompi iteraatio,
        // ei per LM-residuaalikutsu - katso body_pts_f:in kommentti).
        std::vector<cv::Point2f> ring_pts_f = toPoint2fVec(ring_pts);

        if (n_body < BODY_MIN_VALID_POINTS && n_ring < BOUNDARY_MIN_VALID_POINTS) {
            RefineResult res;
            res.X_cm = X0_approx; res.Y_cm = Y0_approx; res.tarkka = false;
            res.n_body = n_body; res.n_ring = n_ring;
            res.has_rms = false; res.has_ring_radius = false;
            res.oversized_reject = oversized_reject;
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

        // clean_body/clean_ring voivat olla ERI pistejoukko kuin body_pts/
        // ring_pts (MAD-poikkeavien hylkays yllakin) - kastetaan omiksi
        // Point2f-joukoikseen TASAN kerran (ei jokaisella ctx2/ctx_final:
        // in kayttamalla residuaalikutsulla).
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
        res.oversized_reject = oversized_reject;
        return res;
    }

    res.X_cm = X_cur; res.Y_cm = Y_cur; res.tarkka = true;
    res.n_body = n_body; res.n_ring = (int)ring_pts.size();
    res.rms_px = rms_px; res.has_rms = has_rms;
    res.ring_radius_cm = R_cur; res.has_ring_radius = true;
    res.oversized_reject = oversized_reject;

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


// ============================================================
// MEAN-SHIFT-SEURANTA (Testi_02_03, kayttajan idea - katso
// keskusteluhistoria): korvaa SEURANNAN ristikkohaun (locateByGrid
// SearchTrackingFast) iteratiivisella painopistesiirrolla.
//
// PERIAATE: kiven ennustettu 3D-projektio (hulli) ja sen 1.1-kertainen
// laajennus (sama HULL_MARGIN_SCALE kuin hullOverlapScore:ssa) jaetaan
// hullin painopisteen kautta kulkevalla vaaka- ja pystyviivalla
// neljaan osaan. Maskin pikselit lasketaan painotettuina (sisahulli
// w_in=2, vain 1.1x-kehaan osuva 1) ja puolikkaiden erotus
//   s_y = sum(w*[maski]*sign(y-cy)) / sum(w)      (alhaalla - ylhaalla)
//   s_x = sum(w*[maski]*sign(x-cx)) / sum(w)      (oikealla - vasemmalla)
// kertoo mihin suuntaan malli pitaa siirtaa. Signaali EI ole lineaarinen
// siirtyman suhteen (kyllastyy kun malli ja maski eivat enaa peity), joten
// "paljonko siirretaan" -painokerroin on OPTIMOITU: s muunnetaan pikseli-
// siirtymaksi d = gain * r * f^-1(s), missa f(delta) on YKSIKKOYMPYRAN
// (r=1) vastefunktio - lasketaan numeerisesti kerran samoilla painoilla
// ja marginaalilla kuin itse mittaus (getResponseLUT). Ellipsille (kiven
// perspektiiviprojektio) tama on affiini-invariantti, kun r on hullin
// puolileveys/-korkeus kyseisessa akselissa. gain on jaljelle jaava
// empiirinen hienosaato (mallin ja maskin kokoero, reunojen sumeus).
// Pikselisiirtyma muunnetaan (X,Y)-senteiksi paikallisella Jakobiaanilla.
// Toistetaan kunnes siirtyma < tol_px, jonka jalkeen normaali
// refinePositionJoint (LM) tarkentaa sub-pikselitasolle - TASMALLEEN
// sama kuin ristikkohaun jalkeen.
// ============================================================

struct MeanShiftParams {
    int max_iter = 10;
    double gain = 1.0;
    double tol_px = 0.10;
    double inner_weight = 2.0;
    double ring_weight = 1.0;   // 1.1x-kehan paino (negatiivinen = ulkopuolisen maskin rangaistus)
    double margin_scale = HULL_MARGIN_SCALE;
    double min_mask_fraction = 0.02;
    // Pehmea etumerkki: g(u)=clamp(u/tau,-1,1), u = (px - keskipiste)/puoliakseli.
    // tau=0 -> kova sign() (alkuperainen). Pienilla (kaukaisilla) kivilla kova
    // sign hyppii pikseliruudukon mukaan, pehmea on jatkuva siirtyman suhteen.
    double tau = 0.0;
    // Lopuksi paikallinen viimeistely hullOverlapScore-pistemaaralla (sama kuin
    // ristikkohaussa): aloitusaskel polish_step_cm, puolitetaan kunnes < min.
    double polish_step_cm = 0.0;
    double polish_min_step_cm = 0.75;
    // Fysikaalinen rajoite: kivi ei liiku taaksepain (Y kasvaa = taaksepain). Salli enintaan
    // back_allow_cm taaksepain edellisesta paikasta (mittauskohina). Iso arvo = ei rajoitusta.
    double back_allow_cm = 1e9;
    double prior_bias = 0.0;
    // Yhdistelmahaku (locate_mode 5): valinta = pistemaara - rangaistukset
    double ens_back_tol_cm = 2.0;   // sallittu taaksepain-siirtyma ilman rangaistusta
    double ens_back_pen = 0.15;     // rangaistus / 10 cm taaksepain (yli toleranssin)
    double ens_pred_pen = 0.02;     // rangaistus / 10 cm poikkeamasta ennustetusta paikasta
};

struct ResponseLUT {
    double dd = 0.01;
    std::vector<double> f;
    int imax = 0;
};

static std::shared_ptr<const ResponseLUT> getResponseLUT(double w_in, double margin, double tau, double w_ring = 1.0)
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
            double w = p1[c] ? P.inner_weight : (p2[c] ? P.ring_weight : 0.0);
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
    auto lut_ptr = getResponseLUT(P.inner_weight, P.margin_scale, P.tau, P.ring_weight);
    const ResponseLUT& lut = *lut_ptr;

    double zc = 0.0;
    for (auto& p : local_pts_search)
        zc += p.z;
    zc /= std::max<size_t>(1, local_pts_search.size());

    MeanShiftResult res;
    res.xy = cv::Point2d(X0, Y0);

    // Hakulaatikko on aina (X0,Y0) keskella; aloituspiste voi olla ennustettu paikka.
    double X = std::isnan(start_X) ? X0 : std::max(X0 - x_half_range, std::min(X0 + x_half_range, start_X));
    double Y = std::isnan(start_Y) ? Y0 : std::max(Y0 - y_half_range, std::min(std::min(Y0 + y_half_range, Y0 + P.back_allow_cm), start_Y));
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

        if (wtot <= 0.0 || wmask < P.min_mask_fraction * wtot)
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
        double newY = std::max(Y0 - y_half_range, std::min(std::min(Y0 + y_half_range, Y0 + P.back_allow_cm), Y + dY));

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

    // Paikallinen viimeistely: 8-naapurusto-kukkulankiipeily hullOverlapScore:lla,
    // askel puolittuu kunnes polish_min_step_cm. Kasittelee mask-epasymmetrian
    // aiheuttaman tasapainopisteen vinouman (katso offline-mittaukset).
    if (P.polish_step_cm > 0.0 && res.score > -0.5) {
        double step = P.polish_step_cm;
        double bx = X, by = Y, bs = res.score;
        while (step >= P.polish_min_step_cm) {
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
                        double cyp = std::max(Y0 - y_half_range, std::min(std::min(Y0 + y_half_range, Y0 + P.back_allow_cm), by + dj * step));
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


struct StoneUpdateResult {
    double score = 0.0;
    bool has_position = false;
    RefineResult refined;
    // Diagnostiikka (Testi_02_03): vaiheiden ajat ja hakuvaiheen tulos.
    double prep_ms = 0.0, locate_ms = 0.0, refine_ms = 0.0;
    int iters = 0;
    bool converged = false;
    bool used_fallback = false;
    double loc_X = 0.0, loc_Y = 0.0;
};


// locate_mode: 0 = alkuperainen laajeneva ristikkohaku (ei muutoksia),
// 1 = mean-shift + ristikkohaku varana jos pistemaara < score_threshold,
// 2 = pelkka mean-shift, 3 = TYHJENTAVA ristikkohaku ilman early-stopia
// (vertailun "kulta-standardi": paras mahdollinen hullOverlapScore-optimi).
static StoneUpdateResult trackStoneUpdateOneEx(
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
    int locate_mode, const MeanShiftParams& ms_params,
    double pred_dX = 0.0, double pred_dY = 0.0, bool edge_fallback = false)
{
    auto tp0 = std::chrono::steady_clock::now();
    auto msSince = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    PT pt_call;   // koko kutsun kello
    PT pt;

    int frame_w = frame_mat.cols, frame_h = frame_mat.rows;

    // ROI-rajaus kayttaa YHTA (konservatiivista, isompaa) half_rangea -
    // se vain mitoittaa kuvaleikkeen (mask/saturaatio) koon, EI itse
    // hakuristikkoa (katso locateByGridSearchTrackingFast alla, joka SAA
    // eri arvot X:lle ja Y:lle) - liian pieni leike olisi virhe, liian
    // iso vain hieman hitaampi.
    double roi_half_range = std::max(track_half_range_x_cm, track_half_range_y_cm)
        + BOUNDARY_MAX_SHIFT_FROM_APPROX_CM;
    cv::Rect roi = trackRoiBounds(X0, Y0, roi_half_range, R_max_cm, H_total_cm, frame_w, frame_h, K, R, t);

    StoneUpdateResult out;

    if (roi.width <= 0 || roi.height <= 0) {
        profAdd(P_CALL_TOTAL, pt_call.lap());
        return out;
    }

#ifdef STONE_TRACKER_DEBUG_TIMING
    auto ts0 = std::chrono::steady_clock::now();
#endif
    cv::Mat crop = frame_mat(roi);
    pt.lap();

    // Kayttajan pyynnosta (katso git-historia): taustanvaimennus (sama
    // suppressStaticBackground jo HAKU:ssa/searchNewStoneOne:ssa) nyt
    // myos SEURANTAlle - ilman tata kivi menetti tarkkuutensa (joskus
    // koko sijaintinsa) aina kun se ylitti staattisen jaamerkinnan
    // (pesan renkaat, hogline, mainokset), koska nama nayttavat
    // maskille TASMALLEEN yhta kivenkaltaisilta kuin itse kivi (matala
    // saturaatio + paikallisesti tumma) - mitattu oikealla videolla:
    // RMS normaalisti ~2px, hogline-ylityksella jopa 130-146px. Sama
    // jarjestys kuin searchNewStoneOne:ssa: MASKI lasketaan vaimenne-
    // tusta crop:ista (jottei staattinen merkinta nay maskissa), mutta
    // SATURAATIO alkuperaisesta (vaimennus valkaisee suppressoidut
    // pikselit, mika loisi keinotekoisen terävän saturaatiorajan
    // kiven reunalle jos sita kaytettaisiin reunanhakuun).
    cv::Mat crop_filtered = crop;
    bool have_bg_reference = false;
    if (!background_reference.empty() && background_reference.size() == frame_mat.size()
        && background_reference.type() == frame_mat.type()) {
        cv::Mat ref_crop = background_reference(roi);
        crop_filtered = suppressStaticBackground(crop, ref_crop, diff_threshold);
        have_bg_reference = true;
    }

    profAdd(P_PREP_CROP_SUPPRESS, pt.lap());
    cv::Mat sat_crop = computeSat(crop);
    profAdd(P_PREP_SAT, pt.lap());
    cv::Mat mask_crop = createGraniteMask(crop_filtered);
    profAdd(P_PREP_GRANITE, pt.lap());

    // Kayttajan ehdottama korjaus (katso keskusteluhistoria ja
    // createForegroundFromWhitened:in oma kommentti ylla): yhdistetaan
    // rakeisuusmaski suoraan (jo laskettuun) taustanvaimennuksen
    // etualamaskiin (OR) ENNEN ristikkohakua ja runkokontuurin
    // (findContourNear) hakua - tama on SEURANNAN ainoa kaytto
    // (HAKU/searchNewStoneOne EI koske tata, pysyy muuttumattomana).
    // mask_crop (pelkka rakeisuus) sailyy omana muuttujanaan, koska
    // sita EI muuteta - vain se MITA alempana valitetaan grid-haulle/
    // refinePositionJoint:lle vaihtuu. have_bg_reference-tarkistus
    // (sama ehto kuin crop_filtered:in laskennassa ylla) estaa taman
    // tapahtumasta jos taustareferenssia ei ole lainkaan kaytettavissa
    // - silloin crop_filtered==crop (ei valkaisua ollenkaan), jolloin
    // "ei-valkoinen" kattaisi lahes koko kuvan eika olisi mielekas
    // etuala-signaali.
    cv::Mat mask_for_track = mask_crop;
    if (have_bg_reference) {
        cv::Mat fg_mask = createForegroundFromWhitened(crop_filtered);
        mask_for_track = mask_crop | fg_mask;
    }
    profAdd(P_PREP_FOREGROUND, pt.lap());
#ifdef STONE_TRACKER_DEBUG_TIMING
    auto ts1 = std::chrono::steady_clock::now();
#endif

    // Kayttajan pyynnosta (katso keskusteluhistoria): kivi ei voi
    // liikkua nopeammin kuin TRACK_MAX_SPEED_Y/X_CM_S (main.py) sallii,
    // joten hakuristikkoa EI TARVITSE koskaan laajentaa sita kauemmas -
    // main.py laskee track_half_range_x/y_cm:n suoraan nopeusrajasta
    // (huomioiden mahdolliset peräkkäiset missit) jokaiselle kivelle
    // erikseen JOKA framella, eika tassa kaytita enaa kiinteaa
    // kamera9_02.py:n TRACK_HALF_RANGE_CM:ia.
    out.prep_ms = msSince(tp0);
    auto tl0 = std::chrono::steady_clock::now();
    pt.lap();

    std::pair<cv::Point2d, double> best;
    bool ms_needed = (locate_mode == 1 || locate_mode == 2);

    if (locate_mode == 0) {
        best = locateByGridSearchTrackingFast(
            local_pts_search, mask_for_track, roi.x, roi.y,
            X0, track_half_range_x_cm, Y0, track_half_range_y_cm,
            coarse_step_cm, fine_step_cm, K, R, t
        );
        profAdd(P_LOC_GRID, pt.lap());
    } else if (locate_mode == 4) {
        // Ristikkohaku ENSIN (ennustetusta keskipisteesta jos liike-ennuste annettu);
        // mean-shift vain jos ristikkohaku ei loyda riittavan hyvaa pistemaaraa.
        best = locateByGridSearchTrackingFast(
            local_pts_search, mask_for_track, roi.x, roi.y,
            X0 + pred_dX, track_half_range_x_cm, Y0 + pred_dY, track_half_range_y_cm,
            coarse_step_cm, fine_step_cm, K, R, t
        );
        ms_needed = best.second < score_threshold;
        out.used_fallback = ms_needed;   // tassa moodissa: MS-varahaku kaytossa
    } else if (locate_mode == 5) {
        // YHDISTELMAHAKU: ristikkohaku (ennustetusta keskipisteesta) + mean-shift (viimeisesta ja
        // ennustetusta paikasta). Kummallakin on oma virhetapansa (viereinen pelaaja vetaa MS:aa tai
        // ristikkohakua), joten valinta tehdaan pistemaarasta MINUS rangaistus fysikaalisesti
        // mahdottomasta liikkeesta (taaksepain = Y kasvaa) ja ennusteesta poikkeamisesta.
        struct Cand { cv::Point2d xy; double score; double adj; int src; };
        std::vector<Cand> cands;
        const bool have_pred = std::hypot(pred_dX, pred_dY) > 0.5;
        auto adjust = [&](const std::pair<cv::Point2d, double>& c, int src) {
            double back = std::max(0.0, (c.first.y - Y0) - ms_params.ens_back_tol_cm);
            double pdev = have_pred ? std::hypot(c.first.x - (X0 + pred_dX), c.first.y - (Y0 + pred_dY)) : 0.0;
            double adj = c.second - ms_params.ens_back_pen * back / 10.0 - ms_params.ens_pred_pen * pdev / 10.0;
            cands.push_back({ c.first, c.second, adj, src });
        };
        auto g = locateByGridSearchTrackingFast(
            local_pts_search, mask_for_track, roi.x, roi.y,
            X0 + pred_dX, track_half_range_x_cm, Y0 + pred_dY, track_half_range_y_cm,
            coarse_step_cm, fine_step_cm, K, R, t
        );
        profAdd(P_LOC_ENS_GRID, pt.lap());
        adjust(g, 0);
        auto m1 = meanShiftLocate(local_pts_search, mask_for_track, roi.x, roi.y,
            X0, Y0, track_half_range_x_cm, track_half_range_y_cm, K, R, t, ms_params);
        profAdd(P_LOC_MS1, pt.lap());
        adjust({ m1.xy, m1.score }, 1);
        out.iters = m1.iters; out.converged = m1.converged;
        if (have_pred) {
            auto m2 = meanShiftLocate(local_pts_search, mask_for_track, roi.x, roi.y,
                X0, Y0, track_half_range_x_cm, track_half_range_y_cm, K, R, t, ms_params,
                X0 + pred_dX, Y0 + pred_dY);
            profAdd(P_LOC_MS2, pt.lap());
            adjust({ m2.xy, m2.score }, 2);
            out.iters += m2.iters;
        }
        // hyvaksyttavat: pistemaara >= kynnys; jos ei yhtaan, paras raaka pistemaara (kuten ennenkin)
        const Cand* pick = nullptr;
        for (auto& c : cands)
            if (c.score >= score_threshold && (!pick || c.adj > pick->adj))
                pick = &c;
        if (!pick)
            for (auto& c : cands)
                if (!pick || c.score > pick->score)
                    pick = &c;
        best = { pick->xy, pick->score };
        out.used_fallback = (pick->src == 0);   // tassa moodissa: valittiin ristikkohaun tulos
    } else if (locate_mode == 3) {
        auto xv = arangeVec(X0 - track_half_range_x_cm, X0 + track_half_range_x_cm + 1e-6, coarse_step_cm);
        auto yv = arangeVec(Y0 - track_half_range_y_cm, Y0 + track_half_range_y_cm + 1e-6, coarse_step_cm);
        auto b1 = gridSearchBest(local_pts_search, mask_for_track, roi.x, roi.y, K, R, t, xv, yv);
        auto xv2 = arangeVec(b1.first.x - coarse_step_cm, b1.first.x + coarse_step_cm + 1e-6, fine_step_cm);
        auto yv2 = arangeVec(b1.first.y - coarse_step_cm, b1.first.y + coarse_step_cm + 1e-6, fine_step_cm);
        best = gridSearchBest(local_pts_search, mask_for_track, roi.x, roi.y, K, R, t, xv2, yv2);
    }

    if (ms_needed) {
        const auto grid_best = best;
        auto ms = meanShiftLocate(
            local_pts_search, mask_for_track, roi.x, roi.y,
            X0, Y0, track_half_range_x_cm, track_half_range_y_cm, K, R, t, ms_params
        );
        best = { ms.xy, ms.score };
        out.iters = ms.iters;
        out.converged = ms.converged;

        // Liike-ennuste: toinen mean-shift ennustetusta paikasta; valitaan parempi pistemaara
        // (ennuste saa pienen edun, koska liike on jatkuvaa ja maski voi olla hairitty).
        if (std::hypot(pred_dX, pred_dY) > 0.5) {
            auto ms2 = meanShiftLocate(
                local_pts_search, mask_for_track, roi.x, roi.y,
                X0, Y0, track_half_range_x_cm, track_half_range_y_cm, K, R, t, ms_params,
                X0 + pred_dX, Y0 + pred_dY
            );
            out.iters += ms2.iters;
            if (ms2.score + 0.02 >= best.second)
                best = { ms2.xy, ms2.score };

            // Prioriehdokas: pelkka viimeistely ennustetussa pisteessa (ei mean-shiftia).
            // Saa pistemaaraetua prior_bias - seuraa liikettä ellei mean-shift ole SELVASTI parempi
            // (esim. viereinen pelaaja vetaa maskin painopistetta ja pistemaara voi olla korkeampi vaarassa paikassa).
            if (ms_params.prior_bias > 0.0) {
                MeanShiftParams p0 = ms_params;
                p0.max_iter = 0;
                auto pr = meanShiftLocate(
                    local_pts_search, mask_for_track, roi.x, roi.y,
                    X0, Y0, track_half_range_x_cm, track_half_range_y_cm, K, R, t, p0,
                    X0 + pred_dX, Y0 + pred_dY
                );
                if (pr.score + ms_params.prior_bias >= best.second)
                    best = { pr.xy, pr.score };
            }
        }

        bool at_edge = false;
        if (edge_fallback) {
            at_edge = std::fabs(best.first.y - Y0) >= track_half_range_y_cm - 0.6 && track_half_range_y_cm >= 6.0;
        }

        if (locate_mode == 4 && grid_best.second > best.second)
            best = grid_best;

        if (locate_mode == 1 && (best.second < score_threshold || at_edge)) {
            auto gb = locateByGridSearchTrackingFast(
                local_pts_search, mask_for_track, roi.x, roi.y,
                X0, track_half_range_x_cm, Y0, track_half_range_y_cm,
                coarse_step_cm, fine_step_cm, K, R, t
            );
            out.used_fallback = true;
            if (gb.second > best.second)
                best = gb;
        }
    }

    out.locate_ms = msSince(tl0);
    profAdd(P_LOC_TOTAL, (long long)(out.locate_ms * 1e6));
    out.loc_X = best.first.x;
    out.loc_Y = best.first.y;

#ifdef STONE_TRACKER_DEBUG_TIMING
    auto ts2 = std::chrono::steady_clock::now();
    auto ms2 = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    fprintf(stderr, "[SEURANTA timing] roi=%dx%d mask+sat=%.2fms grid=%.2fms",
            roi.width, roi.height, ms2(ts0, ts1), ms2(ts1, ts2));
#endif

    out.score = best.second;

    if (out.score < score_threshold) {
#ifdef STONE_TRACKER_DEBUG_TIMING
        fprintf(stderr, " (no refine, score=%.3f)\n", out.score);
#endif
        profAdd(P_CALL_TOTAL, pt_call.lap());
        return out;
    }

    auto tr0 = std::chrono::steady_clock::now();
    out.refined = refinePositionJoint(
        mask_for_track, sat_crop, roi.x, roi.y, frame_w, frame_h, local_pts_body, K, R, t,
        R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, best.first.x, best.first.y
    );
    out.refine_ms = msSince(tr0);
    profAdd(P_REFINE_TOTAL, (long long)(out.refine_ms * 1e6));

    // Jos lahin runkokontuuri hylattiin YKSINOMAAN liian suuren pinta-
    // alan takia (kts. findContourNear/RefineResult::oversized_reject),
    // kandidaattipaikalla on todennakoisesti jokin muu kuin kivi (esim.
    // pelaaja) - hylataan koko havainto (has_position=false) sen sijaan
    // etta palautetaan hilahaun (vaara) sijainti "epatarkkana" tuloksena.
    out.has_position = !out.refined.oversized_reject;

    // Kayttajan pyynnosta (katso keskusteluhistoria): kivi EI VOI
    // koskaan liikkua "taaksepain" (kohti heittopaata, eli Y KASVAA -
    // katso taman tiedoston Y-suunnan kommentti main.py:ssa) enempaa
    // kuin max_backward_cm yhden SEURANTA-paivityksen aikana - aito
    // liikkuva kivi hidastuu mutta EI KOSKAAN peruuta, joten tallainen
    // tulos on lahes aina vaara kandidaatti (esim. ristikkohaku
    // ajautunut viereiseen kiveen/pelaajaan) eika oikea sijainti.
    if (out.has_position && (out.refined.Y_cm - Y0) > max_backward_cm) {
        out.has_position = false;
    }

#ifdef STONE_TRACKER_DEBUG_TIMING
    auto ts3 = std::chrono::steady_clock::now();
    fprintf(stderr, " refine=%.2fms n_body=%d n_ring=%d%s\n", ms2(ts2, ts3), out.refined.n_body, out.refined.n_ring,
            out.refined.oversized_reject ? " OVERSIZED_REJECT" : "");
#endif

    profAdd(P_CALL_TOTAL, pt_call.lap());
    return out;
}



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
    double max_backward_cm)
{
    return trackStoneUpdateOneEx(
        frame_mat, background_reference, diff_threshold, local_pts_body, local_pts_search, K, R, t,
        X0, Y0, track_half_range_x_cm, track_half_range_y_cm, coarse_step_cm, fine_step_cm,
        score_threshold, R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, max_backward_cm,
        0, MeanShiftParams()
    );
}


static py::dict resultToDict(const StoneUpdateResult& r)
{
    py::dict d;
    d["score"] = r.score;
    d["found"] = r.has_position;
    d["prep_ms"] = r.prep_ms;
    d["locate_ms"] = r.locate_ms;
    d["refine_ms"] = r.refine_ms;
    d["iters"] = r.iters;
    d["converged"] = r.converged;
    d["used_fallback"] = r.used_fallback;
    d["loc_X"] = r.loc_X;
    d["loc_Y"] = r.loc_Y;

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
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> background_reference,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_body_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_search_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr,
    double X0, double Y0,
    double track_half_range_x_cm, double track_half_range_y_cm,
    double coarse_step_cm, double fine_step_cm,
    double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    double max_backward_cm,
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
        result = trackStoneUpdateOne(
            frame_mat, ref_mat, diff_threshold, local_pts_body, local_pts_search, K, R, t,
            X0, Y0, track_half_range_x_cm, track_half_range_y_cm, coarse_step_cm, fine_step_cm,
            score_threshold, R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, max_backward_cm
        );
    }

    return resultToDict(result);
}


static py::list track_stones_batch(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_u,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> background_reference,
    py::array_t<double, py::array::c_style | py::array::forcecast> X0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> Y0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> half_range_x_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> half_range_y_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_body_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_search_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr,
    double coarse_step_cm, double fine_step_cm,
    double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    double max_backward_cm,
    double diff_threshold = 30.0,
    int locate_mode = 0, double ms_gain = 1.0, int ms_max_iter = 10, double ms_tol_px = 0.10,
    double ms_inner_weight = 2.0, double ms_margin_scale = HULL_MARGIN_SCALE,
    double ms_tau = 0.0, double ms_polish_step_cm = 0.0,
    py::object pred_dx_obj = py::none(), py::object pred_dy_obj = py::none(),
    bool ms_edge_fallback = false, double ms_back_allow_cm = 1e9, double ms_prior_bias = 0.0,
    double ms_ring_weight = 1.0, double ens_back_tol_cm = 2.0, double ens_back_pen = 0.15,
    double ens_pred_pen = 0.02)
{
    PT bpt_wall, bpt;
    std::vector<double> pdx, pdy;
    if (!pred_dx_obj.is_none()) {
        auto a = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(pred_dx_obj);
        auto u = a.unchecked<1>(); for (py::ssize_t i = 0; i < u.shape(0); ++i) pdx.push_back(u(i));
    }
    if (!pred_dy_obj.is_none()) {
        auto a = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(pred_dy_obj);
        auto u = a.unchecked<1>(); for (py::ssize_t i = 0; i < u.shape(0); ++i) pdy.push_back(u(i));
    }
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

    MeanShiftParams msp;
    msp.gain = ms_gain;
    msp.max_iter = ms_max_iter;
    msp.tol_px = ms_tol_px;
    msp.inner_weight = ms_inner_weight;
    msp.margin_scale = ms_margin_scale;
    msp.tau = ms_tau;
    msp.polish_step_cm = ms_polish_step_cm;
    msp.back_allow_cm = ms_back_allow_cm;
    msp.prior_bias = ms_prior_bias;
    msp.ring_weight = ms_ring_weight;
    msp.ens_back_tol_cm = ens_back_tol_cm;
    msp.ens_back_pen = ens_back_pen;
    msp.ens_pred_pen = ens_pred_pen;

    auto X0b = X0_arr.unchecked<1>();
    auto Y0b = Y0_arr.unchecked<1>();
    auto HXb = half_range_x_arr.unchecked<1>();
    auto HYb = half_range_y_arr.unchecked<1>();
    int n_stones = (int)X0b.shape(0);

    std::vector<StoneUpdateResult> results((size_t)n_stones);
    profAdd(P_BATCH_SETUP, bpt.lap());

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
        // ULOMMASTA per-kivi-rinnakkaistuksesta). ScopedSingleThreadedOpenCV
        // pakottaa jokaisen sisaisen OpenCV-kutsun sarjalliseksi TAMAN
        // kutsun ajaksi, jolloin AINOA rinnakkaistus on tama tiedoston
        // oma per-kivi std::thread-jako. VIITELASKURILLINEN (katso sen
        // oma kommentti) koska search_new_stone (HAKU) voi olla
        // kaynnissa SAMANAIKAISESTI eri saikeesta - naiivi per-kutsu
        // save/restore ei olisi turvallinen silloin.
        ScopedSingleThreadedOpenCV single_threaded_opencv_guard;

        std::atomic<int> next_idx(0);
        unsigned hw = std::thread::hardware_concurrency();
        int worker_count = std::max(1, std::min(n_stones, (int)(hw == 0 ? 4u : hw)));

        auto worker = [&]() {
            while (true) {
                int i = next_idx.fetch_add(1);
                if (i >= n_stones)
                    break;
                results[(size_t)i] = trackStoneUpdateOneEx(
                    frame_mat, ref_mat, diff_threshold, local_pts_body, local_pts_search, K, R, t,
                    X0b(i), Y0b(i), HXb(i), HYb(i), coarse_step_cm, fine_step_cm,
                    score_threshold, R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, max_backward_cm,
                    locate_mode, msp,
                    (size_t)i < pdx.size() ? pdx[(size_t)i] : 0.0, (size_t)i < pdy.size() ? pdy[(size_t)i] : 0.0,
                    ms_edge_fallback
                );
            }
        };

        std::vector<std::thread> workers;
        workers.reserve((size_t)worker_count);
        for (int i = 0; i < worker_count; ++i)
            workers.emplace_back(worker);
        for (auto& w : workers)
            w.join();

        // single_threaded_opencv_guard palauttaa OpenCV:n saiemaaran
        // tahan tuhoutuessaan (scope-lopun RAII, katso sen kommentti).
    }

    bpt.lap();
    py::list out;
    for (auto& r : results)
        out.append(resultToDict(r));
    profAdd(P_BATCH_RESULT, bpt.lap());
    profAdd(P_BATCH_WALL, bpt_wall.lap());

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

// Kayttajan raportoima ongelma (katso keskusteluhistoria): kivia "ei
// tunnisteta tai ne hukataan muutama metri hogin jalkeen" - juurisyy
// loydetty tasta funktiosta: locateByGridSearchFast palauttaa VAIN
// YHDEN, KOKO haku-vyohykkeen parhaiten pistetyn (hullOverlapScore)
// ehdokkaan. hullOverlapScore ei tee eroa kiven ja esim. pelaajan/
// lakaisijan tumman vaatetuksen valilla (katso alla oleva, jo
// olemassaollut kommentti) - PELAAJA ON TYYPILLISESTI LAHEMPANA
// KAMERAA JA ISOMPI kuin kaukainen, ohuena sivuprofiilina nakyva kivi,
// joten pelaaja VOITTAA ristikkohaun LAHES AINA kun molemmat ovat
// samaan aikaan hakuvyohykkeella (esim. lakaisija/skippi seisoo tai
// kavelee hog-linjan tuntumassa, mika on aivan tavallista curlingissa
// juuri talla vyohykkeella). Koska funktio aiemmin kokeili VAIN taman
// YHDEN parhaan ehdokkaan (ja hylkasi sen oikein refined.tarkka:n
// kautta koska pelaaja ei ole pyorea), koko HAKU-yritys palautti
// TYHJAA - kivi EI KOSKAAN paassyt edes yritykseen asti, vaikka se
// olisi ollut samaan aikaan nakyvissa samalla vyohykkeella.
//
// KORJAUS: jos parhaaksi loydetty ehdokas ei lapaise refined.tarkka-
// tarkistusta (ts. ei ole oikeasti pyorea kivi), sen sijaan etta
// luovutetaan valittomasti, POISTETAAN se hakumaskista (taytetaan
// nolliksi koko sen SAMA yhtenainen maskialue, cv::floodFill, ei vain
// kiven kokoinen laatikko - pelaaja on yleensa isompi kuin kivimalli,
// joten pelkka kiven kokoinen alue ei riittaisi estamaan samaa
// pelaajaa "voittamasta" uudelleen seuraavalla yrityksella) ja
// YRITETAAN UUDELLEEN jaljella olevalla vyohykkeella - jolloin
// SEURAAVAKSI paras ehdokas (mahdollisesti oikea, kauempana oleva
// kivi) paasee vuoroon. Rajattu MAX_HAKU_ATTEMPTS_PER_SCAN kertaan
// ettei yksi skannaus voi jaada ikuisesti silmukkaan (esim. useita
// pelaajia vyohykkeella).
static const int MAX_HAKU_ATTEMPTS_PER_SCAN = 5;

// HAKU:n hyvaksymiskynnys HIENOSAADETYLLE (refinePositionJoint:in
// jalkeiselle) peitto-osuudelle (kayttajan pyynnosta, katso keskustelu-
// historia): AIEMMIN uuden kiven hyvaksyminen vaati LISAKSI etta kiven
// pyorea reuna/rengasrakenne oli loydettavissa (refined.tarkka) - tama
// hylkasi kuitenkin myos AIDOT mutta kaukana/pienena nakyvat kivet,
// joiden rengasrakenne ei yksinkertaisesti erotu tarpeeksi harvoista
// pikseleista (havaittu: kaukovyohykkeen kivi, jonka 3D-malli sopi
// hyvin sen maskikontuuriin, mutta rengashaku ei koskaan konvergoinut
// sen pienuuden takia). Korvattu SAMALLA peitto-/ulkopuoli-pisteytyksella
// (hullOverlapScore, katso HULL_MARGIN_SCALE/HULL_OUTSIDE_PENALTY_WEIGHT)
// kuin SEURANNAssakin, mutta MATALAMMALLA kynnyksella kuin k92.py:n
// TRACK_SCORE_THRESHOLD (0.35) - uuden kiven ENSIHAVAINToa ei tarvitse
// vaatia yhta tiukaksi kuin jatkuvaa seurantaa, koska main.py:n oma
// "tarkka-osuus"-esivahvistusikkuna (MIN_PRECONFIRM_TARKKA_*) suodattaa
// pelaajat/lakaisijat pois myohemmin usean framen yli kerätyn datan
// perusteella - tama funktio ei siis ole ainoa suoja vaaria kandidaatteja
// vastaan.
static const double HAKU_ACCEPT_SCORE_THRESHOLD = 0.20;

static StoneUpdateResult searchNewStoneOne(
    const cv::Mat& frame_mat, const cv::Mat& background_reference, double diff_threshold,
    const std::vector<cv::Point3d>& local_pts_body,
    const std::vector<cv::Point3d>& local_pts_search,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double x_center, double x_half_width, double y_center, double y_half_range,
    double coarse_step_cm, double fine_step_cm, double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    std::vector<StoneUpdateResult>* multi_out = nullptr, int max_results = 1,
    int max_attempts = MAX_HAKU_ATTEMPTS_PER_SCAN)
{
    int frame_w = frame_mat.cols, frame_h = frame_mat.rows;

#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t0 = std::chrono::steady_clock::now();
#endif
    PT hpt_total, hpt;
    cv::Mat frame_filtered = suppressStaticBackground(frame_mat, background_reference, diff_threshold);
    profAdd(P_HAKU_SUPPRESS, hpt.lap());
#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t1 = std::chrono::steady_clock::now();
#endif
    cv::Mat mask_search = createGraniteMask(frame_filtered);
    profAdd(P_HAKU_MASK, hpt.lap());
#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t2 = std::chrono::steady_clock::now();
#endif
    cv::Mat sat_search = computeSat(frame_mat);
    profAdd(P_HAKU_SAT, hpt.lap());
#ifdef STONE_TRACKER_DEBUG_TIMING
    auto t3 = std::chrono::steady_clock::now();
#endif

    StoneUpdateResult out;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {

        auto best = locateByGridSearchFast(
            local_pts_search, mask_search, 0, 0,
            x_center, x_half_width, y_center, y_half_range,
            coarse_step_cm, fine_step_cm, K, R, t
        );
        profAdd(P_HAKU_LOCATE, hpt.lap());

#ifdef STONE_TRACKER_DEBUG_TIMING
        auto t4 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        if (attempt == 0)
            fprintf(stderr, "[HAKU timing] suppress=%.2fms mask=%.2fms sat=%.2fms grid=%.2fms",
                    ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4));
#endif

        out.score = best.second;

        if (out.score < score_threshold) {
#ifdef STONE_TRACKER_DEBUG_TIMING
            fprintf(stderr, " (no refine, score=%.3f, attempt=%d)\n", out.score, attempt);
#endif
            return out;
        }

        RefineResult refined = refinePositionJoint(
            mask_search, sat_search, 0, 0, frame_w, frame_h, local_pts_body, K, R, t,
            R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac, best.first.x, best.first.y
        );
        profAdd(P_HAKU_REFINE, hpt.lap());

        // Sama liian-ison-kontuurin hylkays kuin trackStoneUpdateOne:ssa -
        // katso sen kommentti. HYVAKSYNTA (kayttajan pyynnosta, katso
        // keskusteluhistoria - katso myos HAKU_ACCEPT_SCORE_THRESHOLD:in
        // oma kommentti ylempana MIKSI refined.tarkka-vaatimuksesta
        // luovuttiin): lasketaan HIENOSAADETYN (refinePositionJoint:in
        // jalkeisen) sijainnin oma peitto-osuus SAMALLA hullOverlapScore-
        // pisteytyksella kuin SEURANNAssakin (peitto sisalla MIINUS
        // ulkopuolelle vuotava tumma alue) - pyorea, aito kivi saa
        // korkean pisteen (tumma alue loppuu tasan mallin reunalle),
        // pitkanomainen/epasaannollinen kohde (esim. jalka) matalamman
        // (tumma alue jatkuu mallin reunan ohi). Kynnys on tarkoituksella
        // MATALAMPI kuin SEURANNAssa (HAKU_ACCEPT_SCORE_THRESHOLD vs.
        // k92.py:n TRACK_SCORE_THRESHOLD) - katso sen oma kommentti.
        auto refined_hull = predictedHull(local_pts_body, refined.X_cm, refined.Y_cm, K, R, t);
        double refined_score = hullOverlapScore(mask_search, refined_hull, 0, 0);
        bool accept = !refined.oversized_reject && refined_score >= HAKU_ACCEPT_SCORE_THRESHOLD;

#ifdef STONE_TRACKER_DEBUG_TIMING
        fprintf(stderr, " refine=%.2fms attempt=%d score=%.3f%s%s\n", ms(t4, std::chrono::steady_clock::now()),
                attempt, refined_score, refined.oversized_reject ? " OVERSIZED_REJECT" : "",
                accept ? " ACCEPTED" : " (liian matala peitto-osuus, poistetaan alue ja yritetaan uudelleen)");
#endif

        if (accept) {
            out.refined = refined;
            out.has_position = true;
            if (!multi_out)
                return out;
            // Monituloshaku: talletetaan ja jatketaan (poistetaan tama alue alla, etsitaan seuraava).
            multi_out->push_back(out);
            if ((int)multi_out->size() >= max_results)
                return out;
        }

        // Ei kelvannut - poistetaan TAMA yhtenainen maskialue (floodFill,
        // ei vain kiven kokoinen laatikko - katso ylla oleva kommentti)
        // jotta sama (esim. pelaaja) ei voi voittaa uudelleen, ja
        // yritetaan seuraavaa parasta jaljella olevalla vyohykkeella.
        std::vector<cv::Point3d> best_pos3d{
            cv::Point3d(best.first.x, best.first.y, H_total_cm / 2.0)
        };
        auto best_proj = project3d(K, R, t, best_pos3d);
        int px0 = (int)std::lround(best_proj[0].x);
        int py0 = (int)std::lround(best_proj[0].y);

        // Etsitaan LAHIN maskin ei-nolla-pikseli (px0,py0):n ymparilta -
        // hullOverlapScore:n voittanut malliprojektio ei valttamatta osu
        // TASAN oikean kohteen omalle pikselille (mallin keskipisteen
        // projektio vs. kohteen todellinen muoto/paikka maskissa voivat
        // poiketa hieman), mutta jokin ei-nolla pikseli on TAKUULLA
        // lahella koska juuri tama alue voitti peitto-osuuspisteytyksen.
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

        if (seed_px < 0) {
            // Ei loytynyt yhtaan ei-nolla pikselia edes lahialueelta -
            // ei voida poistaa mitaan yhtenaista aluetta, luovutetaan
            // ettei jaada ikuisesti samaan tulokseen.
            out.has_position = false;
            return out;
        }

        cv::Mat flood_mask = cv::Mat::zeros(mask_search.rows + 2, mask_search.cols + 2, CV_8UC1);
        cv::floodFill(
            mask_search, flood_mask, cv::Point(seed_px, seed_py), cv::Scalar(0),
            nullptr, cv::Scalar(0), cv::Scalar(0), 4 | cv::FLOODFILL_MASK_ONLY | (255 << 8)
        );
        // FLOODFILL_MASK_ONLY jattaa mask_search:in itsensa koskematto-
        // maksi (vain flood_mask taytetaan) - poistetaan loydetty alue
        // eksplisiittisesti mask_search:ista flood_mask:in perusteella.
        cv::Mat region = flood_mask(cv::Rect(1, 1, mask_search.cols, mask_search.rows));
        mask_search.setTo(cv::Scalar(0), region);
        profAdd(P_HAKU_FLOOD, hpt.lap());
    }

    out.has_position = false;
    profAdd(P_HAKU_TOTAL, hpt_total.lap());
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
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
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

        // main.py:n elava seuranta kutsuu tata NYT (kayttajan pyynnosta)
        // omalta taustasaikeeltaan SAMAAN AIKAAN kun track_stones_batch
        // (SEURANTA) ajaa OMAa per-kivi-saiejakoaan paasaikeessa - HAKU
        // ja SEURANTA ovat riippumattomia (molemmat vain lukevat jo
        // valmiin framen, eivat toistensa tuloksia). ScopedSingleThreaded
        // OpenCV (katso sen oma kommentti) kytkee OpenCV:n oman sisaisen
        // rinnakkaistuksen pois PAALTA taman YHDEN kutsun ajaksi (VIITE-
        // LASKURILLISESTI, koska track_stones_batch voi olla kaynnissa
        // SAMANAIKAISESTI) ettei se kilpaile SEURANTAn per-kivi-saikeiden
        // kanssa - EI globaalisti koko elavan seurannan ajaksi (se
        // aiemmin kokeiltu lahestymistapa mitattiin kayttajan koneella
        // HITAAMMAKSI, koska se esti OpenCV:ta kayttamasta montaa
        // ydinta MUISSA, tayden framen operaatioissa kuten warpAffine+
        // remap - katso git-historia).
        ScopedSingleThreadedOpenCV single_threaded_opencv_guard;

        result = searchNewStoneOne(
            frame_mat, ref_mat, diff_threshold, local_pts_body, local_pts_search, K, R, t,
            x_center, x_half_width, y_center, y_half_range,
            coarse_step_cm, fine_step_cm, score_threshold,
            R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac
        );
    }

    return resultToDict(result);
}


// Testi_02_03: kuten search_new_stone, mutta palauttaa KAIKKI kelvolliset ehdokkaat (enintaan max_results)
// - alkuperainen palauttaa vain ensimmaisen, jolloin pelaaja/lakaisija voi voittaa kilpailun ja kiven
// rekisterointi jaa nain valiin (havaittu: 5 heittoa 26:sta jai kokonaan rekisteroimatta).
static py::list search_new_stones(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_u,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> background_reference,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_body_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_search_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr,
    double x_center, double x_half_width, double y_center, double y_half_range,
    double coarse_step_cm, double fine_step_cm, double score_threshold,
    double R_max_cm, double H_total_cm, double ring_r_frac_guess, double handle_r_frac,
    double diff_threshold = 30.0, int max_results = 4, int max_attempts = 8)
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

    std::vector<StoneUpdateResult> results;
    {
        py::gil_scoped_release release;
        ScopedSingleThreadedOpenCV single_threaded_opencv_guard;
        searchNewStoneOne(
            frame_mat, ref_mat, diff_threshold, local_pts_body, local_pts_search, K, R, t,
            x_center, x_half_width, y_center, y_half_range,
            coarse_step_cm, fine_step_cm, score_threshold,
            R_max_cm, H_total_cm, ring_r_frac_guess, handle_r_frac,
            &results, max_results, max_attempts
        );
    }
    py::list out;
    for (auto& r : results)
        out.append(resultToDict(r));
    return out;
}


// ============================================================
// KIVIKANDIDAATTIEN SKANNAUS (kamera9_01.py:n find_stone_candidates +
// ray_plane_intersection + kamera9_04.py:n _candidates_in_frame_fast:in
// C++-porttaus, kayttajan pyynnosta) - nopeuttaa main.py:n track_
// stone_in_video_windowed:ia (3D-kiviprofiilin skannausvaihe, katso
// sen oma kommentti) - EI KOSKETA elavan seurannan (HAKU/SEURANTA)
// hot pathia, tama on VAIN kertaluontoisen kalibroinnin/profiilin
// skannausvaihetta varten. Kayttaa jo olemassaolevia createGraniteMask/
// suppressStaticBackground-funktioita (validoitu HAKU:n kautta) -
// UUTTA tassa on vain kontuuri->ellipsi->suodatus->fyysinen sijainti
// -ketju, joka kayttaa SUORAAN samoja OpenCV-alkeisfunktioita
// (cv::findContours/fitEllipse/contourArea) kuin Python cv2-versiokin,
// joten tuloksen pitaisi olla TASMALLEEN sama - validoitu A/B-
// vertailulla Pythonin cv2-versioon (katso git-historia).
// ============================================================

static std::pair<double, double> rayPlaneIntersectionZ0(
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double u, double v)
{
    cv::Matx33d K_inv = K.inv();
    cv::Vec3d uv1(u, v, 1.0);
    cv::Vec3d ray_cam = K_inv * uv1;

    cv::Matx33d R_t = R.t();
    cv::Vec3d C = -(R_t * t);
    cv::Vec3d d = R_t * ray_cam;

    double s = (0.0 - C[2]) / d[2];
    double X = C[0] + s * d[0];
    double Y = C[1] + s * d[1];

    return { X, Y };
}


struct StoneCandidateFast {
    cv::RotatedRect ellipse;
    std::vector<cv::Point> contour;
    double area = 0.0;
    double X_cm = 0.0, Y_cm = 0.0;
};

static std::vector<StoneCandidateFast> findStoneCandidatesFast(
    const cv::Mat& mask, const cv::Matx33d& H_final,
    const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double min_area, double max_area, double min_fill_ratio, double min_aspect_ratio,
    double x_min, double x_max, double y_min, double y_max,
    double pixels_per_cm, double output_x_min_cm, double output_y_max_cm)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<StoneCandidateFast> candidates;

    for (auto& contour : contours) {

        double area = cv::contourArea(contour);

        if (area < min_area || area > max_area || contour.size() < 5)
            continue;

        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        double cx = ellipse.center.x, cy = ellipse.center.y;
        double w = ellipse.size.width, h = ellipse.size.height;

        double ellipse_area = M_PI * (w / 2.0) * (h / 2.0);
        if (ellipse_area < 1e-6)
            continue;

        double fill_ratio = area / ellipse_area;
        if (fill_ratio < min_fill_ratio)
            continue;

        double aspect_ratio = std::min(w, h) / std::max(w, h);
        if (aspect_ratio < min_aspect_ratio)
            continue;

        cv::Vec3d hp = H_final * cv::Vec3d(cx, cy, 1.0);
        double out_px_x = hp[0] / hp[2];
        double out_px_y = hp[1] / hp[2];

        double phys_x = out_px_x / pixels_per_cm + output_x_min_cm;
        double phys_y = output_y_max_cm - out_px_y / pixels_per_cm;

        if (!(phys_x >= x_min && phys_x <= x_max && phys_y >= y_min && phys_y <= y_max))
            continue;

        StoneCandidateFast cand;
        cand.ellipse = ellipse;
        cand.contour = contour;
        cand.area = area;

        auto pos = rayPlaneIntersectionZ0(K, R, t, cx, cy);
        cand.X_cm = pos.first;
        cand.Y_cm = pos.second;

        candidates.push_back(cand);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const StoneCandidateFast& a, const StoneCandidateFast& b) { return a.area > b.area; });

    return candidates;
}


static std::vector<StoneCandidateFast> scanStoneCandidates(
    const cv::Mat& frame_bgr, const cv::Mat& background_reference, double diff_threshold,
    const cv::Matx33d& H_final, const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
    double min_area, double max_area, double min_fill_ratio, double min_aspect_ratio,
    double x_min, double x_max, double y_min, double y_max,
    double pixels_per_cm, double output_x_min_cm, double output_y_max_cm)
{
    cv::Mat frame_filtered = suppressStaticBackground(frame_bgr, background_reference, diff_threshold);
    cv::Mat mask = createGraniteMask(frame_filtered);
    return findStoneCandidatesFast(
        mask, H_final, K, R, t, min_area, max_area, min_fill_ratio, min_aspect_ratio,
        x_min, x_max, y_min, y_max, pixels_per_cm, output_x_min_cm, output_y_max_cm
    );
}


static py::list stoneCandidatesToList(const std::vector<StoneCandidateFast>& cands)
{
    py::list out;

    for (auto& c : cands) {

        py::dict d;

        py::tuple center = py::make_tuple(c.ellipse.center.x, c.ellipse.center.y);
        py::tuple size = py::make_tuple(c.ellipse.size.width, c.ellipse.size.height);
        d["ellipse"] = py::make_tuple(center, size, c.ellipse.angle);

        std::vector<py::ssize_t> shape{ (py::ssize_t)c.contour.size(), 1, 2 };
        py::array_t<int32_t> contour_arr(shape);
        auto buf = contour_arr.mutable_unchecked<3>();
        for (py::ssize_t i = 0; i < (py::ssize_t)c.contour.size(); ++i) {
            buf(i, 0, 0) = c.contour[(size_t)i].x;
            buf(i, 0, 1) = c.contour[(size_t)i].y;
        }
        d["contour"] = contour_arr;

        d["pos_cm"] = py::make_tuple(c.X_cm, c.Y_cm);

        out.append(d);
    }

    return out;
}


static py::list scan_stone_candidates(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_bgr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> background_reference,
    py::array_t<double, py::array::c_style | py::array::forcecast> H_final_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr,
    double min_area, double max_area, double min_fill_ratio, double min_aspect_ratio,
    double x_min, double x_max, double y_min, double y_max,
    double pixels_per_cm, double output_x_min_cm, double output_y_max_cm,
    double diff_threshold = 30.0)
{
    auto buf = frame_bgr.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("frame_bgr must be HxWx3 uint8 BGR");

    cv::Mat frame_mat((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, (void*)buf.ptr);

    cv::Mat ref_mat;
    auto ref_buf = background_reference.request();
    if (ref_buf.ndim == 3 && ref_buf.shape[2] == 3)
        ref_mat = cv::Mat((int)ref_buf.shape[0], (int)ref_buf.shape[1], CV_8UC3, (void*)ref_buf.ptr);

    auto H_final = parseMat33(H_final_arr);
    auto K = parseMat33(K_arr);
    auto R = parseMat33(R_arr);
    auto t = parseVec3(t_arr);

    std::vector<StoneCandidateFast> result;
    {
        py::gil_scoped_release release;
        result = scanStoneCandidates(
            frame_mat, ref_mat, diff_threshold, H_final, K, R, t,
            min_area, max_area, min_fill_ratio, min_aspect_ratio,
            x_min, x_max, y_min, y_max, pixels_per_cm, output_x_min_cm, output_y_max_cm
        );
    }

    return stoneCandidatesToList(result);
}


// Diagnostiikka (Testi_02_03): palauttaa seurannan maskin (samoin kuin
// trackStoneUpdateOneEx sen muodostaa) sekä mean-shift-signaalit annetuissa
// pisteissä - käytetään vain offline-viritykseen (tools/ms_offline.py).
static py::dict ms_debug(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_u,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> background_reference,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_pts_search_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr,
    double X0, double Y0, double roi_half_cm, double R_max_cm, double H_total_cm,
    double diff_threshold,
    py::array_t<double, py::array::c_style | py::array::forcecast> evalX_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> evalY_arr,
    double inner_weight, double margin_scale, double tau)
{
    auto buf = frame_u.request();
    cv::Mat frame_mat((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, (void*)buf.ptr);
    cv::Mat ref_mat;
    auto ref_buf = background_reference.request();
    if (ref_buf.ndim == 3 && ref_buf.shape[2] == 3)
        ref_mat = cv::Mat((int)ref_buf.shape[0], (int)ref_buf.shape[1], CV_8UC3, (void*)ref_buf.ptr);

    auto local_pts_search = parsePts3(local_pts_search_arr);
    auto K = parseMat33(K_arr);
    auto R = parseMat33(R_arr);
    auto t = parseVec3(t_arr);

    cv::Rect roi = trackRoiBounds(X0, Y0, roi_half_cm + BOUNDARY_MAX_SHIFT_FROM_APPROX_CM,
                                  R_max_cm, H_total_cm, frame_mat.cols, frame_mat.rows, K, R, t);
    py::dict out;
    if (roi.width <= 0 || roi.height <= 0)
        return out;

    cv::Mat crop = frame_mat(roi);
    cv::Mat crop_filtered = crop;
    bool have_bg = false;
    if (!ref_mat.empty() && ref_mat.size() == frame_mat.size() && ref_mat.type() == frame_mat.type()) {
        crop_filtered = suppressStaticBackground(crop, ref_mat(roi), diff_threshold);
        have_bg = true;
    }
    cv::Mat mask_granite = createGraniteMask(crop_filtered);
    cv::Mat mask_for_track = mask_granite;
    if (have_bg)
        mask_for_track = mask_granite | createForegroundFromWhitened(crop_filtered);

    MeanShiftParams P;
    P.inner_weight = inner_weight;
    P.margin_scale = margin_scale;
    P.tau = tau;

    auto ex = evalX_arr.unchecked<1>();
    auto ey = evalY_arr.unchecked<1>();
    py::list sigs;
    for (py::ssize_t i = 0; i < ex.shape(0); ++i) {
        auto hull = predictedHull(local_pts_search, ex(i), ey(i), K, R, t);
        py::dict d;
        if (hull.size() >= 3) {
            MsSignals sg = computeMsSignals(mask_for_track, roi.x, roi.y, hull, P);
            d["valid"] = sg.valid;
            d["sx"] = sg.sx; d["sy"] = sg.sy; d["wtot"] = sg.wtot; d["wmask"] = sg.wmask;
            d["rx"] = sg.rx; d["ry"] = sg.ry; d["cx"] = sg.cx + roi.x; d["cy"] = sg.cy + roi.y;
            d["score"] = hullOverlapScore(mask_for_track, hull, roi.x, roi.y);
        } else {
            d["valid"] = false;
        }
        sigs.append(d);
    }

    py::array_t<uint8_t> mask_np({mask_for_track.rows, mask_for_track.cols});
    for (int r = 0; r < mask_for_track.rows; ++r)
        std::memcpy(mask_np.mutable_data(r, 0), mask_for_track.ptr<uchar>(r), (size_t)mask_for_track.cols);

    out["roi"] = py::make_tuple(roi.x, roi.y, roi.width, roi.height);
    out["mask"] = mask_np;
    out["signals"] = sigs;
    return out;
}



// ============================================================
// VARJONSIETOINEN TAUSTANVAIMENNUS + VALOTASAPAINON SOVELLUS YHDELLA LAPIKAYNNILLA
// (Testi_03_01, nopeusoptimointi). Korvaa Pythonin apply_photometric_correction +
// suppress_static_background(_shadow_tolerant_background_mask): ~30-50 ms/ruutu (numpy, useita
// koko frame -valiaikaistaulukoita) -> yksi rivikohtainen lapikaynti (cv::parallel_for_).
// Tulos vastaa tasmalleen Pythonia: OpenCV:n BGR2GRAY-kokonaislukukaava, HSV:n V=max ja
// S=(diff*sdiv+2048)>>12, sekä float->uint8 katkaisu (clip + trunc).
// ============================================================
static py::array_t<uint8_t> suppress_shadow_background(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> reference,
    py::array_t<double, py::array::c_style | py::array::forcecast> gains,
    py::array_t<double, py::array::c_style | py::array::forcecast> biases,
    double diff_threshold, double v_drop_min, double v_drop_max, int ice_s_max, int ice_v_min)
{
    auto fb = frame.request();
    auto rb = reference.request();
    if (fb.ndim != 3 || fb.shape[2] != 3 || rb.ndim != 3 || rb.shape[2] != 3 || fb.shape[0] != rb.shape[0] || fb.shape[1] != rb.shape[1])
        throw std::runtime_error("suppress_shadow_background: frame/reference must be HxWx3 uint8 of equal size");

    const int H = (int)fb.shape[0], W = (int)fb.shape[1];
    double g[3] = {1, 1, 1}, bi[3] = {0, 0, 0};
    bool photo = gains.size() >= 3 && biases.size() >= 3;
    if (photo) { auto gu = gains.unchecked<1>(); auto bu = biases.unchecked<1>(); for (int c = 0; c < 3; ++c) { g[c] = gu(c); bi[c] = bu(c); } }

    py::array_t<uint8_t> out({ (py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)3 });
    uint8_t* dst = (uint8_t*)out.request().ptr;
    const uint8_t* src = (const uint8_t*)fb.ptr;
    const uint8_t* ref = (const uint8_t*)rb.ptr;

    static int sdiv[256];
    static bool sdiv_init = false;
    if (!sdiv_init) {
        sdiv[0] = 0;
        for (int i = 1; i < 256; ++i) sdiv[i] = (int)std::lround((255 << 12) / (1.0 * i));
        sdiv_init = true;
    }
    // valotasapaino LUT per kanava: uint8 -> uint8 (Pythonin float32-clip-astype(uint8) -kaava)
    uint8_t lut[3][256];
    for (int c = 0; c < 3; ++c)
        for (int v = 0; v < 256; ++v) {
            if (!photo) { lut[c][v] = (uint8_t)v; continue; }
            float x = (float)((double)v * g[c] + bi[c]);
            if (x < 0.f) x = 0.f; if (x > 255.f) x = 255.f;
            lut[c][v] = (uint8_t)x;
        }

    {
        py::gil_scoped_release release;
        cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& rr) {
            for (int y = rr.start; y < rr.end; ++y) {
                const uint8_t* sp = src + (size_t)y * W * 3;
                const uint8_t* rp = ref + (size_t)y * W * 3;
                uint8_t* dp = dst + (size_t)y * W * 3;
                for (int x = 0; x < W; ++x, sp += 3, rp += 3, dp += 3) {
                    const int b = lut[0][sp[0]], gg = lut[1][sp[1]], r = lut[2][sp[2]];
                    const int db = std::abs(b - rp[0]), dg = std::abs(gg - rp[1]), dr = std::abs(r - rp[2]);
                    const int gray = (db * 1868 + dg * 9617 + dr * 4899 + 8192) >> 14;
                    bool bg = (double)gray < diff_threshold;
                    if (!bg) {
                        const int vmax = std::max(b, std::max(gg, r));
                        const int vmin = std::min(b, std::min(gg, r));
                        const int rv = std::max((int)rp[0], std::max((int)rp[1], (int)rp[2]));
                        const double vdrop = (double)(rv - vmax);
                        if (vdrop > v_drop_min && vdrop < v_drop_max) bg = true;
                        else {
                            const int sat = ((vmax - vmin) * sdiv[vmax] + 2048) >> 12;
                            if (sat < ice_s_max && vmax > ice_v_min) bg = true;
                        }
                    }
                    if (bg) { dp[0] = dp[1] = dp[2] = 255; }
                    else { dp[0] = (uint8_t)b; dp[1] = (uint8_t)gg; dp[2] = (uint8_t)r; }
                }
            }
        });
    }
    return out;
}

static void prof_reset() { for (int i = 0; i < P_COUNT; ++i) { g_prof_ns[i] = 0; g_prof_n[i] = 0; } }
static py::list prof_snapshot() {
    py::list L;
    for (int i = 0; i < P_COUNT; ++i)
        L.append(py::make_tuple(std::string(PROF_NAMES[i]), (double)g_prof_ns[i].load() / 1e6, (long long)g_prof_n[i].load()));
    return L;
}

PYBIND11_MODULE(stone_tracker, m)
{
    m.def("suppress_shadow_background", &suppress_shadow_background,
          py::arg("frame"), py::arg("reference"), py::arg("gains"), py::arg("biases"),
          py::arg("diff_threshold"), py::arg("v_drop_min"), py::arg("v_drop_max"), py::arg("ice_s_max"), py::arg("ice_v_min"));
    m.def("prof_reset", &prof_reset);
    m.def("prof_snapshot", &prof_snapshot);
    m.doc() = "C++-porttaus SEURANTA- ja HAKU-vaiheiden kuumasta polusta (Task 5+6)";

    m.def("track_stone_update", &track_stone_update,
          "Yhden kiven ristikkohaku+yhteissovitus (SEURANTA-paivitys)",
          py::arg("frame_u"), py::arg("background_reference"),
          py::arg("local_pts_body"), py::arg("local_pts_search"),
          py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("X0"), py::arg("Y0"),
          py::arg("track_half_range_x_cm"), py::arg("track_half_range_y_cm"),
          py::arg("coarse_step_cm"), py::arg("fine_step_cm"),
          py::arg("score_threshold"),
          py::arg("R_max_cm"), py::arg("H_total_cm"), py::arg("ring_r_frac_guess"), py::arg("handle_r_frac"),
          py::arg("max_backward_cm"),
          py::arg("diff_threshold") = 30.0);

    m.def("track_stones_batch", &track_stones_batch,
          "Usean kiven ristikkohaku+yhteissovitus rinnakkain std::thread:eilla",
          py::arg("frame_u"), py::arg("background_reference"),
          py::arg("X0_arr"), py::arg("Y0_arr"),
          py::arg("half_range_x_arr"), py::arg("half_range_y_arr"),
          py::arg("local_pts_body"), py::arg("local_pts_search"),
          py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("coarse_step_cm"), py::arg("fine_step_cm"),
          py::arg("score_threshold"),
          py::arg("R_max_cm"), py::arg("H_total_cm"), py::arg("ring_r_frac_guess"), py::arg("handle_r_frac"),
          py::arg("max_backward_cm"),
          py::arg("diff_threshold") = 30.0,
          py::arg("locate_mode") = 0, py::arg("ms_gain") = 1.0, py::arg("ms_max_iter") = 10,
          py::arg("ms_tol_px") = 0.10, py::arg("ms_inner_weight") = 2.0,
          py::arg("ms_margin_scale") = HULL_MARGIN_SCALE,
          py::arg("ms_tau") = 0.0, py::arg("ms_polish_step_cm") = 0.0,
          py::arg("pred_dx") = py::none(), py::arg("pred_dy") = py::none(),
          py::arg("ms_edge_fallback") = false, py::arg("ms_back_allow_cm") = 1e9,
          py::arg("ms_prior_bias") = 0.0, py::arg("ms_ring_weight") = 1.0,
          py::arg("ens_back_tol_cm") = 2.0, py::arg("ens_back_pen") = 0.15, py::arg("ens_pred_pen") = 0.02);

    m.def("ms_debug", &ms_debug,
          "Diagnostiikka: seurantamaski + mean-shift-signaalit annetuissa pisteissa",
          py::arg("frame_u"), py::arg("background_reference"), py::arg("local_pts_search"),
          py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("X0"), py::arg("Y0"), py::arg("roi_half_cm"), py::arg("R_max_cm"), py::arg("H_total_cm"),
          py::arg("diff_threshold"), py::arg("evalX"), py::arg("evalY"),
          py::arg("inner_weight") = 2.0, py::arg("margin_scale") = HULL_MARGIN_SCALE, py::arg("tau") = 0.0);

    m.def("ms_response_table", [](double w_in, double margin, double tau) {
              auto lut = getResponseLUT(w_in, margin, tau);
              return py::make_tuple(lut->dd, lut->f, lut->imax);
          },
          py::arg("inner_weight") = 2.0, py::arg("margin_scale") = HULL_MARGIN_SCALE, py::arg("tau") = 0.0,
          "Mean-shift-vastefunktion f(delta) taulukkona (dd, f[], imax)");

    m.def("search_new_stones", &search_new_stones,
          "Kuten search_new_stone, mutta palauttaa kaikki kelvolliset ehdokkaat (lista)",
          py::arg("frame_u"), py::arg("background_reference"),
          py::arg("local_pts_body"), py::arg("local_pts_search"),
          py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("x_center"), py::arg("x_half_width"),
          py::arg("y_center"), py::arg("y_half_range"),
          py::arg("coarse_step_cm"), py::arg("fine_step_cm"), py::arg("score_threshold"),
          py::arg("R_max_cm"), py::arg("H_total_cm"), py::arg("ring_r_frac_guess"), py::arg("handle_r_frac"),
          py::arg("diff_threshold") = 30.0, py::arg("max_results") = 4, py::arg("max_attempts") = 8);

    m.def("search_new_stone", &search_new_stone,
          "Uuden kiven haku kiinteältä vyohykkeelta (HAKU), koko frame",
          py::arg("frame_u"), py::arg("background_reference"),
          py::arg("local_pts_body"), py::arg("local_pts_search"),
          py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("x_center"), py::arg("x_half_width"),
          py::arg("y_center"), py::arg("y_half_range"),
          py::arg("coarse_step_cm"), py::arg("fine_step_cm"),
          py::arg("score_threshold"),
          py::arg("R_max_cm"), py::arg("H_total_cm"), py::arg("ring_r_frac_guess"), py::arg("handle_r_frac"),
          py::arg("diff_threshold") = 30.0);

    m.def("scan_stone_candidates", &scan_stone_candidates,
          "Kivikandidaattien skannaus yhdesta framesta (3D-kiviprofiilin "
          "skannausvaihe - EI elavan seurannan hot path)",
          py::arg("frame_bgr"), py::arg("background_reference"),
          py::arg("H_final"), py::arg("K"), py::arg("R"), py::arg("t"),
          py::arg("min_area"), py::arg("max_area"),
          py::arg("min_fill_ratio"), py::arg("min_aspect_ratio"),
          py::arg("x_min"), py::arg("x_max"), py::arg("y_min"), py::arg("y_max"),
          py::arg("pixels_per_cm"), py::arg("output_x_min_cm"), py::arg("output_y_max_cm"),
          py::arg("diff_threshold") = 30.0);
}
