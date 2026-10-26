// ============================================================
// netti_fit.cpp - kiven akselin tarkka sovitus (B-taso) C++:lla (pybind11 + OpenCV).
//
// Korvaa Python-version netti_seuranta.Mittari.sovita_leike:n: yhden ruudun leikkeesta (a) etualamaskit ja graniitin peittavyys (kivi_netti.Kamera.maskit,
// kivi_netti.alfa_kartta, kivi_tarkka.havainto) ja (b) kiven akselin (X, Y) sovitus runkomalliin (kivi_tarkka.sovita) sekä "peitto" (kivi-todiste). Sama malli ja
// sama mittaus kuin Python-versiossa; erot:
//   * optimointi: Levenberg-Marquardt (kaksi parametria; Jacobin eteenpaindiff.-askel 0,2 cm, Jacobi paivitetaan vain tarvittaessa) scipy.optimize.least_squares:n
//     (TRF, ~45 mallin laskentaa / ruutu) tilalle: ~15 mallin laskentaa / ruutu
//   * mallin laskennassa samat OpenCV-kutsut (fillConvexPoly + INTER_AREA, GaussianBlur), joten malli ja ratkaisu ovat samat alipikselitarkkuudella
// Python-puoli: netti_seuranta.Mittari kayttaa tata automaattisesti, jos moduuli netti_fit on kaannetty (muuten Python-toteutus); ympariston muuttuja NETTI_EI_CPP=1 ohittaa.
//
// Kaannos: ks. CMakeLists.txt (kohde netti_fit) tai Linuxissa (yksi rivi):
//   g++ -O3 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) $(pkg-config --cflags opencv4) cpp/netti_fit.cpp -o netti_fit$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") $(pkg-config --libs opencv4)
// ============================================================
#define _USE_MATH_DEFINES
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

using ArrU8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr int SS = 4;                   // peiton ylinaytteistus (kivi_tarkka.SS)
constexpr int NTH = 48;                 // renkaan pisteita (kivi_tarkka.NTH)
constexpr double KUPERA_MARGINAALI = 0.25;   // kupera(): sisapuolisten pisteiden poisto vasta kun piste on tama monta px kahdeksankulmion reunasta sisapuolella

struct Win {
    int x0 = 0, y0 = 0, w = 0, h = 0;
};

// numpy-taulukko -> cv::Mat (ei kopiota; taulukko pidettava hengissa kutsun ajan). 2-D: yksi kanava, 3-D: kanavia = viimeinen ulottuvuus.
cv::Mat viittaus(const ArrU8& a, int tyyppi1, int tyyppi3)
{
    py::buffer_info b = a.request();
    if (b.ndim == 2)
        return cv::Mat((int)b.shape[0], (int)b.shape[1], tyyppi1, b.ptr);
    if (b.ndim == 3 && b.shape[2] == 3)
        return cv::Mat((int)b.shape[0], (int)b.shape[1], tyyppi3, b.ptr);
    throw std::runtime_error("netti_fit: odottamaton taulukon muoto");
}

double lue(const py::dict& d, const char* avain, double oletus)
{
    return d.contains(avain) ? d[avain].cast<double>() : oletus;
}

inline double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

}  // namespace

class BFit {
public:
    BFit(const ArrU8& mode, const ArrU8& lane, const ArrU8& jaa, const ArrU8& vari, const ArrD& K, const ArrD& R, const ArrD& t, const ArrD& r_prof, const ArrD& z_prof,
         double sigma_opt, double b_cm, double gamma, const py::dict& konst)
        : sigma_opt_(sigma_opt), b_cm_(b_cm), gamma_(gamma)
    {
        mode_ = viittaus(mode, CV_8UC1, CV_8UC3).clone();
        cv::GaussianBlur(mode_, mode_blur_, cv::Size(3, 3), 0);           // tausta sumennettuna ja Lab-avaruudessa kerran (maskit / havainto)
        cv::cvtColor(mode_, mode_lab_, cv::COLOR_BGR2Lab);
        lane_ = viittaus(lane, CV_8UC1, CV_8UC3).clone();
        jaa_ = viittaus(jaa, CV_8UC1, CV_8UC3).clone();
        vari_ = viittaus(vari, CV_8UC1, CV_8UC3).clone();
        if (K.size() != 9 || R.size() != 9 || t.size() != 3)
            throw std::runtime_error("netti_fit: K, R, t vaarankokoisia");
        std::copy(K.data(), K.data() + 9, K_);
        std::copy(R.data(), R.data() + 9, R_);
        std::copy(t.data(), t.data() + 3, t_);
        const int nz = (int)r_prof.size();
        if (nz < 2 || (int)z_prof.size() != nz)
            throw std::runtime_error("netti_fit: profiili puutteellinen");
        // rengaspisteiden siirtymat kameran kehyksessa (R * delta), jotta jokaisessa laskennassa vain siirretaan (X, Y) ja projisoidaan
        D_.resize((size_t)nz * NTH * 3);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < NTH; ++j) {
                const double th = 2.0 * M_PI * j / NTH;
                const double dx = r_prof.data()[k] * std::cos(th), dy = r_prof.data()[k] * std::sin(th), dz = z_prof.data()[k];
                double* d = &D_[((size_t)k * NTH + j) * 3];
                for (int i = 0; i < 3; ++i)
                    d[i] = R_[i * 3 + 0] * dx + R_[i * 3 + 1] * dy + R_[i * 3 + 2] * dz;
            }
        Dyla_.resize((size_t)NTH * 3);                     // ylapinnan rengas (viimeinen profiilin taso)
        for (int j = 0; j < NTH; ++j) {
            const double th = 2.0 * M_PI * j / NTH;
            const double dx = r_prof.data()[nz - 1] * std::cos(th), dy = r_prof.data()[nz - 1] * std::sin(th), dz = z_prof.data()[nz - 1];
            double* d = &Dyla_[(size_t)j * 3];
            for (int i = 0; i < 3; ++i)
                d[i] = R_[i * 3 + 0] * dx + R_[i * 3 + 1] * dy + R_[i * 3 + 2] * dz;
        }
        nz_ = nz;
        fg_kynnys_ = (int)lue(konst, "fg_kynnys", 28);
        g_vmin_ = (int)lue(konst, "g_vmin", 75);
        g_vmax_ = (int)lue(konst, "g_vmax", 175);
        g_smax_ = (int)lue(konst, "g_smax", 70);
        varjo_ = (float)lue(konst, "varjo", 0.3);
        chr_max_ = (float)lue(konst, "chr_max", 9.0);
        l_tumma_ = lue(konst, "l_tumma", 28.0);
        reika_max_ala_ = (int)lue(konst, "reika_max_ala", 80);
        dev_max_vari_ = (float)lue(konst, "dev_max_vari", 45.0);
        moni_kaista_ = (int)lue(konst, "moni_kaista", 6);
        moni_s0_ = lue(konst, "moni_s0", 0.25);
        moni_ep_kelv_ = lue(konst, "moni_ep_kelv", 0.45);
        moni_ep_peitto_ = lue(konst, "moni_ep_peitto", 0.75);
        moni_peitto_min_ = lue(konst, "moni_peitto_min", 0.35);
        moni_rms_max_ = lue(konst, "moni_rms_max", 0.30);
        moni_kelv_min_ = lue(konst, "moni_kelv_min", 0.08);
        esi_sigma_ = lue(konst, "esi_sigma", 5.0);
        esi_raja_ = lue(konst, "esi_raja", 3.0);
        esi_kappa_ = lue(konst, "esi_kappa", 0.01);
        if (konst.contains("graniitti")) {
            py::sequence g = konst["graniitti"].cast<py::sequence>();
            for (int i = 0; i < 3; ++i)
                gran_[i] = g[i].cast<float>();
        }
    }

    // Yhden ruudun B-mittaus: roi = BGR-leike oikaistusta kuvasta (vasen ylakulma (x0, y0) oikaistussa kuvassa), alkuarvo (X0, Y0) cm.
    // Palauttaa dict(X, Y, rms, kelvolliset, peitto, sR, H) (kuten Python: Mittari.sovita_leike).
    // laaja: liikkuva kivi; epavarma sovitus (kelvollisia < moni_ep_kelv_ tai peitto < moni_ep_peitto_) toistetaan rangaistun kaistakustannuksen minimoinnilla (kivi_tarkka.minimoi_rangaistu):
    // skannaus siemenen ymparilla + kuviohaku; esi_x, esi_y: kahvan (A) antama paikka kustannuksen esitietona (NaN: ei esitietoa). Tulos "moni" = 1, jos minimi valittiin.
    py::dict sovita(const ArrU8& roi_, int x0, int y0, double X0, double Y0, bool laaja = false, double esi_x = std::nan(""), double esi_y = std::nan("")) const
    {
        cv::Mat roi = viittaus(roi_, CV_8UC1, CV_8UC3);
        if (roi.type() != CV_8UC3)
            throw std::runtime_error("netti_fit: leikkeen pitaa olla BGR (h x w x 3)");
        double X = 0.0, Y = 0.0, rms = 0.0, kelv = 0.0, peitto = 0.0;
        int moni = 0;
        Aika aika;
        {
            py::gil_scoped_release vapauta;
            laske(roi, x0, y0, X0, Y0, X, Y, rms, kelv, peitto, aika, laaja, esi_x, esi_y, moni);
        }
        py::dict out;
        out["X"] = X;
        out["Y"] = Y;
        out["rms"] = rms;
        out["kelvolliset"] = kelv;
        out["peitto"] = peitto;
        out["sR"] = 1.0;
        out["moni"] = moni;
        out["t_maskit_ms"] = aika.maskit_ms;  // diagnostiikka: ajan jakautuminen
        out["t_sovitus_ms"] = aika.sovitus_ms;
        out["n_malli"] = aika.n_malli;
        out["n_kaista1"] = aika.n_kaista1;
        out["n_kaista2"] = aika.n_kaista2;
        return out;
    }

    // Testikoukut (tools/testaa_vakiot.py): nopeutettu laskenta antaa saman tuloksen kuin yleinen reitti.
    // Siluetin kupera peite (kaikki tasot ja peitteen kulmatasot) paikassa (X, Y): kupera() antaa samat indeksit kuin cv::convexHull.
    bool tarkista_kupera(double X, double Y) const
    {
        std::vector<cv::Point2f> pts;
        pisteet(X, Y, pts, nullptr);
        std::vector<int> a, b;
        cv::convexHull(pts, a, false, false);
        kupera(pts, b);
        if (a != b)
            return false;
        std::vector<char> mukana((size_t)nz_, 0);
        for (int i : a)
            mukana[(size_t)(i / NTH)] = 1;
        std::vector<int> tasot;
        for (int k = 0; k < nz_; ++k)
            if (mukana[(size_t)k])
                tasot.push_back(k);
        pisteet(X, Y, pts, &tasot);
        cv::convexHull(pts, a, false, false);
        kupera(pts, b);
        return a == b;
    }

    // Reunakaista paikassa (X, Y) (siluetin peite ikkunassa, jonka marginaali 14 px): (nopea reitti kaytossa, tulos sama kuin dilate - erode).
    py::tuple tarkista_kaista(double X, double Y, int kaista) const
    {
        std::vector<cv::Point2f> h;
        hull(X, Y, h);
        float minx = h[0].x, maxx = h[0].x, miny = h[0].y, maxy = h[0].y;
        for (const auto& p : h) {
            minx = std::min(minx, p.x);
            maxx = std::max(maxx, p.x);
            miny = std::min(miny, p.y);
            maxy = std::max(maxy, p.y);
        }
        const int marg = 14;
        Win w{(int)std::floor(minx) - marg, (int)std::floor(miny) - marg, 0, 0};
        w.w = (int)std::ceil(maxx) + marg - w.x0;
        w.h = (int)std::ceil(maxy) + marg - w.y0;
        cv::Mat c0f;
        peite(h, w, c0f);
        const cv::Mat c0 = c0f > 0.5;
        std::vector<int> nopea;
        const bool kaytossa = reunakaista_valeista(c0, kaista, nopea);
        const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kaista + 1, 2 * kaista + 1));
        cv::Mat d, e, band;
        cv::dilate(c0, d, k);
        cv::erode(c0, e, k);
        cv::subtract(d, e, band);
        std::vector<int> yleinen;
        for (int y = 0; y < band.rows; ++y)
            for (int x = 0; x < band.cols; ++x)
                if (band.at<uint8_t>(y, x) > 0)
                    yleinen.push_back(y * band.cols + x);
        return py::make_tuple(kaytossa, !kaytossa || nopea == yleinen, (int)yleinen.size());
    }

private:
    struct Aika {
        double maskit_ms = 0.0, sovitus_ms = 0.0;
        int n_malli = 0, n_kaista1 = 0, n_kaista2 = 0;
    };
    mutable int malli_laskuri_ = 0;
    mutable std::vector<cv::Point2f> pts_muisti_;               // kaikkien tasojen pisteet viimeisimmassa paikassa (kaikki_pisteet)
    mutable double pts_muisti_X_ = std::nan(""), pts_muisti_Y_ = std::nan("");
    mutable std::vector<int> elementit_[16];                    // ellipsielementin rivien puoliskot kaistan sateen mukaan (reunakaista_valeista)
    mutable cv::Mat wt_puskuri_;
    mutable cv::Mat mask_;                                  // ylinaytteistetun peittomaskin puskuri (ei ole saieturvallinen: yksi kutsu kerrallaan / olio)

    // ---- geometria ---------------------------------------------------------------------------------------------
    void projisoi(const double c[3], double& u, double& v) const
    {
        const double a0 = K_[0] * c[0] + K_[1] * c[1] + K_[2] * c[2];
        const double a1 = K_[3] * c[0] + K_[4] * c[1] + K_[5] * c[2];
        const double a2 = K_[6] * c[0] + K_[7] * c[1] + K_[8] * c[2];
        u = a0 / a2;
        v = a1 / a2;
    }

    void kehys0(double X, double Y, double c0[3]) const        // R * (X, Y, 0) + t
    {
        for (int i = 0; i < 3; ++i)
            c0[i] = R_[i * 3 + 0] * X + R_[i * 3 + 1] * Y + t_[i];
    }

    // Siluetin rengaspisteet kuvassa. tasot != null: vain nama profiilin tasot (pisteet = tasot x NTH): hullin kulmat tulevat muutamalta tasolta, jotka pysyvat samoina
    // pienilla (alle ~2 cm) siirroilla; kokonaisen peitteen laskenta (1488 pistetta) on mallin laskennan kallein vaihe.
    void pisteet(double X, double Y, std::vector<cv::Point2f>& pts, const std::vector<int>* tasot) const
    {
        double c0[3];
        kehys0(X, Y, c0);
        const int n_tasoa = tasot ? (int)tasot->size() : nz_;
        pts.resize((size_t)n_tasoa * NTH);
        for (int a = 0; a < n_tasoa; ++a) {
            const int k = tasot ? (*tasot)[a] : a;
            for (int j = 0; j < NTH; ++j) {
                const double* d = &D_[((size_t)k * NTH + j) * 3];
                const double c[3] = {c0[0] + d[0], c0[1] + d[1], c0[2] + d[2]};
                double u, v;
                projisoi(c, u, v);
                pts[(size_t)a * NTH + j] = cv::Point2f((float)u, (float)v);
            }
        }
    }

    // Kaikkien tasojen pisteet paikassa (X, Y): viimeisin muistissa, koska saman paikan mahtuu-tarkistus, kehys ja malli tarvitsevat samat pisteet.
    const std::vector<cv::Point2f>& kaikki_pisteet(double X, double Y) const
    {
        if (!(pts_muisti_X_ == X && pts_muisti_Y_ == Y) || pts_muisti_.empty()) {
            pisteet(X, Y, pts_muisti_, nullptr);
            pts_muisti_X_ = X;
            pts_muisti_Y_ = Y;
        }
        return pts_muisti_;
    }

    // Pisteiden kupera peite (indeksit pisteisiin, kuten cv::convexHull(pts, idx, false, false)). Ennen laskentaa poistetaan pisteet, jotka ovat selvasti (KUPERA_MARGINAALI px)
    // kahdeksan aariipisteen muodostaman monikulmion sisapuolella: ne eivat voi olla peitteen kulmia, joten tulos on sama kuin koko joukon peitteella, mutta jarjestettavia pisteita on
    // murto-osa (jarjestys on peitteen kallein vaihe).
    void kupera(const std::vector<cv::Point2f>& pts, std::vector<int>& idx_ulos) const
    {
        const int n = (int)pts.size();
        if (n < 64) {
            cv::convexHull(pts, idx_ulos, false, false);
            return;
        }
        int ext[8] = {0, 0, 0, 0, 0, 0, 0, 0};                       // min u, min u+v, min v, max u-v, max u, max u+v, max v, min u-v
        for (int i = 1; i < n; ++i) {
            const float u = pts[i].x, v = pts[i].y;
            if (u < pts[ext[0]].x) ext[0] = i;
            if (u + v < pts[ext[1]].x + pts[ext[1]].y) ext[1] = i;
            if (v < pts[ext[2]].y) ext[2] = i;
            if (u - v > pts[ext[3]].x - pts[ext[3]].y) ext[3] = i;
            if (u > pts[ext[4]].x) ext[4] = i;
            if (u + v > pts[ext[5]].x + pts[ext[5]].y) ext[5] = i;
            if (v > pts[ext[6]].y) ext[6] = i;
            if (u - v < pts[ext[7]].x - pts[ext[7]].y) ext[7] = i;
        }
        // kahdeksankulmion sivut (a -> b) ja sisapuolisuuden kynnys: ristitulo / sivun pituus >= marginaali (sisapuoli on ristitulon etumerkin suunnassa)
        double ax[8], ay[8], ex[8], ey[8], raja[8];
        int m = 0;
        double ala = 0.0;
        for (int i = 0; i < 8; ++i) {
            const cv::Point2f &a = pts[ext[i]], &b = pts[ext[(i + 1) % 8]];
            const double dx = (double)b.x - a.x, dy = (double)b.y - a.y, len = std::hypot(dx, dy);
            ala += (double)a.x * b.y - (double)b.x * a.y;
            if (len < 1e-9)
                continue;
            ax[m] = a.x; ay[m] = a.y; ex[m] = dx; ey[m] = dy; raja[m] = KUPERA_MARGINAALI * len;
            ++m;
        }
        if (m < 3 || std::fabs(ala) < 1e-6) {
            cv::convexHull(pts, idx_ulos, false, false);
            return;
        }
        const double merkki = ala > 0.0 ? 1.0 : -1.0;               // monikulmion kiertosuunta
        std::vector<cv::Point2f> pr;
        std::vector<int> alku;
        pr.reserve(256);
        alku.reserve(256);
        for (int i = 0; i < n; ++i) {
            bool sisalla = true;
            for (int e = 0; e < m && sisalla; ++e) {
                const double r = ex[e] * ((double)pts[i].y - ay[e]) - ey[e] * ((double)pts[i].x - ax[e]);
                sisalla = merkki * r > raja[e];
            }
            if (!sisalla) {
                pr.push_back(pts[i]);
                alku.push_back(i);
            }
        }
        std::vector<int> h;
        cv::convexHull(pr, h, false, false);
        idx_ulos.resize(h.size());
        for (size_t i = 0; i < h.size(); ++i)
            idx_ulos[i] = alku[(size_t)h[i]];
    }

    // Siluetin kupera peite (kulmat); kulmatasot: tasot, joilla peitteen kulmat ovat (+- 1 taso)
    void hull(double X, double Y, std::vector<cv::Point2f>& out, const std::vector<int>* tasot = nullptr, std::vector<int>* kulmatasot = nullptr) const
    {
        std::vector<cv::Point2f> oma;
        const std::vector<cv::Point2f>* pts = &oma;
        if (tasot == nullptr)
            pts = &kaikki_pisteet(X, Y);
        else
            pisteet(X, Y, oma, tasot);
        std::vector<int> idx;
        kupera(*pts, idx);
        if (kulmatasot != nullptr) {                         // mitka tasot antavat hullin kulmat (indeksit)
            std::vector<char> mukana((size_t)nz_, 0);
            for (int i : idx) {
                const int a = i / NTH;
                const int k = tasot ? (*tasot)[a] : a;
                for (int dk = -1; dk <= 1; ++dk)
                    if (k + dk >= 0 && k + dk < nz_)
                        mukana[(size_t)(k + dk)] = 1;
            }
            kulmatasot->clear();
            for (int k = 0; k < nz_; ++k)
                if (mukana[(size_t)k])
                    kulmatasot->push_back(k);
        }
        out.resize(idx.size());
        for (size_t i = 0; i < idx.size(); ++i)
            out[i] = (*pts)[(size_t)idx[i]];
    }

    void ylapinta(double X, double Y, std::vector<cv::Point2f>& out) const
    {
        double c0[3];
        kehys0(X, Y, c0);
        out.resize(NTH);
        for (int j = 0; j < NTH; ++j) {
            const double* d = &Dyla_[(size_t)j * 3];
            const double c[3] = {c0[0] + d[0], c0[1] + d[1], c0[2] + d[2]};
            double u, v;
            projisoi(c, u, v);
            out[j] = cv::Point2f((float)u, (float)v);
        }
    }

    double sivumittakaava(double X, double Y) const      // sivusuuntainen kuvamittakaava (px / cm) keskikorkeudella z = 7 cm
    {
        double a[3], b[3];
        double u0, v0, u1, v1;
        kehys0(X, Y, a);
        kehys0(X + 1.0, Y, b);
        for (int i = 0; i < 3; ++i) {
            a[i] += R_[i * 3 + 2] * 7.0;
            b[i] += R_[i * 3 + 2] * 7.0;
        }
        projisoi(a, u0, v0);
        projisoi(b, u1, v1);
        return std::hypot(u1 - u0, v1 - v0);
    }

    // ---- malli -------------------------------------------------------------------------------------------------
    // Monikulmion peitto ikkunassa (alipikseli, SS-kertainen ylinaytteistus): kivi_tarkka.peite. Tulos sama kuin cv::resize(INTER_AREA) uint8-maskille (4 x 4 -lohkon keskiarvo
    // pyoristettyna kokonaislukuun, / 255), mutta ilman erillista resize-kutsua (nopeampi).
    void peite(const std::vector<cv::Point2f>& poly, const Win& w, cv::Mat& out) const
    {
        static_assert(SS == 4, "alasnaytteistus on kirjoitettu SS = 4:lle");
        cv::Mat& mk = mask_;
        if (mk.rows != w.h * SS || mk.cols != w.w * SS)
            mk.create(w.h * SS, w.w * SS, CV_8UC1);
        mk.setTo(cv::Scalar(0));
        std::vector<cv::Point> q(poly.size());
        for (size_t i = 0; i < poly.size(); ++i) {
            const float qx = (poly[i].x - (float)w.x0 + 0.5f) * (float)SS - 0.5f;
            const float qy = (poly[i].y - (float)w.y0 + 0.5f) * (float)SS - 0.5f;
            q[i] = cv::Point(cvRound(qx * 16.0f), cvRound(qy * 16.0f));
        }
        cv::fillConvexPoly(mk, q, cv::Scalar(255), cv::LINE_8, 4);
        out.create(w.h, w.w, CV_32F);
        const float k255 = 1.0f / 255.0f;
        for (int y = 0; y < w.h; ++y) {
            const uint8_t* r0 = mk.ptr<uint8_t>(SS * y);
            const uint8_t* r1 = mk.ptr<uint8_t>(SS * y + 1);
            const uint8_t* r2 = mk.ptr<uint8_t>(SS * y + 2);
            const uint8_t* r3 = mk.ptr<uint8_t>(SS * y + 3);
            float* po = out.ptr<float>(y);
            for (int x = 0; x < w.w; ++x) {
                const int a = SS * x;
                const int summa = r0[a] + r0[a + 1] + r0[a + 2] + r0[a + 3] + r1[a] + r1[a + 1] + r1[a + 2] + r1[a + 3] + r2[a] + r2[a + 1] + r2[a + 2] + r2[a + 3] + r3[a] +
                                  r3[a + 1] + r3[a + 2] + r3[a + 3];
                po[x] = (float)((summa + 8) >> 4) * k255;
            }
        }
    }

    // Mallin graniittipeitto (sumennettu, gamma) ja, jos wt, ylapinnan pehmea "ei valia" -paino: kivi_tarkka.malli_kuva
    void malli(double X, double Y, const Win& w, cv::Mat& c, cv::Mat* wt, const std::vector<int>* tasot = nullptr) const
    {
        ++malli_laskuri_;
        std::vector<cv::Point2f> h;
        hull(X, Y, h, tasot);
        peite(h, w, c);
        const double sig = std::hypot(sigma_opt_, b_cm_ * sivumittakaava(X, Y));
        if (sig > 0.05)
            cv::GaussianBlur(c, c, cv::Size(0, 0), sig);
        cv::min(c, 1.0, c);
        cv::max(c, 0.0, c);
        cv::pow(c, gamma_, c);
        if (wt != nullptr) {
            std::vector<cv::Point2f> y;
            ylapinta(X, Y, y);
            cv::Mat t;
            peite(y, w, t);
            cv::GaussianBlur(t, wt_puskuri_, cv::Size(0, 0), 1.5);
            wt->create(w.h, w.w, CV_32F);
            for (int r = 0; r < w.h; ++r) {                       // wt = 1 - clip(2 t, 0, 1)
                const float* pt = wt_puskuri_.ptr<float>(r);
                float* pw = wt->ptr<float>(r);
                for (int x = 0; x < w.w; ++x)
                    pw[x] = 1.0f - std::min(std::max(2.0f * pt[x], 0.0f), 1.0f);
            }
        }
    }

    // ---- sovituskehys (kivi_tarkka.Kehys) ---------------------------------------------------------------------------
    struct Kehys {
        Win win;
        cv::Mat o, ok;               // alfa ja kelvolliset pikselit ikkunassa (float32, yhtenainen muisti)
        std::vector<int> idx;        // reunakaistan pikselit (rivi * leveys + sarake ikkunassa)
        std::vector<int> tasot;      // profiilin tasot, jotka antavat hullin kulmat alkupisteessa (+- 1 taso)
    };

    // Ellipsirakenneelementin (2 kaista + 1)^2 rivien puoliskot: hw[j + kaista] = rivin j (-kaista..kaista) vasemman / oikean reunan etaisyys keskelta; false, jos rivit eivat ole
    // yhtenaisia ja symmetrisia (silloin yleinen morfologia).
    bool elementin_puoliskot(int kaista, std::vector<int>& hw) const
    {
        const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kaista + 1, 2 * kaista + 1));
        hw.assign((size_t)(2 * kaista + 1), -1);
        for (int j = 0; j < k.rows; ++j) {
            const uint8_t* p = k.ptr<uint8_t>(j);
            int a = -1, b = -1, n = 0;
            for (int x = 0; x < k.cols; ++x)
                if (p[x]) {
                    if (a < 0) a = x;
                    b = x;
                    ++n;
                }
            if (a < 0 || n != b - a + 1 || a + b != 2 * kaista)
                return false;
            hw[(size_t)j] = kaista - a;
        }
        return true;
    }

    // Reunakaista dilate(c0) - erode(c0) (ellipsielementti): jos maskin jokainen rivi on yksi yhtenainen jakso eika maski kosketa ikkunan reunaa (kaistan sateen + 1 px sisalla), dilaatio ja
    // eroosio saadaan riveittaisista valeista (rivin y valiunioni / -leikkaus elementin rivien puoliskoilla) ilman kuvaoperaatioita; tulos on tasmalleen sama kuin yleisella reitilla.
    bool reunakaista_valeista(const cv::Mat& c0, int kaista, std::vector<int>& idx) const
    {
        if (kaista < 1 || kaista > 15)
            return false;
        auto& hw = elementit_[(size_t)kaista];
        if (hw.empty() && !elementin_puoliskot(kaista, hw))
            hw.assign(1, -2);                                    // ei kayttokelpoinen: merkitaan, ettei yritetty uudelleen
        if (hw.size() != (size_t)(2 * kaista + 1))
            return false;
        const int H = c0.rows, W = c0.cols, r = kaista;
        std::vector<int> L((size_t)H, W), R((size_t)H, -1);
        for (int y = 0; y < H; ++y) {
            const uint8_t* p = c0.ptr<uint8_t>(y);
            int a = -1, b = -1, n = 0;
            for (int x = 0; x < W; ++x)
                if (p[x]) {
                    if (a < 0) a = x;
                    b = x;
                    ++n;
                }
            if (a < 0)
                continue;
            if (n != b - a + 1)
                return false;                                    // rivissa aukko
            if (y < r + 1 || y >= H - r - 1 || a < r + 1 || b >= W - r - 1)
                return false;                                    // maski koskettaa ikkunan reunaa: reunaehdot (yleinen reitti)
            L[(size_t)y] = a;
            R[(size_t)y] = b;
        }
        idx.clear();
        for (int y = 0; y < H; ++y) {
            int dl = W, dr = -1, el = -(1 << 29), er = 1 << 29;
            bool e_tyhja = false;
            for (int j = -r; j <= r; ++j) {
                const int yy = y + j;
                if (yy < 0 || yy >= H)
                    continue;
                const int h = hw[(size_t)(j + r)];
                if (R[(size_t)yy] < 0) {
                    e_tyhja = true;                              // elementin rivilla ei maskia: eroosio tyhja
                    continue;
                }
                dl = std::min(dl, L[(size_t)yy] - h);
                dr = std::max(dr, R[(size_t)yy] + h);
                el = std::max(el, L[(size_t)yy] + h);
                er = std::min(er, R[(size_t)yy] - h);
            }
            if (dr < 0)
                continue;                                        // dilaatiossa ei mitaan
            dl = std::max(dl, 0);
            dr = std::min(dr, W - 1);
            const int base = y * W;
            if (e_tyhja || el > er) {
                for (int x = dl; x <= dr; ++x)
                    idx.push_back(base + x);
            } else {
                for (int x = dl; x < el; ++x)
                    idx.push_back(base + x);
                for (int x = er + 1; x <= dr; ++x)
                    idx.push_back(base + x);
            }
        }
        return true;
    }

    Kehys tee_kehys(const cv::Mat& alfa, const cv::Mat& ok8, int ox, int oy, double X0, double Y0, int kaista) const
    {
        const int marg = 14;
        std::vector<cv::Point2f> h;
        std::vector<int> kulmatasot;
        hull(X0, Y0, h, nullptr, &kulmatasot);
        float minx = h[0].x, maxx = h[0].x, miny = h[0].y, maxy = h[0].y;
        for (const auto& p : h) {
            minx = std::min(minx, p.x);
            maxx = std::max(maxx, p.x);
            miny = std::min(miny, p.y);
            maxy = std::max(maxy, p.y);
        }
        int wx0 = (int)std::floor(minx) - marg, wy0 = (int)std::floor(miny) - marg;
        int wx1 = (int)std::ceil(maxx) + marg, wy1 = (int)std::ceil(maxy) + marg;
        wx0 = std::max(ox, wx0);
        wy0 = std::max(oy, wy0);
        wx1 = std::min(ox + alfa.cols, wx1);
        wy1 = std::min(oy + alfa.rows, wy1);
        Kehys f;
        f.win = Win{wx0, wy0, wx1 - wx0, wy1 - wy0};
        if (f.win.w < 4 || f.win.h < 4)
            throw std::runtime_error("netti_fit: ikkuna tyhja");
        const cv::Rect rc(wx0 - ox, wy0 - oy, f.win.w, f.win.h);
        alfa(rc).copyTo(f.o);
        ok8(rc).convertTo(f.ok, CV_32F);
        cv::Mat c0f;
        peite(h, f.win, c0f);
        cv::Mat c0 = c0f > 0.5;                                  // 0 / 255
        if (!reunakaista_valeista(c0, kaista, f.idx)) {          // yleinen reitti: maski ei ole yhtenaisia rivivaleja tai koskettaa ikkunan reunaa
            f.idx.clear();
            cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kaista + 1, 2 * kaista + 1));
            cv::Mat d, e, band;
            cv::dilate(c0, d, k);
            cv::erode(c0, e, k);
            cv::subtract(d, e, band);
            for (int y = 0; y < band.rows; ++y) {
                const uint8_t* p = band.ptr<uint8_t>(y);
                for (int x = 0; x < band.cols; ++x)
                    if (p[x] > 0)
                        f.idx.push_back(y * band.cols + x);
            }
        }
        if (f.idx.empty())
            throw std::runtime_error("netti_fit: reunakaista tyhja");
        f.tasot = kulmatasot;
        return f;
    }

    // residuaalit ((malli - alfa) * kelvollinen * paino) reunakaistalla; kelv != null: kaistan kelvollinen osuus (ok * paino keskiarvo)
    void residuaalit(const Kehys& f, double X, double Y, std::vector<double>& r, double* kelv = nullptr, bool kaikki_tasot = false) const
    {
        cv::Mat c, wt;
        malli(X, Y, f.win, c, &wt, kaikki_tasot ? nullptr : &f.tasot);
        const float* pc = c.ptr<float>();
        const float* pw = wt.ptr<float>();
        const float* po = f.o.ptr<float>();
        const float* pk = f.ok.ptr<float>();
        r.resize(f.idx.size());
        double s = 0.0;
        for (size_t i = 0; i < f.idx.size(); ++i) {
            const int j = f.idx[i];
            r[i] = ((double)pc[j] - (double)po[j]) * (double)pk[j] * (double)pw[j];
            s += (double)pk[j] * (double)pw[j];
        }
        if (kelv != nullptr)
            *kelv = s / (double)f.idx.size();
    }

    // Levenberg-Marquardt (X, Y): rajat alkupisteen ymparilla +-8 / +-12 cm. Jacobi lasketaan alussa eteenpaindiff.-askelella (0,2 cm) ja paivitetaan sen jalkeen Broydenin
    // rank-1-paivityksella (ei lisaa mallin laskentoja); uusiksi differensseilla vain jos askel ei paranna. Paattyy kun askel < tol cm.
    void sovita_kaista(const Kehys& f, double& X, double& Y, double tol, int max_iter) const
    {
        const double bx0 = X - 8.0, bx1 = X + 8.0, by0 = Y - 12.0, by1 = Y + 12.0;
        const double h = 0.2;
        std::vector<double> r, r1, r2, rn, j0, j1;
        residuaalit(f, X, Y, r);
        double kustannus = 0.5 * dot(r, r);
        double lambda = 1e-3;
        bool uusi_jacobi = true;
        auto jacobi_differensseilla = [&]() {
            const double hx = (X + h <= bx1) ? h : -h, hy = (Y + h <= by1) ? h : -h;
            residuaalit(f, X + hx, Y, r1);
            residuaalit(f, X, Y + hy, r2);
            j0.resize(r.size());
            j1.resize(r.size());
            for (size_t i = 0; i < r.size(); ++i) {
                j0[i] = (r1[i] - r[i]) / hx;
                j1[i] = (r2[i] - r[i]) / hy;
            }
        };
        bool j_tuore = false;                                    // Jacobi on laskettu differensseilla eika sen jalkeen ole otettu askelta
        for (int it = 0; it < max_iter; ++it) {
            if (uusi_jacobi) {
                jacobi_differensseilla();
                uusi_jacobi = false;
                j_tuore = true;
            }
            const double A00 = dot(j0, j0), A01 = dot(j0, j1), A11 = dot(j1, j1), g0 = dot(j0, r), g1 = dot(j1, r);
            bool parani = false;
            double askel = 0.0;
            for (int yritys = 0; yritys < 6; ++yritys) {
                const double a00 = A00 * (1.0 + lambda) + 1e-12, a11 = A11 * (1.0 + lambda) + 1e-12, a01 = A01;
                const double det = a00 * a11 - a01 * a01;
                if (!(det > 0.0))
                    break;
                const double dx = -(a11 * g0 - a01 * g1) / det, dy = -(-a01 * g0 + a00 * g1) / det;
                const double Xn = std::min(std::max(X + dx, bx0), bx1), Yn = std::min(std::max(Y + dy, by0), by1);
                residuaalit(f, Xn, Yn, rn);
                const double kn = 0.5 * dot(rn, rn);
                if (kn < kustannus) {
                    const double sx = Xn - X, sy = Yn - Y, ss = sx * sx + sy * sy;
                    if (ss > 0.0) {                              // Broyden: J += ((dr - J s) s^T) / (s^T s)
                        for (size_t i = 0; i < r.size(); ++i) {
                            const double poik = (rn[i] - r[i]) - (j0[i] * sx + j1[i] * sy);
                            j0[i] += poik * sx / ss;
                            j1[i] += poik * sy / ss;
                        }
                    }
                    askel = std::sqrt(ss);
                    X = Xn;
                    Y = Yn;
                    kustannus = kn;
                    r.swap(rn);
                    lambda = std::max(lambda / 3.0, 1e-9);
                    parani = true;
                    j_tuore = false;
                    break;
                }
                lambda *= 5.0;
            }
            if (!parani) {
                if (j_tuore)
                    break;                                       // tuorekaan Jacobi ei tuota parannusta: minimi (kohinan tarkkuudella) saavutettu
                uusi_jacobi = true;                              // Broyden-Jacobi saattoi vanhentua: yksi yritys differensseilla
                lambda = 1e-3;
                continue;
            }
            if (askel < tol)
                break;
        }
    }

    // ---- mittaus: maskit, alfa, kelvolliset pikselit ----------------------------------------------------------------
    void havainto(const cv::Mat& roi, const cv::Mat& mode, const cv::Mat& mode_blur, const cv::Mat& lm, const cv::Mat& lane, const cv::Mat& jaa, const cv::Mat& vari, cv::Mat& alfa,
                  cv::Mat& ok) const
    {
        const int h = roi.rows, w = roi.cols;
        // --- kivi_netti.Kamera.maskit: graniittimaski ---
        cv::Mat hsv;
        cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
        cv::Mat bf, ad;
        cv::GaussianBlur(roi, bf, cv::Size(3, 3), 0);
        cv::absdiff(bf, mode_blur, ad);
        std::vector<cv::Mat> ach;
        cv::split(ad, ach);
        cv::Mat diff;
        cv::max(ach[0], ach[1], diff);
        cv::max(diff, ach[2], diff);
        std::vector<cv::Mat> hch;
        cv::split(hsv, hch);
        cv::Mat gran(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y) {
            const uint8_t* pd = diff.ptr<uint8_t>(y);
            const uint8_t* pl = lane.ptr<uint8_t>(y);
            const uint8_t* ps = hch[1].ptr<uint8_t>(y);
            const uint8_t* pv = hch[2].ptr<uint8_t>(y);
            uint8_t* pg = gran.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x)
                pg[x] = (pd[x] > fg_kynnys_ && pl[x] > 0 && pv[x] >= g_vmin_ && pv[x] <= g_vmax_ && ps[x] <= g_smax_) ? 255 : 0;
        }
        cv::morphologyEx(gran, gran, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8UC1));
        cv::morphologyEx(gran, gran, cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8UC1));
        // --- kivi_netti.alfa_kartta: graniitin peittavyys varien sekoitusmallista ---
        cv::Mat near;
        cv::dilate(gran, near, cv::Mat::ones(5, 5, CV_8UC1));
        alfa.create(h, w, CV_32F);
        for (int y = 0; y < h; ++y) {
            const uint8_t* pp = roi.ptr<uint8_t>(y);
            const uint8_t* pb = mode.ptr<uint8_t>(y);
            const uint8_t* pn = near.ptr<uint8_t>(y);
            const uint8_t* pv = hch[2].ptr<uint8_t>(y);
            float* pa = alfa.ptr<float>(y);
            for (int x = 0; x < w; ++x) {
                const float b0 = pb[3 * x], b1 = pb[3 * x + 1], b2 = pb[3 * x + 2];
                const float d0 = gran_[0] - b0, d1 = gran_[1] - b1, d2 = gran_[2] - b2;
                const float dd = d0 * d0 + d1 * d1 + d2 * d2;
                float a = ((pp[3 * x] - b0) * d0 + (pp[3 * x + 1] - b1) * d1 + (pp[3 * x + 2] - b2) * d2) / std::max(dd, 1.0f);
                if (!(dd > 2500.0f))
                    a = 0.0f;
                a = std::min(std::max(a, 0.0f), 1.0f);
                a = std::min(std::max((a - varjo_) / (1.0f - 2.0f * varjo_), 0.0f), 1.0f);
                pa[x] = (pn[x] > 0 && pv[x] > 60) ? a : 0.0f;
            }
        }
        // --- kivi_tarkka.havainto: kelvolliset pikselit ---
        cv::Mat lab;
        cv::cvtColor(roi, lab, cv::COLOR_BGR2Lab);
        cv::Mat dch(h, w, CV_32F), dev(h, w, CV_32F);
        cv::Mat tumma(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y) {
            const uint8_t* pl = lab.ptr<uint8_t>(y);
            const uint8_t* pm = lm.ptr<uint8_t>(y);
            const uint8_t* pp = roi.ptr<uint8_t>(y);
            const uint8_t* pb = mode.ptr<uint8_t>(y);
            float* pc = dch.ptr<float>(y);
            float* pe = dev.ptr<float>(y);
            uint8_t* pt = tumma.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) {
                const float da = (float)pl[3 * x + 1] - (float)pm[3 * x + 1], db = (float)pl[3 * x + 2] - (float)pm[3 * x + 2];
                pc[x] = std::sqrt(da * da + db * db);
                pt[x] = ((float)pl[3 * x] * (float)(100.0 / 255.0) < (float)l_tumma_) ? 1 : 0;
                const float b0 = pb[3 * x], b1 = pb[3 * x + 1], b2 = pb[3 * x + 2];
                const float d0 = gran_[0] - b0, d1 = gran_[1] - b1, d2 = gran_[2] - b2;
                const float p0 = pp[3 * x], p1 = pp[3 * x + 1], p2 = pp[3 * x + 2];
                float a = ((p0 - b0) * d0 + (p1 - b1) * d1 + (p2 - b2) * d2) / std::max(d0 * d0 + d1 * d1 + d2 * d2, 1.0f);
                a = std::min(std::max(a, 0.0f), 1.0f);
                const float r0 = p0 - (b0 + a * d0), r1 = p1 - (b1 + a * d1), r2 = p2 - (b2 + a * d2);
                pe[x] = std::sqrt(r0 * r0 + r1 * r1 + r2 * r2);
            }
        }
        cv::GaussianBlur(dch, dch, cv::Size(0, 0), 1.0);
        cv::GaussianBlur(dev, dev, cv::Size(0, 0), 1.0);
        cv::Mat muu_jaa(h, w, CV_8UC1), muu_vari(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y) {
            const float* pc = dch.ptr<float>(y);
            const float* pe = dev.ptr<float>(y);
            const uint8_t* pt = tumma.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) {
                muu_jaa.at<uint8_t>(y, x) = (pc[x] > chr_max_ || pt[x]) ? 1 : 0;
                muu_vari.at<uint8_t>(y, x) = (pe[x] > dev_max_vari_ || pt[x]) ? 1 : 0;
            }
        }
        cv::Mat k5 = cv::Mat::ones(5, 5, CV_8UC1);
        cv::dilate(muu_jaa, muu_jaa, k5);
        cv::dilate(muu_vari, muu_vari, k5);
        cv::Mat ice(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y) {
            const float* pa = alfa.ptr<float>(y);
            uint8_t* pi = ice.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x)
                pi[x] = pa[x] < 0.5f ? 1 : 0;
        }
        cv::Mat lab_c, st, cen;
        const int n = cv::connectedComponentsWithStats(ice, lab_c, st, cen, 4);
        std::vector<uint8_t> pieni((size_t)n, 0);
        for (int i = 1; i < n; ++i) {
            const int* s = st.ptr<int>(i);
            pieni[i] = (s[cv::CC_STAT_AREA] < reika_max_ala_ && s[cv::CC_STAT_LEFT] > 0 && s[cv::CC_STAT_TOP] > 0 && s[cv::CC_STAT_LEFT] + s[cv::CC_STAT_WIDTH] < w &&
                        s[cv::CC_STAT_TOP] + s[cv::CC_STAT_HEIGHT] < h) ? 1 : 0;
        }
        cv::Mat holes(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y) {
            const int* pl = lab_c.ptr<int>(y);
            uint8_t* ph = holes.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x)
                ph[x] = pieni[pl[x]];
        }
        cv::dilate(holes, holes, cv::Mat::ones(3, 3, CV_8UC1));
        ok.create(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y) {
            const uint8_t* pj = jaa.ptr<uint8_t>(y);
            const uint8_t* pv = vari.ptr<uint8_t>(y);
            const uint8_t* mj = muu_jaa.ptr<uint8_t>(y);
            const uint8_t* mv = muu_vari.ptr<uint8_t>(y);
            const uint8_t* ph = holes.ptr<uint8_t>(y);
            uint8_t* po = ok.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) {
                const bool eihole = ph[x] == 0;
                po[x] = ((pj[x] && !mj[x] && eihole) || (pv[x] && !mv[x] && eihole)) ? 1 : 0;
            }
        }
    }

    // graniitin alfa-keskiarvo mallisiluetin (mallikuva > 0,5) sisalla koko leikkeessa (kuten peitto laskennassa sovituksen jalkeen)
    double peitto_laske(const cv::Mat& alfa, int x0, int y0, int w, int h, double X, double Y) const
    {
        cv::Mat mc;
        malli(X, Y, Win{x0, y0, w, h}, mc, nullptr);
        double summa = 0.0;
        long long n = 0;
        for (int y = 0; y < h; ++y) {
            const float* pm = mc.ptr<float>(y);
            const float* pa = alfa.ptr<float>(y);
            for (int x = 0; x < w; ++x)
                if (pm[x] > 0.5f) {
                    summa += pa[x];
                    ++n;
                }
        }
        return n > 0 ? summa / (double)n : 0.0;
    }

    // rangaistu kaistakustannus paikassa (X, Y) (kivi_tarkka.kaistakustannus): kehys rakennetaan paikassa; kelvoton kaistapikseli maksaa s0^2
    double rangaistu(const cv::Mat& alfa, const cv::Mat& ok, int ox, int oy, double X, double Y, int kaista, double* kelv_ulos) const
    {
        Kehys f = tee_kehys(alfa, ok, ox, oy, X, Y, kaista);
        cv::Mat c, wt;
        malli(X, Y, f.win, c, &wt, &f.tasot);
        const float* pc = c.ptr<float>();
        const float* pw = wt.ptr<float>();
        const float* po = f.o.ptr<float>();
        const float* pk = f.ok.ptr<float>();
        double sr2 = 0.0, sw = 0.0, sinv = 0.0;
        for (size_t i = 0; i < f.idx.size(); ++i) {
            const int j = f.idx[i];
            const double w = (double)pk[j] * (double)pw[j];
            const double r = ((double)pc[j] - (double)po[j]) * w;
            sr2 += r * r;
            sw += w;
            sinv += 1.0 - std::min(std::max(w, 0.0), 1.0);
        }
        const double n = (double)f.idx.size();
        if (kelv_ulos != nullptr)
            *kelv_ulos = sw / n;
        return std::sqrt((sr2 + moni_s0_ * moni_s0_ * sinv) / n);
    }

    double esikustannus(double X, double Y, double esi_x, double esi_y) const
    {
        if (std::isnan(esi_x) || std::isnan(esi_y))
            return 0.0;
        const double t = std::hypot(X - esi_x, Y - esi_y) / esi_sigma_;
        return esi_kappa_ * std::min(t * t, esi_raja_ * esi_raja_);
    }

    // siluetti mahtuu leikkeeseen (6 px reunaa jaa): kaikkien rengaspisteiden rajauslaatikko (peitteen kulmat ovat ääripisteita, joten sama kuin peitteen kulmille)
    bool mahtuu(double X, double Y, int ox, int oy, int w, int h) const
    {
        const std::vector<cv::Point2f>& pts = kaikki_pisteet(X, Y);
        for (const auto& p : pts)
            if (p.x < ox + 6 || p.y < oy + 6 || p.x > ox + w - 6 || p.y > oy + h - 6)
                return false;
        return true;
    }

    // skannaus (dy x dx) + kuviohaku (askeleet) rangaistun kustannuksen (+ esikustannus) minimille; false, jos mikaan ehdokas ei mahdu leikkeeseen
    bool moni_haku(const cv::Mat& alfa, const cv::Mat& ok, int ox, int oy, int w, int h, double X0, double Y0, double esi_x, double esi_y, double& Xo, double& Yo, double& co) const
    {
        struct Muisti { double X, Y, c; bool ok; };
        std::vector<Muisti> muisti;                                 // jo arvioidut paikat (kuviohaku palaa usein edelliseen paikkaan): sama paikka antaa saman kustannuksen
        auto kust = [&](double X, double Y, double& c) -> bool {
            for (const auto& m : muisti)
                if (m.X == X && m.Y == Y) {
                    c = m.c;
                    return m.ok;
                }
            bool onnistui = false;
            try {
                if (mahtuu(X, Y, ox, oy, w, h)) {
                    c = rangaistu(alfa, ok, ox, oy, X, Y, moni_kaista_, nullptr) + esikustannus(X, Y, esi_x, esi_y);
                    onnistui = true;
                }
            } catch (const std::exception&) {
                onnistui = false;
            }
            muisti.push_back(Muisti{X, Y, onnistui ? c : 0.0, onnistui});
            return onnistui;
        };
        bool on = false;
        double bc = 0.0, bX = 0.0, bY = 0.0;
        for (double dy = -48.0; dy <= 48.0 + 1e-9; dy += 6.0)
            for (int ix = -1; ix <= 1; ++ix) {
                const double X = X0 + 12.0 * ix, Y = Y0 + dy;
                double c = 0.0;
                if (kust(X, Y, c) && (!on || c < bc)) {
                    on = true;
                    bc = c;
                    bX = X;
                    bY = Y;
                }
            }
        if (!on)
            return false;
        const double askeleet[3] = {3.0, 1.5, 0.75};
        for (double st : askeleet) {
            bool parani = true;
            while (parani) {
                parani = false;
                const double dd[4][2] = {{st, 0.0}, {-st, 0.0}, {0.0, st}, {0.0, -st}};
                for (const auto& d : dd) {
                    double c = 0.0;
                    if (kust(bX + d[0], bY + d[1], c) && c < bc - 1e-6) {
                        bc = c;
                        bX += d[0];
                        bY += d[1];
                        parani = true;
                    }
                }
            }
        }
        Xo = bX;
        Yo = bY;
        co = bc;
        return true;
    }

    void laske(const cv::Mat& roi, int x0, int y0, double X0, double Y0, double& X, double& Y, double& rms, double& kelv, double& peitto, Aika& aika, bool laaja, double esi_x, double esi_y,
               int& moni) const
    {
        using kello = std::chrono::steady_clock;
        const auto t_a = kello::now();
        malli_laskuri_ = 0;
        const int h = roi.rows, w = roi.cols;
        if (x0 < 0 || y0 < 0 || x0 + w > mode_.cols || y0 + h > mode_.rows || w < 8 || h < 8)
            throw std::runtime_error("netti_fit: leike kuvan ulkopuolella");
        const cv::Rect rc(x0, y0, w, h);
        cv::Mat alfa, ok;
        havainto(roi, mode_(rc), mode_blur_(rc), mode_lab_(rc), lane_(rc), jaa_(rc), vari_(rc), alfa, ok);
        const auto t_b = kello::now();
        // kivi_tarkka.sovita: kaksi kierrosta (kaista 8 -> 5 px); kehys jokaisen kierroksen alkupisteessa
        X = X0;
        Y = Y0;
        const int kaistat[2] = {8, 5};
        Kehys f;
        for (int i = 0; i < 2; ++i) {
            const int n_ennen = malli_laskuri_;
            f = tee_kehys(alfa, ok, x0, y0, X, Y, kaistat[i]);
            sovita_kaista(f, X, Y, i == 0 ? 0.15 : 0.03, i == 0 ? 8 : 12);      // 1. kierros: karkea (kaista 8 px), 2. kierros: tarkka (kaista 5 px)
            (i == 0 ? aika.n_kaista1 : aika.n_kaista2) = malli_laskuri_ - n_ennen;
        }
        std::vector<double> r;
        residuaalit(f, X, Y, r, &kelv, true);                    // lopullinen arvo kokonaisella hullilla (kaikki tasot)
        rms = std::sqrt(dot(r, r) / (double)r.size());
        // peitto: graniitin alfa-keskiarvo sovitetun rungon siluetin sisalla (mallikuva > 0,5), koko leikkeessa
        cv::Mat mc;
        malli(X, Y, Win{x0, y0, w, h}, mc, nullptr);
        double summa = 0.0;
        long long n = 0;
        for (int y = 0; y < h; ++y) {
            const float* pm = mc.ptr<float>(y);
            const float* pa = alfa.ptr<float>(y);
            for (int x = 0; x < w; ++x)
                if (pm[x] > 0.5f) {
                    summa += pa[x];
                    ++n;
                }
        }
        peitto = n > 0 ? summa / (double)n : 0.0;
        moni = 0;
        if (laaja && (kelv < moni_ep_kelv_ || peitto < moni_ep_peitto_)) {
            double Xm = 0.0, Ym = 0.0, cm = 0.0;
            if (moni_haku(alfa, ok, x0, y0, w, h, X0, Y0, esi_x, esi_y, Xm, Ym, cm)) {
                // minimin mittarit kuten sovituksen toisella kierroksella (kaista 5 px, koko hull) ja peitto
                Kehys f5 = tee_kehys(alfa, ok, x0, y0, Xm, Ym, 5);
                std::vector<double> r5;
                double kelv5 = 0.0;
                residuaalit(f5, Xm, Ym, r5, &kelv5, true);
                const double rms5 = std::sqrt(dot(r5, r5) / (double)r5.size());
                const double peitto5 = peitto_laske(alfa, x0, y0, w, h, Xm, Ym);
                if (peitto5 >= moni_peitto_min_ && rms5 <= moni_rms_max_ && kelv5 >= moni_kelv_min_) {
                    const double c0 = rangaistu(alfa, ok, x0, y0, X, Y, moni_kaista_, nullptr) + esikustannus(X, Y, esi_x, esi_y);
                    if (cm < c0) {
                        X = Xm;
                        Y = Ym;
                        rms = rms5;
                        kelv = kelv5;
                        peitto = peitto5;
                        moni = 1;
                    }
                }
            }
        }
        const auto t_c = kello::now();
        aika.maskit_ms = std::chrono::duration<double, std::milli>(t_b - t_a).count();
        aika.sovitus_ms = std::chrono::duration<double, std::milli>(t_c - t_b).count();
        aika.n_malli = malli_laskuri_;
    }

    cv::Mat mode_, mode_blur_, mode_lab_, lane_, jaa_, vari_;
    double K_[9], R_[9], t_[3];
    std::vector<double> D_, Dyla_;
    int nz_ = 0;
    double sigma_opt_, b_cm_, gamma_;
    int fg_kynnys_ = 28, g_vmin_ = 75, g_vmax_ = 175, g_smax_ = 70, reika_max_ala_ = 80;
    float varjo_ = 0.3f, chr_max_ = 9.0f, dev_max_vari_ = 45.0f;
    double l_tumma_ = 28.0;
    int moni_kaista_ = 6;
    double moni_s0_ = 0.25, moni_ep_kelv_ = 0.45, moni_ep_peitto_ = 0.75, moni_peitto_min_ = 0.35, moni_rms_max_ = 0.30, moni_kelv_min_ = 0.08;
    double esi_sigma_ = 5.0, esi_raja_ = 3.0, esi_kappa_ = 0.01;
    float gran_[3] = {130.0f, 126.0f, 136.0f};
};

PYBIND11_MODULE(netti_fit, m)
{
    m.doc() = "Kiven akselin tarkka sovitus (B-taso) C++:lla: BFit(mode, lane, jaa, vari, K, R, t, r_profiili, z_profiili, sigma_opt, b_cm, gamma, vakiot).sovita(roi, x0, y0, X0, Y0)";
    py::class_<BFit>(m, "BFit")
        .def(py::init<const ArrU8&, const ArrU8&, const ArrU8&, const ArrU8&, const ArrD&, const ArrD&, const ArrD&, const ArrD&, const ArrD&, double, double, double,
                      const py::dict&>(),
             py::arg("mode"), py::arg("lane"), py::arg("jaa"), py::arg("vari"), py::arg("K"), py::arg("R"), py::arg("t"), py::arg("r_profiili"), py::arg("z_profiili"),
             py::arg("sigma_opt"), py::arg("b_cm"), py::arg("gamma"), py::arg("vakiot") = py::dict())
        .def("sovita", &BFit::sovita, py::arg("roi"), py::arg("x0"), py::arg("y0"), py::arg("X0"), py::arg("Y0"), py::arg("laaja") = false, py::arg("esi_x") = std::nan(""),
             py::arg("esi_y") = std::nan(""))
        .def("tarkista_kupera", &BFit::tarkista_kupera, py::arg("X"), py::arg("Y"))
        .def("tarkista_kaista", &BFit::tarkista_kaista, py::arg("X"), py::arg("Y"), py::arg("kaista"))
        .def_property_readonly("monialku", [](const BFit&) { return true; });
    m.def("saikeet", [](int n) { cv::setNumThreads(n); }, py::arg("n"), "OpenCV:n oma rinnakkaistus (tyoprosessissa 1)");
    m.def("versio", []() { return std::string("netti_fit ") + __DATE__ + " " + __TIME__ + " (OpenCV " + CV_VERSION + ")"; });
}
