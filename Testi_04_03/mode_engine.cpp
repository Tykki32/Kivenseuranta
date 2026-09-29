#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;


// ============================================================
// MODE ENGINE
// ============================================================

class ModeEngine
{
public:

    // ========================================================
    // CONSTRUCTOR
    // ========================================================

    ModeEngine(
        const std::string& video_file,
        int tile_size = 128
    )
        : tile_size_(tile_size)
    {
        cap_.open(
            video_file,
            cv::CAP_MSMF
        );

        if (!cap_.isOpened())
        {
            throw std::runtime_error(
                "Videon avaaminen epaonnistui."
            );
        }

        width_ = static_cast<int>(
            cap_.get(cv::CAP_PROP_FRAME_WIDTH)
            );

        height_ = static_cast<int>(
            cap_.get(cv::CAP_PROP_FRAME_HEIGHT)
            );

        fps_ = cap_.get(
            cv::CAP_PROP_FPS
        );

        total_frames_ = static_cast<int64_t>(
            cap_.get(cv::CAP_PROP_FRAME_COUNT)
            );

        if (width_ <= 0 || height_ <= 0)
        {
            throw std::runtime_error(
                "Videon koko ei ole kelvollinen."
            );
        }

        if (fps_ <= 0.0)
        {
            fps_ = 25.0;
        }

        if (tile_size_ <= 0)
        {
            throw std::runtime_error(
                "Virheellinen tile_size."
            );
        }

        stabilization_matrix_ =
            cv::Mat::eye(
                2,
                3,
                CV_64F
            );

        // Perspektiivimatriisin oletusarvo:
        // identiteetti.
        perspective_matrix_ =
            cv::Mat::eye(
                3,
                3,
                CV_64F
            );

        perspective_enabled_ = false;
    }


    // ========================================================
    // DESTRUCTOR
    // ========================================================

    ~ModeEngine()
    {
        if (mode_thread_.joinable())
        {
            mode_thread_.join();
        }

        if (video_writer_.isOpened())
        {
            video_writer_.release();
        }

        if (cap_.isOpened())
        {
            cap_.release();
        }
    }


    // ========================================================
    // GETTERS
    // ========================================================

    int width() const
    {
        return width_;
    }


    int height() const
    {
        return height_;
    }


    double fps() const
    {
        return fps_;
    }


    int64_t total_frames() const
    {
        return total_frames_;
    }

    int mode_frame_count()
    {
        return static_cast<int>(
            sampled_frames_.size()
            );
    }


    // ========================================================
    // PERSPEKTIIVIKORJAUKSEN ASETTAMINEN
    //
    // Python antaa 3x3 homografian.
    // ========================================================

    void set_perspective_matrix(
        py::array_t<double,
        py::array::c_style | py::array::forcecast> matrix
    )
    {
        auto buffer = matrix.request();

        if (buffer.ndim != 2)
        {
            throw std::runtime_error(
                "Perspektiivimatriisin pitaa olla 2-ulotteinen."
            );
        }

        if (buffer.shape[0] != 3 ||
            buffer.shape[1] != 3)
        {
            throw std::runtime_error(
                "Perspektiivimatriisin pitaa olla kokoa 3x3."
            );
        }

        const double* data =
            static_cast<const double*>(
                buffer.ptr
                );

        std::lock_guard<std::mutex> lock(
            transform_mutex_
        );

        perspective_matrix_ =
            cv::Mat(
                3,
                3,
                CV_64F
            ).clone();

        std::memcpy(
            perspective_matrix_.ptr<double>(),
            data,
            9 * sizeof(double)
        );

        perspective_enabled_ = true;
    }


    // ========================================================
    // PERSPEKTIIVIKORJAUKSEN POISTAMINEN
    // ========================================================

    void disable_perspective()
    {
        std::lock_guard<std::mutex> lock(
            transform_mutex_
        );

        perspective_enabled_ = false;

        perspective_matrix_ =
            cv::Mat::eye(
                3,
                3,
                CV_64F
            );
    }


    // ========================================================
    // MODE BACKGROUND THREAD
    // ========================================================

    void start_mode_background(
        const std::string& output_file
    )
    {
        if (mode_thread_.joinable())
        {
            throw std::runtime_error(
                "Mode-laskentaketju on jo kaynnissa."
            );
        }

        {
            std::lock_guard<std::mutex> lock(
                mode_mutex_
            );

            mode_error_.clear();
            mode_ready_ = false;
            mode_output_file_ = output_file;
        }

        mode_thread_ =
            std::thread(
                [this, output_file]()
                {
                    try
                    {
                        save_mode(
                            output_file
                        );
                    }
                    catch (const std::exception& e)
                    {
                        std::lock_guard<std::mutex> lock(
                            mode_mutex_
                        );

                        mode_error_ =
                            e.what();

                        mode_ready_ = false;
                    }
                    catch (...)
                    {
                        std::lock_guard<std::mutex> lock(
                            mode_mutex_
                        );

                        mode_error_ =
                            "Tuntematon virhe mode-laskennassa.";

                        mode_ready_ = false;
                    }
                }
        );
    }


    // ========================================================
    // WAIT FOR MODE
    // ========================================================

    void wait_for_mode()
    {
        if (mode_thread_.joinable())
        {
            mode_thread_.join();
        }

        std::lock_guard<std::mutex> lock(
            mode_mutex_
        );

        if (!mode_error_.empty())
        {
            throw std::runtime_error(
                mode_error_
            );
        }
    }


    // ========================================================
    // READ
    // ========================================================

    py::array_t<uint8_t> read()
    {
        cv::Mat frame;

        bool ok;

        {
            py::gil_scoped_release release;

            ok = cap_.read(
                frame
            );
        }

        if (!ok || frame.empty())
        {
            return py::array_t<uint8_t>();
        }

        current_frame_ =
            frame;

        return mat_to_numpy(
            current_frame_
        );
    }


    // ========================================================
    // SET STABILIZATION MATRIX
    // ========================================================

    void set_transform(
        py::array_t<double,
        py::array::c_style | py::array::forcecast> matrix
    )
    {
        auto buffer = matrix.request();

        if (buffer.ndim != 2)
        {
            throw std::runtime_error(
                "Stabilointimatriisin pitaa olla 2-ulotteinen."
            );
        }

        if (buffer.shape[0] != 2 ||
            buffer.shape[1] != 3)
        {
            throw std::runtime_error(
                "Stabilointimatriisin pitaa olla kokoa 2x3."
            );
        }

        const double* data =
            static_cast<const double*>(
                buffer.ptr
                );

        std::lock_guard<std::mutex> lock(
            transform_mutex_
        );

        stabilization_matrix_ =
            cv::Mat(
                2,
                3,
                CV_64F
            ).clone();

        std::memcpy(
            stabilization_matrix_.ptr<double>(),
            data,
            6 * sizeof(double)
        );
    }


    // ========================================================
    // ADD MODE FRAME
    //
    // Frame:
    //
    // raw
    //   ?
    // stabilization
    //   ?
    // perspective correction
    //   ?
    // mode sample
    // ========================================================

    void add_mode_frame()
    {
        if (current_frame_.empty())
        {
            throw std::runtime_error(
                "Nykyista framea ei ole."
            );
        }

        cv::Mat stabilization;
        cv::Mat perspective;
        bool perspective_enabled;

        {
            std::lock_guard<std::mutex> lock(
                transform_mutex_
            );

            stabilization =
                stabilization_matrix_.clone();

            perspective =
                perspective_matrix_.clone();

            perspective_enabled =
                perspective_enabled_;
        }

        cv::Mat stabilized;

        {
            py::gil_scoped_release release;

            cv::warpAffine(
                current_frame_,
                stabilized,
                stabilization,
                cv::Size(
                    width_,
                    height_
                ),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(
                    0,
                    0,
                    0
                )
            );
        }

        cv::Mat corrected;

        if (perspective_enabled)
        {
            py::gil_scoped_release release;

            cv::warpPerspective(
                stabilized,
                corrected,
                perspective,
                cv::Size(
                    perspective_width_,
                    perspective_height_
                ),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(
                    0,
                    0,
                    0
                )
            );
        }
        else
        {
            corrected =
                stabilized;
        }

        sampled_frames_.push_back(
            corrected
        );
    }


    // ========================================================
    // SET PERSPECTIVE OUTPUT SIZE
    // ========================================================

    void set_perspective_output_size(
        int width,
        int height
    )
    {
        if (width <= 0 ||
            height <= 0)
        {
            throw std::runtime_error(
                "Perspektiivikorjauksen tulostuskoko "
                "on virheellinen."
            );
        }

        perspective_width_ =
            width;

        perspective_height_ =
            height;
    }


// ========================================================
// PROCESS FRAME
//
// Tata kutsutaan jokaiselle videoruudulle.
//
// raw
//   -> stabilization
//   -> perspective correction
//   -> [mode filter, jos moodi valmis]
//   -> video
//
// distance_threshold < 0:
//     EI moodisuodatusta
//
// distance_threshold >= 0:
//     moodisuodatus
//
// Jos moodikuvaa viela lasketaan:
//     frame jatetaan kirjoittamatta.
//
// Tama mahdollistaa:
// ENABLE_MODE_FILTER = false
//     -> video alkaa heti
//
// ENABLE_MODE_FILTER = true
//     -> video alkaa heti kun moodikuva valmistuu.
// ========================================================

    void process_frame(
        double distance_threshold
    )
    {
        if (current_frame_.empty())
        {
            return;
        }


        // ----------------------------------------------------
        // Kopioidaan tamanhetkiset muunnokset
        // ----------------------------------------------------

        cv::Mat stabilization;
        cv::Mat perspective;
        bool perspective_enabled;

        {
            std::lock_guard<std::mutex> lock(
                transform_mutex_
            );

            stabilization =
                stabilization_matrix_.clone();

            perspective =
                perspective_matrix_.clone();

            perspective_enabled =
                perspective_enabled_;
        }


        // ----------------------------------------------------
        // STABILOINTI
        // ----------------------------------------------------

        cv::Mat stabilized;

        {
            py::gil_scoped_release release;

            cv::warpAffine(
                current_frame_,
                stabilized,
                stabilization,
                cv::Size(
                    width_,
                    height_
                ),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(
                    0,
                    0,
                    0
                )
            );
        }


        // ----------------------------------------------------
        // PERSPEKTIIVIKORJAUS
        // ----------------------------------------------------

        cv::Mat corrected;

        if (perspective_enabled)
        {
            {
                py::gil_scoped_release release;

                cv::warpPerspective(
                    stabilized,
                    corrected,
                    perspective,
                    cv::Size(
                        perspective_width_,
                        perspective_height_
                    ),
                    cv::INTER_LINEAR,
                    cv::BORDER_CONSTANT,
                    cv::Scalar(
                        0,
                        0,
                        0
                    )
                );
            }
        }
        else
        {
            corrected =
                stabilized;
        }


        // ----------------------------------------------------
        // SELVITETaaN MOODIN TILA
        // ----------------------------------------------------

        bool ready;

        {
            std::lock_guard<std::mutex> lock(
                mode_mutex_
            );

            ready =
                mode_ready_;
        }


        // ----------------------------------------------------
        // MODE FILTER POIS
        //
        // distance_threshold < 0 tarkoittaa:
        // kirjoita kuva sellaisenaan.
        // ----------------------------------------------------

        if (distance_threshold < 0.0)
        {
            if (!video_writer_.isOpened())
            {
                open_video_writer();
            }

            video_writer_.write(
                corrected
            );

            return;
        }


        // ----------------------------------------------------
        // MODE FILTER PaaLLa, MUTTA MOODI EI OLE VALMIS
        //
        // Talloin ruutua ei kirjoiteta.
        //
        // Tama on tarkoituksellista:
        // ensimmainen videoruudun timestamp vastaa hetkea,
        // jolloin moodikuva on valmis.
        // ----------------------------------------------------

        if (!ready)
        {
            return;
        }


        // ----------------------------------------------------
        // AVAA VIDEO WRITER ENSIMMaISELLa
        // SUODATETTAVALLA RUUDULLA
        // ----------------------------------------------------

        if (!video_writer_.isOpened())
        {
            open_video_writer();
        }


        // ----------------------------------------------------
        // MOODISUODATUS
        // ----------------------------------------------------

        cv::Mat filtered;

        {
            py::gil_scoped_release release;

            filter_frame(
                corrected,
                filtered,
                distance_threshold
            );
        }


        // ----------------------------------------------------
        // KIRJOITA
        // ----------------------------------------------------

        video_writer_.write(
            filtered
        );
    }


private:

    // ========================================================
    // MAT -> NUMPY
    // ========================================================

    py::array_t<uint8_t> mat_to_numpy(
        const cv::Mat& mat
    )
    {
        if (mat.empty())
        {
            return py::array_t<uint8_t>();
        }

        if (mat.type() != CV_8UC3)
        {
            throw std::runtime_error(
                "Framein pitaa olla CV_8UC3."
            );
        }

        auto result =
            py::array_t<uint8_t>(
                {
                    mat.rows,
                    mat.cols,
                    3
                }
        );

        auto buffer =
            result.request();

        uint8_t* dst =
            static_cast<uint8_t*>(
                buffer.ptr
                );

        const size_t row_bytes =
            static_cast<size_t>(
                mat.cols
                ) * 3;

        for (int y = 0;
            y < mat.rows;
            ++y)
        {
            std::memcpy(
                dst +
                static_cast<size_t>(y) *
                row_bytes,

                mat.ptr<uint8_t>(y),

                row_bytes
            );
        }

        return result;
    }


    // ========================================================
    // SAVE MODE
    //
    // Exact pixel-wise mode.
    // ========================================================

    void save_mode(
        const std::string& output_file
    )
    {
        if (sampled_frames_.empty())
        {
            throw std::runtime_error(
                "Modeen ei ole keratty yhtaan framea."
            );
        }

        const int sample_count =
            static_cast<int>(
                sampled_frames_.size()
                );

        const int width =
            perspective_enabled_
            ? perspective_width_
            : width_;

        const int height =
            perspective_enabled_
            ? perspective_height_
            : height_;


        cv::Mat mode_image(
            height,
            width,
            CV_8UC3
        );


        // ----------------------------------------------------
        // Tile-koko
        // ----------------------------------------------------

        const int tile_size =
            tile_size_;


        // ----------------------------------------------------
        // Tyontekijoiden maara
        // ----------------------------------------------------

        const unsigned int hardware_threads =
            std::thread::hardware_concurrency();

        const int worker_count =
            std::max(
                1u,
                std::min(
                    hardware_threads == 0
                    ? 8u
                    : hardware_threads,
                    8u
                )
            );


        // ----------------------------------------------------
        // Tilejen maara
        // ----------------------------------------------------

        const int tiles_x =
            (width + tile_size - 1)
            / tile_size;

        const int tiles_y =
            (height + tile_size - 1)
            / tile_size;

        const int total_tiles =
            tiles_x * tiles_y;


        // ----------------------------------------------------
        // Atominen seuraava tile
        // ----------------------------------------------------

        std::atomic<int> next_tile(
            0
        );


        // ----------------------------------------------------
        // Worker
        // ----------------------------------------------------

        auto worker =
            [&]()
        {
            while (true)
            {
                const int tile_index =
                    next_tile.fetch_add(
                        1
                    );

                if (tile_index >= total_tiles)
                {
                    break;
                }

                const int tile_y =
                    tile_index / tiles_x;

                const int tile_x =
                    tile_index % tiles_x;

                const int x0 =
                    tile_x *
                    tile_size;

                const int y0 =
                    tile_y *
                    tile_size;

                const int x1 =
                    std::min(
                        x0 + tile_size,
                        width
                    );

                const int y1 =
                    std::min(
                        y0 + tile_size,
                        height
                    );

                const int tw =
                    x1 - x0;

                const int th =
                    y1 - y0;


                // ------------------------------------------------
                // Havaitut (B,G,R)-arvot pikselia kohti.
                //
                // KORJATTU (Testi_01_01): alkuperainen versio laski
                // moodin JOKAISELLE KANAVALLE ERIKSEEN (3 x 256-
                // korin histogrammi). Tama voi tuottaa pikselin,
                // jonka B, G ja R tulevat KOLMESTA ERI naytteesta -
                // varin, jota missaan yksittaisessa framessa ei
                // koskaan ollut. Kaukaisen pesan seudulla (pieni
                // pikselimaara raakakuvassa, topdown-warpissa
                // valtava suurennuskerroin) tama nakyi keinote-
                // koisina, kirkkaina pystyraitoina jotka konta-
                // minoivat pesan reunan tunnistuksen kalibroin-
                // nissa.
                //
                // Korjaus: moodi lasketaan KOKO PIKSELILLE (B,G,R
                // yhdessa) - lopputulos on AINA jonkin todellisen
                // naytteen oikea, aidosti esiintynyt vari. Naytteita
                // on vain muutamia kymmenia, joten per-pikseli
                // lineaarihaku pienesta listasta (ei 256-korin
                // histogrammia per kanava) on seka oikeampi etta
                // kevyempi (myos muistissa).
                // ------------------------------------------------

                const size_t pixels =
                    static_cast<size_t>(
                        tw
                        ) *
                    static_cast<size_t>(
                        th
                        );

                // Ryhman EDUSTAJA-arvo (ensimmainen havaittu jasen -
                // kaytetaan VAIN uusien naytteiden toleranssiver-
                // tailuun, ei lopputulokseen).
                std::vector<uint32_t> group_repr(
                    pixels * static_cast<size_t>(sample_count),
                    0
                );

                // Ryhman jasenten kanavasummat - lopullinen vari on
                // naiden KESKIARVO, ei pelkka ensimmainen havainto
                // (katso perustelu ylla).
                std::vector<uint32_t> group_sum_b(
                    pixels * static_cast<size_t>(sample_count),
                    0
                );

                std::vector<uint32_t> group_sum_g(
                    pixels * static_cast<size_t>(sample_count),
                    0
                );

                std::vector<uint32_t> group_sum_r(
                    pixels * static_cast<size_t>(sample_count),
                    0
                );

                std::vector<uint16_t> group_counts(
                    pixels * static_cast<size_t>(sample_count),
                    0
                );

                std::vector<uint16_t> group_found(
                    pixels,
                    0
                );

                // Tarkka pikseliyhtasuuruus EI riita ryhmittelyyn:
                // videon oma pakkauskohina + muutaman pikselin
                // jaannosvirhe stabiloinnissa nayttein valilla
                // tekee samasta, muuttumattomasta taustasta lahes
                // aina hieman eri 8-bittisen arvon joka naytteessa,
                // jolloin "tasan sama arvo" -ryhmittely hajottaa
                // OIKEAN, selvan enemmiston moneksi 1 kpl:n ryhmaksi.
                // Ratkaisu: kaksi nayteta kuuluvat samaan ryhmaan
                // jos ne ovat toleranssin sisalla TOISISTAAN (ei
                // bittitarkkaa yhtasuuruutta).
                const int color_tolerance = 10;


                // ------------------------------------------------
                // Kaydaan kaikki mode-framet
                // ------------------------------------------------

                for (const cv::Mat& sample :
                    sampled_frames_)
                {
                    for (int y = y0;
                        y < y1;
                        ++y)
                    {
                        const cv::Vec3b* row =
                            sample.ptr<cv::Vec3b>(
                                y
                                );

                        const int local_y =
                            y - y0;

                        const size_t row_offset =
                            static_cast<size_t>(
                                local_y
                                ) *
                            static_cast<size_t>(
                                tw
                                );

                        for (int x = x0;
                            x < x1;
                            ++x)
                        {
                            const int local_x =
                                x - x0;

                            const size_t pixel_index =
                                row_offset +
                                static_cast<size_t>(
                                    local_x
                                    );

                            const cv::Vec3b pixel =
                                row[x];

                            const int pb =
                                static_cast<int>(pixel[0]);
                            const int pg =
                                static_cast<int>(pixel[1]);
                            const int pr =
                                static_cast<int>(pixel[2]);

                            const size_t base =
                                pixel_index *
                                static_cast<size_t>(sample_count);

                            const int found =
                                group_found[pixel_index];

                            int slot = -1;

                            for (int k = 0;
                                k < found;
                                ++k)
                            {
                                const uint32_t repr =
                                    group_repr[base + k];

                                const int db = std::abs(
                                    static_cast<int>((repr >> 16) & 0xFFu) - pb
                                    );
                                const int dg = std::abs(
                                    static_cast<int>((repr >> 8) & 0xFFu) - pg
                                    );
                                const int dr = std::abs(
                                    static_cast<int>(repr & 0xFFu) - pr
                                    );

                                if (db <= color_tolerance &&
                                    dg <= color_tolerance &&
                                    dr <= color_tolerance)
                                {
                                    slot = k;
                                    break;
                                }
                            }

                            if (slot >= 0)
                            {
                                ++group_counts[base + slot];
                                group_sum_b[base + slot] +=
                                    static_cast<uint32_t>(pb);
                                group_sum_g[base + slot] +=
                                    static_cast<uint32_t>(pg);
                                group_sum_r[base + slot] +=
                                    static_cast<uint32_t>(pr);
                            }
                            else
                            {
                                group_repr[base + found] =
                                    (static_cast<uint32_t>(pb) << 16) |
                                    (static_cast<uint32_t>(pg) << 8) |
                                    static_cast<uint32_t>(pr);
                                group_counts[base + found] = 1;
                                group_sum_b[base + found] =
                                    static_cast<uint32_t>(pb);
                                group_sum_g[base + found] =
                                    static_cast<uint32_t>(pg);
                                group_sum_r[base + found] =
                                    static_cast<uint32_t>(pr);
                                group_found[pixel_index] =
                                    static_cast<uint16_t>(found + 1);
                            }
                        }
                    }
                }


                // ------------------------------------------------
                // Valitaan suurin ryhma, tulos sen KESKIARVO.
                //
                // Tasatilanteessa pienempi edustaja-arvo (sama
                // periaate kuin alkuperaisessa: pienempi voittaa).
                // ------------------------------------------------

                cv::Mat* output =
                    &mode_image;

                for (int y = y0;
                    y < y1;
                    ++y)
                {
                    cv::Vec3b* row =
                        output->ptr<cv::Vec3b>(
                            y
                            );

                    const int local_y =
                        y - y0;

                    const size_t row_offset =
                        static_cast<size_t>(
                            local_y
                            ) *
                        static_cast<size_t>(
                            tw
                            );


                    for (int x = x0;
                        x < x1;
                        ++x)
                    {
                        const int local_x =
                            x - x0;

                        const size_t pixel_index =
                            row_offset +
                            static_cast<size_t>(
                                local_x
                                );

                        const size_t base =
                            pixel_index *
                            static_cast<size_t>(sample_count);

                        const int found =
                            group_found[pixel_index];

                        int best_count = -1;
                        uint32_t best_repr = 0;
                        int best_slot = 0;

                        for (int k = 0;
                            k < found;
                            ++k)
                        {
                            const int count =
                                group_counts[base + k];

                            const uint32_t repr =
                                group_repr[base + k];

                            if (
                                count > best_count ||
                                (
                                    count == best_count &&
                                    repr < best_repr
                                    )
                                )
                            {
                                best_count = count;
                                best_repr = repr;
                                best_slot = k;
                            }
                        }

                        const size_t best_base =
                            base + static_cast<size_t>(best_slot);

                        const uint32_t denom =
                            std::max(
                                1u,
                                static_cast<unsigned int>(best_count)
                                );

                        row[x] =
                            cv::Vec3b(
                                static_cast<uint8_t>(
                                    (group_sum_b[best_base] + denom / 2) / denom
                                    ),
                                static_cast<uint8_t>(
                                    (group_sum_g[best_base] + denom / 2) / denom
                                    ),
                                static_cast<uint8_t>(
                                    (group_sum_r[best_base] + denom / 2) / denom
                                    )
                            );
                    }
                }
            }
        };


        // ----------------------------------------------------
        // Kaynnistetaan workerit
        // ----------------------------------------------------

        std::vector<std::thread> workers;

        workers.reserve(
            worker_count
        );

        for (int i = 0;
            i < worker_count;
            ++i)
        {
            workers.emplace_back(
                worker
            );
        }


        // ----------------------------------------------------
        // Odotetaan
        // ----------------------------------------------------

        for (auto& thread :
            workers)
        {
            thread.join();
        }


        // ----------------------------------------------------
        // Suola-pippuri-kohinan siivous.
        //
        // Ohuilla/pienilla yksityiskohdilla (esim. jaahan
        // maalattu ohut viiva) muutaman pikselin jaannosvirhe
        // stabiloinnissa nayttein valilla riittaa siihen ettei
        // yhdellakaan (B,G,R)-arvolla ole selvaa enemmistoa -
        // lopputulos on tallon satunnainen pikseli kohden.
        // Koska moodi lasketaan nyt KOKO PIKSELILLE (katso
        // workerin kommentti ylla), tama nakyy yksittaisina
        // "suola-pippuri"-pikseleina, ei enaa vanhan per-kanava-
        // version kaltaisena keinotekoisena mutta visuaalisesti
        // "sileana" varisekoituksena. Pieni mediaanisuodin
        // siivoaa nama yksittaiset poikkeavat pikselit sailyttaen
        // oikeat, isommat rakenteet (renkaat, viivat, teksti).
        // ----------------------------------------------------

        cv::medianBlur(
            mode_image,
            mode_image,
            3
        );


        // ----------------------------------------------------
        // Tallennetaan mode PNG
        // ----------------------------------------------------

        if (!cv::imwrite(
            output_file,
            mode_image
        ))
        {
            throw std::runtime_error(
                "Mode-kuvan tallennus epaonnistui."
            );
        }


        // ----------------------------------------------------
        // Mode valmis
        // ----------------------------------------------------

        {
            std::lock_guard<std::mutex> lock(
                mode_mutex_
            );

            mode_image_ =
                std::move(
                    mode_image
                );

            mode_ready_ =
                true;
        }


        std::cout
            << "Mode valmis: "
            << output_file
            << std::endl;

        std::cout
            << "Mode-frameja: "
            << sample_count
            << std::endl;
    }


    // ========================================================
    // OPEN VIDEO WRITER
    // ========================================================

    void open_video_writer()
    {
        if (video_writer_.isOpened())
        {
            return;
        }


        const std::string output_file =
            create_filtered_filename(
                mode_output_file_
            );


        const int output_width =
            perspective_enabled_
            ? perspective_width_
            : width_;

        const int output_height =
            perspective_enabled_
            ? perspective_height_
            : height_;


        video_writer_.open(
            output_file,
            cv::VideoWriter::fourcc(
                'M',
                'J',
                'P',
                'G'
            ),
            fps_,
            cv::Size(
                output_width,
                output_height
            ),
            true
        );


        if (!video_writer_.isOpened())
        {
            throw std::runtime_error(
                "Suodatetun videon VideoWriterin "
                "avaaminen epaonnistui."
            );
        }


        video_output_file_ =
            output_file;


        std::cout
            << "Suodatettu video: "
            << output_file
            << std::endl;
    }


    // ========================================================
    // CREATE FILTERED FILENAME
    // ========================================================

    std::string create_filtered_filename(
        const std::string& mode_file
    )
    {
        const size_t dot =
            mode_file.find_last_of(
                '.'
            );

        if (dot == std::string::npos)
        {
            return mode_file +
                "_filtered.avi";
        }

        return
            mode_file.substr(
                0,
                dot
            )
            +
            "_filtered.avi";
    }


    // ========================================================
    // FILTER FRAME
    //
    // Euclidean RGB distance.
    //
    // <= threshold:
    //     white
    //
    // > threshold:
    //     alkuperainen stabiloitu/
    //     perspektiivikorjattu pixel
    // ========================================================

    void filter_frame(
        const cv::Mat& stabilized,
        cv::Mat& filtered,
        double distance_threshold
    )
    {
        cv::Mat mode;

        {
            std::lock_guard<std::mutex> lock(
                mode_mutex_
            );

            mode =
                mode_image_;
        }


        if (mode.empty())
        {
            throw std::runtime_error(
                "Mode-kuva puuttuu."
            );
        }


        if (mode.size() !=
            stabilized.size())
        {
            throw std::runtime_error(
                "Mode-kuvan ja videoruudun koko "
                "ei tasmaa."
            );
        }


        if (distance_threshold < 0.0)
        {
            distance_threshold = 0.0;
        }


        // ----------------------------------------------------
        // HUOM:
        //
        // Tama EI voi olla constexpr,
        // koska distance_threshold tulee
        // Pythonista ajon aikana.
        // ----------------------------------------------------

        const double threshold_squared =
            distance_threshold *
            distance_threshold;


        filtered =
            stabilized.clone();


        for (int y = 0;
            y < stabilized.rows;
            ++y)
        {
            const cv::Vec3b* src =
                stabilized.ptr<cv::Vec3b>(
                    y
                    );

            const cv::Vec3b* mode_row =
                mode.ptr<cv::Vec3b>(
                    y
                    );

            cv::Vec3b* dst =
                filtered.ptr<cv::Vec3b>(
                    y
                    );


            for (int x = 0;
                x < stabilized.cols;
                ++x)
            {
                const int db =
                    static_cast<int>(
                        src[x][0]
                        )
                    -
                    static_cast<int>(
                        mode_row[x][0]
                        );

                const int dg =
                    static_cast<int>(
                        src[x][1]
                        )
                    -
                    static_cast<int>(
                        mode_row[x][1]
                        );

                const int dr =
                    static_cast<int>(
                        src[x][2]
                        )
                    -
                    static_cast<int>(
                        mode_row[x][2]
                        );


                const double distance_squared =
                    static_cast<double>(
                        db * db
                        )
                    +
                    static_cast<double>(
                        dg * dg
                        )
                    +
                    static_cast<double>(
                        dr * dr
                        );


                if (
                    distance_squared <=
                    threshold_squared
                    )
                {
                    dst[x] =
                        cv::Vec3b(
                            255,
                            255,
                            255
                        );
                }
                else
                {
                    dst[x] =
                        src[x];
                }
            }
        }
    }


    // ========================================================
    // PRIVATE DATA
    // ========================================================

    cv::VideoCapture cap_;

    cv::Mat current_frame_;


    // --------------------------------------------------------
    // Stabilointi
    // --------------------------------------------------------

    cv::Mat stabilization_matrix_;


    // --------------------------------------------------------
    // Perspektiivi
    // --------------------------------------------------------

    cv::Mat perspective_matrix_;

    bool perspective_enabled_ =
        false;

    int perspective_width_ =
        400;

    int perspective_height_ =
        4041;


    // --------------------------------------------------------
    // Transform mutex
    // --------------------------------------------------------

    std::mutex transform_mutex_;


    // --------------------------------------------------------
    // Mode
    // --------------------------------------------------------

    std::vector<cv::Mat> sampled_frames_;

    cv::Mat mode_image_;

    std::thread mode_thread_;

    std::mutex mode_mutex_;

    std::string mode_error_;

    bool mode_ready_ =
        false;

    std::string mode_output_file_;


    // --------------------------------------------------------
    // Video output
    // --------------------------------------------------------

    cv::VideoWriter video_writer_;

    std::string video_output_file_;


    // --------------------------------------------------------
    // Video properties
    // --------------------------------------------------------

    int width_ =
        0;

    int height_ =
        0;

    double fps_ =
        25.0;

    int64_t total_frames_ =
        0;


    // --------------------------------------------------------
    // Tile
    // --------------------------------------------------------

    int tile_size_ =
        128;
};


// ============================================================
// PYBIND11
// ============================================================

PYBIND11_MODULE(
    mode_engine,
    m
)
{
    py::class_<ModeEngine>(
        m,
        "ModeEngine"
        )

        .def(
            py::init<
            const std::string&,
            int
            >(),
            py::arg("video_file"),
            py::arg("tile_size") = 128
        )

        .def(
            "width",
            &ModeEngine::width
        )

        .def(
            "height",
            &ModeEngine::height
        )

        .def(
            "fps",
            &ModeEngine::fps
        )

        .def(
            "total_frames",
            &ModeEngine::total_frames
        )

        .def(
            "mode_frame_count",
            &ModeEngine::mode_frame_count
        )

        .def(
            "read",
            &ModeEngine::read
        )

        .def(
            "set_transform",
            &ModeEngine::set_transform
        )

        .def(
            "set_perspective_matrix",
            &ModeEngine::set_perspective_matrix
        )

        .def(
            "disable_perspective",
            &ModeEngine::disable_perspective
        )

        .def(
            "set_perspective_output_size",
            &ModeEngine::set_perspective_output_size
        )

        .def(
            "add_mode_frame",
            &ModeEngine::add_mode_frame
        )

        .def(
            "start_mode_background",
            &ModeEngine::start_mode_background
        )

        .def(
            "wait_for_mode",
            &ModeEngine::wait_for_mode
        )

        .def(
            "process_frame",
            &ModeEngine::process_frame,
            py::arg("distance_threshold")
        );
}