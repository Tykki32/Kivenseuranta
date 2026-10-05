// ============================================================
// mode_engine.cpp - videon luku ja kalibroinnin moodikuva (pybind11-moduuli mode_engine).
//
// ModeEngine(video_file, tile_size, hw_decode): videotiedosto; read() palauttaa seuraavan ruudun (GIL vapaana).
// ModeEngine(width, height, fps, tile_size):    live-tila; ruudut annetaan set_frame()-kutsulla.
// Kalibrointi: set_transform(stabilointimatriisi) + add_mode_frame() keraa stabiloidun naytteen nykyisesta ruudusta;
// start_mode_background(png) laskee moodikuvan taustasaikeessa, wait_for_mode() odottaa sen valmistumista.
//
// Moodi lasketaan koko pikselille (B, G, R yhdessa): naytteet ryhmitellaan toleranssilla (kanavaero <= 10) ja suurimman
// ryhman keskiarvo on tulos (tasatilanteessa pienempi edustaja-arvo). Kanavakohtainen moodi voisi tuottaa varin, jota
// yhdessakaan naytteessa ei ollut. Lopuksi 3x3-mediaanisuodin poistaa yksittaiset poikkeavat pikselit (ohuet viivat,
// joilla mikaan arvo ei saa selvaa enemmistoa).
// ============================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

class ModeEngine
{
public:
    ModeEngine(const std::string& video_file, int tile_size, bool hw_decode)
        : tile_size_(tile_size)
    {
#ifdef _WIN32
        const int video_backend = cv::CAP_MSMF;
#else
        const int video_backend = cv::CAP_ANY;
#endif
        // Valinnainen laitteistodekoodaus (OpenCV 4.5+, Windowsissa MSMF/D3D11). Ruutu kopioituu silti CPU-muistiin.
        bool hw_opened = false;
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 5)
        if (hw_decode) {
            std::vector<int> hw_params{ cv::CAP_PROP_HW_ACCELERATION, static_cast<int>(cv::VIDEO_ACCELERATION_ANY) };
            cap_.open(video_file, video_backend, hw_params);
            hw_opened = cap_.isOpened();
            std::cout << "[ModeEngine] laitteistodekoodaus "
                      << (hw_opened ? "pyydetty" : "ei kaytettavissa, palataan ohjelmistodekoodaukseen") << std::endl;
        }
#endif
        if (!hw_opened)
            cap_.open(video_file, video_backend);
        if (!cap_.isOpened())
            throw std::runtime_error("Videon avaaminen epaonnistui.");

        width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        total_frames_ = static_cast<int64_t>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
        if (width_ <= 0 || height_ <= 0)
            throw std::runtime_error("Videon koko ei ole kelvollinen.");
        if (fps_ <= 0.0)
            fps_ = 25.0;
        if (tile_size_ <= 0)
            throw std::runtime_error("Virheellinen tile_size.");
        stabilization_matrix_ = cv::Mat::eye(2, 3, CV_64F);
    }

    ModeEngine(int width, int height, double fps, int tile_size)
        : tile_size_(tile_size)
    {
        if (width <= 0 || height <= 0)
            throw std::runtime_error("Videon koko ei ole kelvollinen.");
        if (tile_size_ <= 0)
            throw std::runtime_error("Virheellinen tile_size.");
        width_ = width;
        height_ = height;
        fps_ = fps > 0.0 ? fps : 25.0;
        total_frames_ = 0;
        stabilization_matrix_ = cv::Mat::eye(2, 3, CV_64F);
    }

    ~ModeEngine()
    {
        if (mode_thread_.joinable())
            mode_thread_.join();
        if (cap_.isOpened())
            cap_.release();
    }

    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }
    int64_t total_frames() const { return total_frames_; }
    int mode_frame_count() const { return static_cast<int>(sampled_frames_.size()); }

    // Seuraava ruutu (tyhja taulukko videon lopussa)
    py::array_t<uint8_t> read()
    {
        cv::Mat frame;
        bool ok;
        {
            py::gil_scoped_release release;
            ok = cap_.read(frame);
        }
        if (!ok || frame.empty())
            return py::array_t<uint8_t>();
        current_frame_ = frame;
        return mat_to_numpy(current_frame_);
    }

    // Live-tila: nykyinen ruutu Pythonista (sama vaikutus kuin read())
    void set_frame(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame)
    {
        auto b = frame.request();
        if (b.ndim != 3 || b.shape[2] != 3)
            throw std::runtime_error("set_frame: ruudun pitaa olla HxWx3 uint8");
        cv::Mat view((int)b.shape[0], (int)b.shape[1], CV_8UC3, b.ptr);
        current_frame_ = view.clone();
    }

    void set_transform(py::array_t<double, py::array::c_style | py::array::forcecast> matrix)
    {
        auto buffer = matrix.request();
        if (buffer.ndim != 2 || buffer.shape[0] != 2 || buffer.shape[1] != 3)
            throw std::runtime_error("Stabilointimatriisin pitaa olla kokoa 2x3.");
        std::lock_guard<std::mutex> lock(transform_mutex_);
        stabilization_matrix_ = cv::Mat(2, 3, CV_64F).clone();
        std::memcpy(stabilization_matrix_.ptr<double>(), buffer.ptr, 6 * sizeof(double));
    }

    // Nykyinen ruutu stabiloituna moodin naytteeksi
    void add_mode_frame()
    {
        if (current_frame_.empty())
            throw std::runtime_error("Nykyista framea ei ole.");
        cv::Mat stabilization;
        {
            std::lock_guard<std::mutex> lock(transform_mutex_);
            stabilization = stabilization_matrix_.clone();
        }
        cv::Mat stabilized;
        {
            py::gil_scoped_release release;
            cv::warpAffine(current_frame_, stabilized, stabilization, cv::Size(width_, height_),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        sampled_frames_.push_back(stabilized);
    }

    void start_mode_background(const std::string& output_file)
    {
        if (mode_thread_.joinable())
            throw std::runtime_error("Mode-laskentaketju on jo kaynnissa.");
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            mode_error_.clear();
        }
        mode_thread_ = std::thread([this, output_file]() {
            try {
                save_mode(output_file);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                mode_error_ = e.what();
            } catch (...) {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                mode_error_ = "Tuntematon virhe mode-laskennassa.";
            }
        });
    }

    void wait_for_mode()
    {
        if (mode_thread_.joinable())
            mode_thread_.join();
        std::lock_guard<std::mutex> lock(mode_mutex_);
        if (!mode_error_.empty())
            throw std::runtime_error(mode_error_);
    }

private:
    static py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat)
    {
        if (mat.empty())
            return py::array_t<uint8_t>();
        if (mat.type() != CV_8UC3)
            throw std::runtime_error("Framein pitaa olla CV_8UC3.");
        auto result = py::array_t<uint8_t>({ mat.rows, mat.cols, 3 });
        uint8_t* dst = static_cast<uint8_t*>(result.request().ptr);
        const size_t row_bytes = static_cast<size_t>(mat.cols) * 3;
        for (int y = 0; y < mat.rows; ++y)
            std::memcpy(dst + static_cast<size_t>(y) * row_bytes, mat.ptr<uint8_t>(y), row_bytes);
        return result;
    }

    // Moodikuva tiileittain rinnakkain (enintaan 8 saietta), tallennus PNG:ksi
    void save_mode(const std::string& output_file)
    {
        if (sampled_frames_.empty())
            throw std::runtime_error("Modeen ei ole keratty yhtaan framea.");
        const int sample_count = static_cast<int>(sampled_frames_.size());
        const int width = width_, height = height_;
        cv::Mat mode_image(height, width, CV_8UC3);

        const int tile_size = tile_size_;
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        const int worker_count = std::max(1u, std::min(hardware_threads == 0 ? 8u : hardware_threads, 8u));
        const int tiles_x = (width + tile_size - 1) / tile_size;
        const int tiles_y = (height + tile_size - 1) / tile_size;
        const int total_tiles = tiles_x * tiles_y;
        const int color_tolerance = 10;      // naytteet samaan ryhmaan jos kaikki kanavaerot <= tama
        std::atomic<int> next_tile(0);

        auto worker = [&]() {
            while (true) {
                const int tile_index = next_tile.fetch_add(1);
                if (tile_index >= total_tiles)
                    break;
                const int x0 = (tile_index % tiles_x) * tile_size, y0 = (tile_index / tiles_x) * tile_size;
                const int x1 = std::min(x0 + tile_size, width), y1 = std::min(y0 + tile_size, height);
                const int tw = x1 - x0, th = y1 - y0;
                const size_t pixels = static_cast<size_t>(tw) * static_cast<size_t>(th);
                const size_t slots = pixels * static_cast<size_t>(sample_count);

                // ryhmat pikselia kohti: edustaja (ensimmainen jasen, vain vertailuun), kanavasummat ja jasenmaara
                std::vector<uint32_t> group_repr(slots, 0), group_sum_b(slots, 0), group_sum_g(slots, 0), group_sum_r(slots, 0);
                std::vector<uint16_t> group_counts(slots, 0);
                std::vector<uint16_t> group_found(pixels, 0);

                for (const cv::Mat& sample : sampled_frames_) {
                    for (int y = y0; y < y1; ++y) {
                        const cv::Vec3b* row = sample.ptr<cv::Vec3b>(y);
                        const size_t row_offset = static_cast<size_t>(y - y0) * static_cast<size_t>(tw);
                        for (int x = x0; x < x1; ++x) {
                            const size_t pixel_index = row_offset + static_cast<size_t>(x - x0);
                            const int pb = row[x][0], pg = row[x][1], pr = row[x][2];
                            const size_t base = pixel_index * static_cast<size_t>(sample_count);
                            const int found = group_found[pixel_index];
                            int slot = -1;
                            for (int k = 0; k < found; ++k) {
                                const uint32_t repr = group_repr[base + k];
                                if (std::abs(static_cast<int>((repr >> 16) & 0xFFu) - pb) <= color_tolerance &&
                                    std::abs(static_cast<int>((repr >> 8) & 0xFFu) - pg) <= color_tolerance &&
                                    std::abs(static_cast<int>(repr & 0xFFu) - pr) <= color_tolerance) {
                                    slot = k;
                                    break;
                                }
                            }
                            if (slot >= 0) {
                                ++group_counts[base + slot];
                                group_sum_b[base + slot] += static_cast<uint32_t>(pb);
                                group_sum_g[base + slot] += static_cast<uint32_t>(pg);
                                group_sum_r[base + slot] += static_cast<uint32_t>(pr);
                            } else {
                                group_repr[base + found] = (static_cast<uint32_t>(pb) << 16) | (static_cast<uint32_t>(pg) << 8)
                                                           | static_cast<uint32_t>(pr);
                                group_counts[base + found] = 1;
                                group_sum_b[base + found] = static_cast<uint32_t>(pb);
                                group_sum_g[base + found] = static_cast<uint32_t>(pg);
                                group_sum_r[base + found] = static_cast<uint32_t>(pr);
                                group_found[pixel_index] = static_cast<uint16_t>(found + 1);
                            }
                        }
                    }
                }

                // suurin ryhma (tasatilanteessa pienempi edustaja), tulos sen keskiarvo
                for (int y = y0; y < y1; ++y) {
                    cv::Vec3b* row = mode_image.ptr<cv::Vec3b>(y);
                    const size_t row_offset = static_cast<size_t>(y - y0) * static_cast<size_t>(tw);
                    for (int x = x0; x < x1; ++x) {
                        const size_t pixel_index = row_offset + static_cast<size_t>(x - x0);
                        const size_t base = pixel_index * static_cast<size_t>(sample_count);
                        const int found = group_found[pixel_index];
                        int best_count = -1;
                        uint32_t best_repr = 0;
                        int best_slot = 0;
                        for (int k = 0; k < found; ++k) {
                            const int count = group_counts[base + k];
                            const uint32_t repr = group_repr[base + k];
                            if (count > best_count || (count == best_count && repr < best_repr)) {
                                best_count = count;
                                best_repr = repr;
                                best_slot = k;
                            }
                        }
                        const size_t best_base = base + static_cast<size_t>(best_slot);
                        const uint32_t denom = std::max(1u, static_cast<unsigned int>(best_count));
                        row[x] = cv::Vec3b(static_cast<uint8_t>((group_sum_b[best_base] + denom / 2) / denom),
                                           static_cast<uint8_t>((group_sum_g[best_base] + denom / 2) / denom),
                                           static_cast<uint8_t>((group_sum_r[best_base] + denom / 2) / denom));
                    }
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (int i = 0; i < worker_count; ++i)
            workers.emplace_back(worker);
        for (auto& thread : workers)
            thread.join();

        cv::medianBlur(mode_image, mode_image, 3);
        if (!cv::imwrite(output_file, mode_image))
            throw std::runtime_error("Mode-kuvan tallennus epaonnistui.");
        std::cout << "Mode valmis: " << output_file << std::endl;
        std::cout << "Mode-frameja: " << sample_count << std::endl;
    }

    cv::VideoCapture cap_;
    cv::Mat current_frame_;
    cv::Mat stabilization_matrix_;
    std::mutex transform_mutex_;
    std::vector<cv::Mat> sampled_frames_;
    std::thread mode_thread_;
    std::mutex mode_mutex_;
    std::string mode_error_;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    int64_t total_frames_ = 0;
    int tile_size_ = 128;
};

PYBIND11_MODULE(mode_engine, m)
{
    m.def("build_info", []() { return std::string("mode_engine kaannetty ") + __DATE__ + " " + __TIME__; });
    py::class_<ModeEngine>(m, "ModeEngine")
        .def(py::init<const std::string&, int, bool>(), py::arg("video_file"), py::arg("tile_size") = 128,
             py::arg("hw_decode") = false)
        .def(py::init<int, int, double, int>(), py::arg("width"), py::arg("height"), py::arg("fps"),
             py::arg("tile_size") = 128)
        .def("width", &ModeEngine::width)
        .def("height", &ModeEngine::height)
        .def("fps", &ModeEngine::fps)
        .def("total_frames", &ModeEngine::total_frames)
        .def("mode_frame_count", &ModeEngine::mode_frame_count)
        .def("read", &ModeEngine::read)
        .def("set_frame", &ModeEngine::set_frame)
        .def("set_transform", &ModeEngine::set_transform)
        .def("add_mode_frame", &ModeEngine::add_mode_frame)
        .def("start_mode_background", &ModeEngine::start_mode_background)
        .def("wait_for_mode", &ModeEngine::wait_for_mode);
}
