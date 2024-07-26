#include <iostream>
#include "include/NeaTSL.hpp"
#include "include/AdaptiveApproximation.hpp"

using x_t = uint32_t;
using y_t = int64_t;

template<int64_t error_bound, typename T0, typename T1, typename T2>
void lossy_compression(const auto &fn, std::ostream &out, bool first_size = false) {
    neats::lossy_compressor<x_t, y_t, error_bound, T0, T1, T2> lc;
    auto data = pfa::algorithm::io::read_data_binary<y_t, y_t>(fn, first_size);
    auto range = *std::max_element(data.begin(), data.end()) - *std::min_element(data.begin(), data.end());
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    std::for_each(data.begin(), data.end(), [min_data](auto &d) { d -= (min_data - error_bound); });
    lc.partitioning(data.begin(), data.end());

    std::cout << "Number of segments: " << lc.num_partitions() << std::endl;
    std::vector<y_t> decompressed(data.size());
    lc.decompress(decompressed.begin(), decompressed.end());

    for (auto i = 0; i < data.size(); ++i) {
        auto err = data[i] - decompressed[i];
        if (err > error_bound || err < -(error_bound + 1)) {
            std::cout << i << ": " << data[i] << "!=" << decompressed[i] << std::endl;
            exit(1);
        }
    }

    auto uncompressed_bit_size = data.size() * sizeof(y_t) * 8;
    auto compressed_bit_size = lc.size_in_bits();
    auto cr = (long double) (compressed_bit_size) / (long double) (uncompressed_bit_size);

    out << "NeaTS-Lossy," << fn << "," << error_bound << "," << uncompressed_bit_size << "," << data.size() << ","
        << compressed_bit_size << "," << cr << "," << range << std::endl;
}

template<int64_t error_bound, typename T0, typename T1, typename T2>
void pla_compression(const auto &fn, std::ostream &out, bool first_size = false) {
    using poa_t = pfa::piecewise_optimal_approximation<x_t, y_t, T0, T1, T2>;
    auto data = pfa::algorithm::io::read_data_binary<y_t, y_t>(fn, first_size);
    auto range = *std::max_element(data.begin(), data.end()) - *std::min_element(data.begin(), data.end());
    //auto min_data = *std::min_element(data.begin(), data.end());
    //min_data = min_data < 0 ? (min_data - 1) : -1;
    //std::for_each(data.begin(), data.end(), [min_data](auto &d) { d -= min_data; });
    auto res = pfa::algorithm::make_pla<typename poa_t::pla_t, error_bound>(data.begin(), data.end());

    auto uncompressed_bit_size = data.size() * sizeof(y_t) * 8;
    size_t compressed_bit_size = 0;
    compressed_bit_size += res.size() * sizeof(x_t) * 8; // starting positions // num pla
    compressed_bit_size += res.size() * sizeof(T1) * 8; // slope
    compressed_bit_size += res.size() * sizeof(T2) * 8; // intercept

    std::cout << "Number of segments: " << res.size() << std::endl;

    auto cr = (long double) (compressed_bit_size) / (long double) (uncompressed_bit_size);

    out << "PLA," << fn << "," << error_bound << "," << uncompressed_bit_size << "," << data.size() << ","
        << compressed_bit_size << "," << cr << "," << range << std::endl;
}


template<int64_t error_bound, typename T0, typename T1, typename T2>
void adaptive_approximation_full(const auto &fn, std::ostream &out, bool first_size = false) {
    auto data = pfa::algorithm::io::read_data_binary<y_t, y_t>(fn, first_size);
    auto range = *std::max_element(data.begin(), data.end()) - *std::min_element(data.begin(), data.end());
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    std::for_each(data.begin(), data.end(), [min_data](auto &d) { d -= min_data; });

    auto AA = adaptive_approximation<T0, x_t, y_t, T1, T2>(data.begin(), data.end(), error_bound);
    AA.check_partitions(data.begin(), data.end());

    auto uncompressed_bit_size = data.size() * sizeof(y_t) * 8;
    auto compressed_bit_size = AA.size_in_bits();
    auto cr = (long double) (compressed_bit_size) / (long double) (uncompressed_bit_size);

    out << "AA," << fn << "," << error_bound << "," << uncompressed_bit_size << "," << data.size() << ","
        << compressed_bit_size << "," << cr << "," << range << std::endl;
}

int main(int argc, char *argv[]) {
    auto path = std::string(argv[1]);
    //lossy_compression<27, double, float, double>(path + std::string("uk.bin"), std::cout);
    //lossy_compression<23, double, float, double>(path + std::string("IR-temp.bin"), std::cout);
    lossy_compression<9, double, float, float>(path + std::string("uk.bin"), std::cout, true);
    pla_compression<9, double, float, float>(path + std::string("uk.bin"), std::cout, true);
    //adaptive_approximation_full<1800, double, float, float>(path + std::string("wind-dir.bin"), std::cout, true);

}