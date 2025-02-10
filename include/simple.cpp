/*
 * This example shows how to index and query a vector of random integers with the PGM-index.
 * Compile with:
 *   g++ simple.cpp -std=c++17 -I../include -o simple
 * Run with:
 *   ./simple
 */

#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "piecewise_nonlinear_model.hpp"
#include "PGM-Index.hpp"
#include <random>
#include <chrono>

int main() {
    using data_type = int64_t;

    std::random_device rd;   // a seed source for the random number engine
    std::mt19937 gen(123);  // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<data_type> distrib(0, (1L << 62));

    auto size = 10000000;
    std::vector<data_type> data(size);
    for (auto &d : data) {
        d = distrib(gen);
    }
    //data.push_back(2589341134886);
    auto value_to_find = data[6873781];
    std::cout << "Value to find: " << value_to_find << std::endl;
    /*
    auto size = 10000000;
    std::vector<data_type> data(size);
    std::generate(data.begin(), data.end(), std::rand);
     */
    std::sort(data.begin(), data.end());

    // POSITION IS 6873781

    using PlaType = neats::PiecewiseOptimalModel<data_type, uint32_t, double, float, double, true>;
    // Construct the PGM-index
    data_type epsilon = 8; // space-time trade-off parameter
    PlaType model(epsilon);

    auto t1 = std::chrono::high_resolution_clock::now();
    //pgm::PGMIndex<data_type, epsilon, 1, float> index(data);
    //auto models = model.make_fragmentation(data.begin(), data.end());
    pgm::PGMIndex<data_type, 8, 4> index(data);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto speed = ((data.size() * sizeof(data_type)) / 1e6) / (time / 1e3);

    std::cout << "Height: " << index.height() << " First level size: " << index.segments_count() << std::endl;
    std::cout << "NUM SEGMENTS TOTAL: " << index.num_segments() << std::endl;
    std::cout << "Construction time: " << time << " ms, speed: " << speed << " MB/s" << std::endl;

    // Query the PGM-index
    auto q = value_to_find;
    auto range = index.search(q);
    auto lo = data.begin() + range.lo;
    auto hi = data.begin() + range.hi;


    std::cout << "Value found: "<< *std::lower_bound(lo, hi, q);
    return 0;
}