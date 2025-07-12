# NeaTS: Learned Compression of Nonlinear Time Series With Random Access

<p align="center">
  <strong>NeaTS</strong> is a learned compressor for integer time series that uniquely combines superior compression ratios, surpassing state-of-the-art methods, with significantly faster decompression, range queries performance, and random access speeds.
</p>

---

## 🔧 Features

- 🧩 Piecewise nonlinear approximations
- 🚀 Efficient random access and range queries without full decompression
- 🧠 SIMD-accelerated decompression (AVX2/AVX-512)
- 📉 Lossless and lossy modes

---

## 📦 Requirements

- C++23 compatible compiler (e.g., GCC ≥ 13 or Clang ≥ 16)
- CMake ≥ 3.22
- [SDSL-lite](https://github.com/simongog/sdsl-lite)
- [Squash 0.7+](https://quixdb.github.io/squash/) (**required for benchmarking**)

---

## ⚙️ Build the project

```
git clone https://github.com/and-gue/NeaTS.git
cd NeaTS
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## 🚀 Executables

| Binary             | Description                                                    |
| ------------------ | -------------------------------------------------------------- |
| `DecompressorSIMD` | Runs NeaTS for compression ratio, decompression, random access |
| `NeaTSL`           | Evaluates lossy compression and compares with PLA/AA models    |
| `Benchmark`        | Runs benchmarking suite (**requires Squash**)                  |

---

## 🧪 NeaTS Usage
```
./DecompressorSIMD path/to/data.bin <bpc>
```
- data.bin — Binary file of 64-bit signed integers
- `<bpc>` — Maximum residual (size in bits)

---

## 📚 Citation

If you use NeaTS for research, please cite:
```
@inproceedings{guerra2025neats,
  author    = {Guerra, Alessandro and Vinciguerra, Giuseppe and Boffa, Alessandro and Ferragina, Paolo},
  title     = {Learned Compression of Nonlinear Time Series with Random Access},
  booktitle = {2025 IEEE 41st International Conference on Data Engineering (ICDE)},
  year      = {2025},
  pages     = {1579--1592},
  doi       = {10.1109/ICDE65448.2025.00122}
}
```

## License 🪪

This project is released for academic purposes under the terms of the GNU General Public License v3.0.
