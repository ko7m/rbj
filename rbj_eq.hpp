#pragma once

#include <cmath>
#include <vector>
#include <string>

namespace rbj {

struct Coeffs {
    double b0, b1, b2;
    double a1, a2;  // a0 is normalized to 1
};

// ---------- helpers ----------

inline double A_from_db(double dB) {
    return std::pow(10.0, dB / 40.0);
}

inline double omega0(double f0, double Fs) {
    static const double pi = 3.14159265358979323846;
    return 2.0 * pi * f0 / Fs;
}

inline double alpha_from_Q(double w0, double Q) {
    return std::sin(w0) / (2.0 * Q);
}

inline Coeffs normalize(double b0, double b1, double b2,
                        double a0, double a1, double a2) {
    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= a0;
    a2 /= a0;
    return Coeffs{b0, b1, b2, a1, a2};
}

// ---------- RBJ coefficient designers (Q-based) ----------

inline Coeffs design_lpf(double f0, double Q, double Fs) {
    double w0    = omega0(f0, Fs);
    double cosw0 = std::cos(w0);
    double alpha = alpha_from_Q(w0, Q);

    double b0 = (1.0 - cosw0) * 0.5;
    double b1 = 1.0 - cosw0;
    double b2 = (1.0 - cosw0) * 0.5;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha;

    return normalize(b0, b1, b2, a0, a1, a2);
}

inline Coeffs design_hpf(double f0, double Q, double Fs) {
    double w0    = omega0(f0, Fs);
    double cosw0 = std::cos(w0);
    double alpha = alpha_from_Q(w0, Q);

    double b0 = (1.0 + cosw0) * 0.5;
    double b1 = -(1.0 + cosw0);
    double b2 = (1.0 + cosw0) * 0.5;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha;

    return normalize(b0, b1, b2, a0, a1, a2);
}

inline Coeffs design_bpf_peak(double f0, double Q, double Fs) {
    double w0    = omega0(f0, Fs);
    double cosw0 = std::cos(w0);
    double alpha = alpha_from_Q(w0, Q);

    double b0 = alpha;
    double b1 = 0.0;
    double b2 = -alpha;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha;

    return normalize(b0, b1, b2, a0, a1, a2);
}

inline Coeffs design_notch(double f0, double Q, double Fs) {
    double w0    = omega0(f0, Fs);
    double cosw0 = std::cos(w0);
    double alpha = alpha_from_Q(w0, Q);

    double b0 = 1.0;
    double b1 = -2.0 * cosw0;
    double b2 = 1.0;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha;

    return normalize(b0, b1, b2, a0, a1, a2);
}

inline Coeffs design_apf(double f0, double Q, double Fs) {
    double w0    = omega0(f0, Fs);
    double cosw0 = std::cos(w0);
    double alpha = alpha_from_Q(w0, Q);

    double b0 = 1.0 - alpha;
    double b1 = -2.0 * cosw0;
    double b2 = 1.0 + alpha;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha;

    return normalize(b0, b1, b2, a0, a1, a2);
}

inline Coeffs design_peaking(double f0, double Q, double dBgain, double Fs) {
    double A     = A_from_db(dBgain);
    double w0    = omega0(f0, Fs);
    double cosw0 = std::cos(w0);
    double alpha = alpha_from_Q(w0, Q);

    double b0 = 1.0 + alpha * A;
    double b1 = -2.0 * cosw0;
    double b2 = 1.0 - alpha * A;
    double a0 = 1.0 + alpha / A;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha / A;

    return normalize(b0, b1, b2, a0, a1, a2);
}

inline Coeffs design_lowshelf(double f0, double dBgain, double S, double Fs) {
    double A     = A_from_db(dBgain);
    double w0    = omega0(f0, Fs);
    double cosw0 = std::cos(w0);
    double sinw0 = std::sin(w0);

    double alpha = sinw0 / 2.0 *
                   std::sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0);
    double two_sqrtA_alpha = 2.0 * std::sqrt(A) * alpha;

    double b0 = A * ((A + 1.0) - (A - 1.0) * cosw0 + two_sqrtA_alpha);
    double b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0);
    double b2 = A * ((A + 1.0) - (A - 1.0) * cosw0 - two_sqrtA_alpha);
    double a0 = (A + 1.0) + (A - 1.0) * cosw0 + two_sqrtA_alpha;
    double a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosw0);
    double a2 = (A + 1.0) + (A - 1.0) * cosw0 - two_sqrtA_alpha;

    return normalize(b0, b1, b2, a0, a1, a2);
}

inline Coeffs design_highshelf(double f0, double dBgain, double S, double Fs) {
    double A     = A_from_db(dBgain);
    double w0    = omega0(f0, Fs);
    double cosw0 = std::cos(w0);
    double sinw0 = std::sin(w0);

    double alpha = sinw0 / 2.0 *
                   std::sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0);
    double two_sqrtA_alpha = 2.0 * std::sqrt(A) * alpha;

    double b0 = A * ((A + 1.0) + (A - 1.0) * cosw0 + two_sqrtA_alpha);
    double b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0);
    double b2 = A * ((A + 1.0) + (A - 1.0) * cosw0 - two_sqrtA_alpha);
    double a0 = (A + 1.0) - (A - 1.0) * cosw0 + two_sqrtA_alpha;
    double a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw0);
    double a2 = (A + 1.0) - (A - 1.0) * cosw0 - two_sqrtA_alpha;

    return normalize(b0, b1, b2, a0, a1, a2);
}

// ---------- DF2T biquad ----------

class Biquad {
public:
    Biquad()
        : b0(1.0), b1(0.0), b2(0.0),
          a1(0.0), a2(0.0),
          z1(0.0), z2(0.0) {}

    explicit Biquad(const Coeffs& c)
        : b0(c.b0), b1(c.b1), b2(c.b2),
          a1(c.a1), a2(c.a2),
          z1(0.0), z2(0.0) {}

    void set_coeffs(const Coeffs& c) {
        b0 = c.b0; b1 = c.b1; b2 = c.b2;
        a1 = c.a1; a2 = c.a2;
    }

    void reset() {
        z1 = 0.0;
        z2 = 0.0;
    }

    inline double process_sample(double x) {
        double y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }

    void process_block(const double* x, double* y, std::size_t n) {
        double local_z1 = z1;
        double local_z2 = z2;

        for (std::size_t i = 0; i < n; ++i) {
            double xi = x[i];
            double yi = b0 * xi + local_z1;
            local_z1 = b1 * xi - a1 * yi + local_z2;
            local_z2 = b2 * xi - a2 * yi;
            y[i] = yi;
        }

        z1 = local_z1;
        z2 = local_z2;
    }

private:
    double b0, b1, b2;
    double a1, a2;
    double z1, z2;
};

// ---------- EQ bands + chain ----------

struct EQBand {
    std::string kind;  // "lpf", "hpf", "peaking", ...
    double f0;
    double Q;
    double gain_db;
    double S;
    double Fs;
    bool enabled;
    Biquad biquad;

    EQBand(const std::string& kind_,
           double f0_,
           double Q_       = 1.0,
           double gain_db_ = 0.0,
           double S_       = 1.0,
           double Fs_      = 48000.0,
           bool enabled_   = true)
        : kind(kind_), f0(f0_), Q(Q_),
          gain_db(gain_db_), S(S_),
          Fs(Fs_), enabled(enabled_), biquad()
    {
        update_biquad();
    }

    void update(double f0_, double Q_, double gain_db_, double S_) {
        f0 = f0_;
        Q = Q_;
        gain_db = gain_db_;
        S = S_;
        update_biquad();
    }

    void update_biquad() {
        Coeffs c = design();
        biquad.set_coeffs(c);
    }

    Coeffs design() const {
        std::string k = kind;
        // You might want to normalize to lowercase here yourself.
        if (k == "lpf")       return design_lpf(f0, Q, Fs);
        else if (k == "hpf")  return design_hpf(f0, Q, Fs);
        else if (k == "bpf")  return design_bpf_peak(f0, Q, Fs);
        else if (k == "notch")return design_notch(f0, Q, Fs);
        else if (k == "apf")  return design_apf(f0, Q, Fs);
        else if (k == "peaking")
            return design_peaking(f0, Q, gain_db, Fs);
        else if (k == "lowshelf")
            return design_lowshelf(f0, gain_db, S, Fs);
        else if (k == "highshelf")
            return design_highshelf(f0, gain_db, S, Fs);
        else
            throw std::runtime_error("Unknown EQ band kind: " + kind);
    }

    void process_block(const double* x, double* y, std::size_t n) {
        if (!enabled) {
            // in-place copy if x != y, or nothing if x == y
            if (x != y) {
                for (std::size_t i = 0; i < n; ++i) y[i] = x[i];
            }
            return;
        }
        biquad.process_block(x, y, n);
    }
};

class ParametricEQ {
public:
    explicit ParametricEQ(double Fs_ = 48000.0)
        : Fs(Fs_) {}

    EQBand& add_band(const std::string& kind,
                     double f0,
                     double Q       = 1.0,
                     double gain_db = 0.0,
                     double S       = 1.0,
                     bool enabled   = true) {
        bands.emplace_back(kind, f0, Q, gain_db, S, Fs, enabled);
        return bands.back();
    }

    void process_block(double* x, std::size_t n) {
        // in-place processing
        temp.resize(n);
        for (auto& band : bands) {
            band.process_block(x, temp.data(), n);
            // copy back for next stage
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = temp[i];
            }
        }
    }

    void reset() {
        for (auto& band : bands) {
            band.biquad.reset();
        }
    }

private:
    double Fs;
    std::vector<EQBand> bands;
    std::vector<double> temp; // scratch buffer for chaining
};

} // namespace rbj
