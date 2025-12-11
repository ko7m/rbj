// rbj filter design tests
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "rbj_eq.hpp"  // the header with your C++ port

static rbj::Coeffs design_band_cpp(const std::string& kind,
                                   double f0, double Q,
                                   double gain_db, double S, double Fs)
{
    std::string k = kind;
    // make it lowercase if you want robustness
    // (assuming all-lowercase from Python, you can skip this)
    if (k == "lpf")        return rbj::design_lpf(f0, Q, Fs);
    else if (k == "hpf")   return rbj::design_hpf(f0, Q, Fs);
    else if (k == "bpf")   return rbj::design_bpf_peak(f0, Q, Fs);
    else if (k == "notch") return rbj::design_notch(f0, Q, Fs);
    else if (k == "apf")   return rbj::design_apf(f0, Q, Fs);
    else if (k == "peaking")
        return rbj::design_peaking(f0, Q, gain_db, Fs);
    else if (k == "lowshelf")
        return rbj::design_lowshelf(f0, gain_db, S, Fs);
    else if (k == "highshelf")
        return rbj::design_highshelf(f0, gain_db, S, Fs);
    else
        throw std::runtime_error("Unknown kind: " + kind);
}

int main()
{
    std::ifstream in("rbj_golden.txt");
    if (!in) {
        std::cerr << "Failed to open rbj_golden.txt\n";
        return 1;
    }

    const double coeff_tol   = 1e-12;
    const double impulse_tol = 1e-10;

    int test_index = 0;
    int pass_count = 0;
    int fail_count = 0;

    while (true) {
        std::string line;

        // Skip comments / blank lines until we hit a test header or EOF
        std::streampos pos = in.tellg();
        if (!std::getline(in, line)) {
            break; // EOF
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // We've read the first line of a test in 'line'
        // Put it in a stringstream to parse.
        std::stringstream header(line);

        std::string kind;
        double f0, Q, gain_db, S, Fs;
        int N;

        if (!(header >> kind >> f0 >> Q >> gain_db >> S >> Fs >> N)) {
            std::cerr << "Error parsing header line: " << line << "\n";
            return 1;
        }

        // Next: coeffs line
        std::string coeff_label;
        double b0_ref, b1_ref, b2_ref, a1_ref, a2_ref;

        if (!std::getline(in, line)) {
            std::cerr << "Unexpected EOF reading coeffs line.\n";
            return 1;
        }
        {
            std::stringstream coeffs_ss(line);
            if (!(coeffs_ss >> coeff_label >> b0_ref >> b1_ref >> b2_ref >> a1_ref >> a2_ref)) {
                std::cerr << "Error parsing coeffs line: " << line << "\n";
                return 1;
            }
            if (coeff_label != "coeffs") {
                std::cerr << "Expected 'coeffs' label, got: " << coeff_label << "\n";
                return 1;
            }
        }

        // Next: impulse line
        std::string impulse_label;
        std::vector<double> impulse_ref(N);

        if (!std::getline(in, line)) {
            std::cerr << "Unexpected EOF reading impulse line.\n";
            return 1;
        }
        {
            std::stringstream impulse_ss(line);
            if (!(impulse_ss >> impulse_label)) {
                std::cerr << "Error parsing impulse label.\n";
                return 1;
            }
            if (impulse_label != "impulse") {
                std::cerr << "Expected 'impulse' label, got: " << impulse_label << "\n";
                return 1;
            }
            for (int i = 0; i < N; ++i) {
                if (!(impulse_ss >> impulse_ref[i])) {
                    std::cerr << "Error parsing impulse sample " << i
                              << " in line: " << line << "\n";
                    return 1;
                }
            }
        }

        // You might have a blank line after this; ignore it
        // (if present)
        std::getline(in, line);

        // Now compute C++ values
        rbj::Coeffs c = design_band_cpp(kind, f0, Q, gain_db, S, Fs);

        // Compare coefficients
        auto absdiff = [](double a, double b) {
            return std::fabs(a - b);
        };

        bool coeffs_ok = true;
        if (absdiff(c.b0, b0_ref) > coeff_tol ||
            absdiff(c.b1, b1_ref) > coeff_tol ||
            absdiff(c.b2, b2_ref) > coeff_tol ||
            absdiff(c.a1, a1_ref) > coeff_tol ||
            absdiff(c.a2, a2_ref) > coeff_tol)
        {
            coeffs_ok = false;
        }

        // Compute impulse response via C++ Biquad
        std::vector<double> x(N, 0.0), y(N, 0.0);
        x[0] = 1.0;

        rbj::Biquad biq(c);
        biq.process_block(x.data(), y.data(), static_cast<std::size_t>(N));

        bool impulse_ok = true;
        for (int i = 0; i < N; ++i) {
            if (absdiff(y[i], impulse_ref[i]) > impulse_tol) {
                impulse_ok = false;
                break;
            }
        }

        bool ok = coeffs_ok && impulse_ok;
        std::cout << "Test " << test_index << " (" << kind << " f0=" << f0
                  << " Hz, Q=" << Q << ", gain=" << gain_db << " dB): "
                  << (ok ? "PASS" : "FAIL") << "\n";

        if (!coeffs_ok) {
            std::cout << "  Coeff mismatch:\n"
                      << "    Python: b0=" << b0_ref << " b1=" << b1_ref
                      << " b2=" << b2_ref << " a1=" << a1_ref << " a2=" << a2_ref << "\n"
                      << "    C++:    b0=" << c.b0    << " b1=" << c.b1
                      << " b2=" << c.b2    << " a1=" << c.a1    << " a2=" << c.a2    << "\n";
        }

        if (!impulse_ok) {
            std::cout << "  Impulse mismatch (first few samples):\n";
            for (int i = 0; i < std::min(N, 10); ++i) {
                std::cout << "    i=" << i << " py=" << impulse_ref[i]
                          << " cpp=" << y[i]
                          << " diff=" << absdiff(impulse_ref[i], y[i]) << "\n";
            }
        }

        if (ok) ++pass_count;
        else     ++fail_count;

        ++test_index;
    }

    std::cout << "Summary: " << pass_count << " passed, " << fail_count << " failed.\n";
    return fail_count == 0 ? 0 : 1;
}
