#include "gravity_lab/dense_policy.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void close(double actual, double expected, const std::string& message) {
    check(std::abs(actual - expected) < 1e-12, message);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("expected fixture and round-trip paths");
        const auto policy = gravity_lab::DenseQPolicy::load(argv[1]);
        check(policy.environment_id() == "fixture-v1", "environment ID is loaded");
        check(policy.observation_size() == 3, "input size is loaded");
        check(policy.action_count() == 2, "output size is loaded");
        const std::array<double, 3> observation{1.0, 2.0, -1.0};
        const auto values = policy.evaluate(observation);
        close(values[0], 4.0, "output-major dense inference computes Q[0]");
        close(values[1], -4.0, "output-major dense inference computes Q[1]");
        check(policy.action(observation) == 0, "argmax selects the highest Q value");

        gravity_lab::DenseLayer tied_layer{
            1, 3, gravity_lab::Activation::Linear, {0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
        gravity_lab::DenseQPolicy tied_policy("fixture-v1", {1.0}, {0.0}, {tied_layer});
        check(tied_policy.action(std::array<double, 1>{42.0}) == 0,
              "argmax ties select the first action deterministically");

        policy.save(argv[2]);
        const auto loaded = gravity_lab::DenseQPolicy::load(argv[2]);
        check(loaded.evaluate(observation) == values, "C++ checkpoint round trip preserves inference");

        bool dimension_rejected = false;
        try { (void)policy.evaluate(std::array<double, 2>{1.0, 2.0}); }
        catch (const std::invalid_argument&) { dimension_rejected = true; }
        check(dimension_rejected, "wrong observation dimension is rejected");

        bool nonfinite_rejected = false;
        try { (void)policy.evaluate(std::array<double, 3>{1.0, std::numeric_limits<double>::infinity(), 2.0}); }
        catch (const std::invalid_argument&) { nonfinite_rejected = true; }
        check(nonfinite_rejected, "non-finite observation is rejected");

        if (failures == 0) std::cout << "all dense-policy tests passed\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "test setup failed: " << error.what() << '\n';
        return 2;
    }
}
