#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace gravity_lab {

enum class Activation {
    Relu,
    Tanh,
    Linear,
};

struct DenseLayer {
    std::size_t input_size{};
    std::size_t output_size{};
    Activation activation{Activation::Linear};
    // Output-major: weights[output * input_size + input].
    std::vector<double> weights;
    std::vector<double> biases;
};

class DenseQPolicy {
public:
    DenseQPolicy(
        std::string environment_id,
        std::vector<double> input_scale,
        std::vector<double> input_bias,
        std::vector<DenseLayer> layers
    );

    static DenseQPolicy load(const std::filesystem::path& path);
    void save(const std::filesystem::path& path) const;

    [[nodiscard]] std::vector<double> evaluate(std::span<const double> observation) const;
    [[nodiscard]] std::size_t action(std::span<const double> observation) const;
    [[nodiscard]] const std::string& environment_id() const noexcept;
    [[nodiscard]] std::size_t observation_size() const noexcept;
    [[nodiscard]] std::size_t action_count() const noexcept;
    [[nodiscard]] const std::vector<double>& input_scale() const noexcept;
    [[nodiscard]] const std::vector<double>& input_bias() const noexcept;
    [[nodiscard]] const std::vector<DenseLayer>& layers() const noexcept;

private:
    std::string environment_id_;
    std::vector<double> input_scale_;
    std::vector<double> input_bias_;
    std::vector<DenseLayer> layers_;
};

}  // namespace gravity_lab
