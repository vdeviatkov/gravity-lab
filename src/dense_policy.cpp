#include "gravity_lab/dense_policy.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace gravity_lab {
namespace {

constexpr std::size_t kMaximumLayers = 32;
constexpr std::size_t kMaximumDimension = 65'536;
constexpr std::size_t kMaximumParameters = 100'000'000;

std::string activation_name(Activation activation) {
    switch (activation) {
        case Activation::Relu: return "relu";
        case Activation::Tanh: return "tanh";
        case Activation::Linear: return "linear";
    }
    throw std::logic_error("unknown dense-policy activation");
}

Activation parse_activation(const std::string& value) {
    if (value == "relu") return Activation::Relu;
    if (value == "tanh") return Activation::Tanh;
    if (value == "linear") return Activation::Linear;
    throw std::runtime_error("unsupported dense-policy activation: " + value);
}

void expect(std::istream& input, const char* expected) {
    std::string actual;
    if (!(input >> actual) || actual != expected) {
        throw std::runtime_error("malformed dense policy: expected " + std::string(expected));
    }
}

void validate_finite(const std::vector<double>& values, const char* field) {
    if (std::any_of(values.begin(), values.end(), [](double value) { return !std::isfinite(value); })) {
        throw std::invalid_argument(std::string(field) + " contains a non-finite value");
    }
}

std::size_t checked_parameters(std::size_t input_size, std::size_t output_size) {
    if (input_size == 0 || output_size == 0 || input_size > kMaximumDimension || output_size > kMaximumDimension ||
        input_size > kMaximumParameters / output_size) {
        throw std::runtime_error("dense-policy layer dimensions are invalid or too large");
    }
    return input_size * output_size;
}

void write_values(std::ostream& output, const char* label, const std::vector<double>& values) {
    output << label;
    for (double value : values) output << ' ' << value;
    output << '\n';
}

double activate(double value, Activation activation) {
    switch (activation) {
        case Activation::Relu: return std::max(0.0, value);
        case Activation::Tanh: return std::tanh(value);
        case Activation::Linear: return value;
    }
    throw std::logic_error("unknown dense-policy activation");
}

}  // namespace

DenseQPolicy::DenseQPolicy(
    std::string environment_id,
    std::vector<double> input_scale,
    std::vector<double> input_bias,
    std::vector<DenseLayer> layers
) : environment_id_(std::move(environment_id)), input_scale_(std::move(input_scale)),
    input_bias_(std::move(input_bias)), layers_(std::move(layers)) {
    if (environment_id_.empty()) throw std::invalid_argument("environment_id must not be empty");
    if (input_scale_.empty() || input_scale_.size() > kMaximumDimension) {
        throw std::invalid_argument("dense-policy observation size is invalid");
    }
    if (input_bias_.size() != input_scale_.size()) {
        throw std::invalid_argument("input scale and bias dimensions differ");
    }
    if (layers_.empty() || layers_.size() > kMaximumLayers) {
        throw std::invalid_argument("dense policy must contain between 1 and 32 layers");
    }
    validate_finite(input_scale_, "input_scale");
    validate_finite(input_bias_, "input_bias");

    std::size_t expected_input = input_scale_.size();
    std::size_t total_parameters = 0;
    for (const auto& layer : layers_) {
        const std::size_t weight_count = checked_parameters(layer.input_size, layer.output_size);
        if (layer.input_size != expected_input) throw std::invalid_argument("dense-policy layers do not connect");
        if (layer.weights.size() != weight_count || layer.biases.size() != layer.output_size) {
            throw std::invalid_argument("dense-policy parameter dimensions do not match layer dimensions");
        }
        if (weight_count > kMaximumParameters - total_parameters) {
            throw std::invalid_argument("dense policy has too many parameters");
        }
        total_parameters += weight_count;
        validate_finite(layer.weights, "weights");
        validate_finite(layer.biases, "biases");
        expected_input = layer.output_size;
    }
}

DenseQPolicy DenseQPolicy::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read dense policy: " + path.string());
    std::string format;
    std::getline(input, format);
    if (format != "gravity-lab-dense-q-policy-v1") throw std::runtime_error("unsupported dense-policy format");

    expect(input, "environment");
    std::string environment_id;
    if (!(input >> std::quoted(environment_id))) throw std::runtime_error("malformed dense-policy environment");
    expect(input, "input_size");
    std::size_t input_size{};
    if (!(input >> input_size) || input_size == 0 || input_size > kMaximumDimension) {
        throw std::runtime_error("invalid dense-policy input size");
    }

    std::vector<double> input_scale(input_size);
    std::vector<double> input_bias(input_size);
    expect(input, "input_scale");
    for (double& value : input_scale) {
        if (!(input >> value)) throw std::runtime_error("malformed dense-policy input scale");
    }
    expect(input, "input_bias");
    for (double& value : input_bias) {
        if (!(input >> value)) throw std::runtime_error("malformed dense-policy input bias");
    }

    expect(input, "layers");
    std::size_t layer_count{};
    if (!(input >> layer_count) || layer_count == 0 || layer_count > kMaximumLayers) {
        throw std::runtime_error("invalid dense-policy layer count");
    }
    std::vector<DenseLayer> layers;
    layers.reserve(layer_count);
    std::size_t total_parameters = 0;
    for (std::size_t i = 0; i < layer_count; ++i) {
        expect(input, "layer");
        DenseLayer layer;
        std::string activation;
        if (!(input >> layer.input_size >> layer.output_size >> activation)) {
            throw std::runtime_error("malformed dense-policy layer");
        }
        const std::size_t weight_count = checked_parameters(layer.input_size, layer.output_size);
        if (weight_count > kMaximumParameters - total_parameters) {
            throw std::runtime_error("dense policy has too many parameters");
        }
        total_parameters += weight_count;
        layer.activation = parse_activation(activation);
        layer.biases.resize(layer.output_size);
        layer.weights.resize(weight_count);
        expect(input, "bias");
        for (double& value : layer.biases) {
            if (!(input >> value)) throw std::runtime_error("malformed dense-policy bias");
        }
        expect(input, "weights");
        for (double& value : layer.weights) {
            if (!(input >> value)) throw std::runtime_error("malformed dense-policy weights");
        }
        layers.push_back(std::move(layer));
    }
    expect(input, "end");
    std::string trailing;
    if (input >> trailing) throw std::runtime_error("unexpected data after dense-policy end marker");
    return DenseQPolicy(std::move(environment_id), std::move(input_scale), std::move(input_bias), std::move(layers));
}

void DenseQPolicy::save(const std::filesystem::path& path) const {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary);
    if (!output) throw std::runtime_error("cannot write dense policy: " + temporary.string());
    output << "gravity-lab-dense-q-policy-v1\n" << std::setprecision(17);
    output << "environment " << std::quoted(environment_id_) << '\n';
    output << "input_size " << observation_size() << '\n';
    write_values(output, "input_scale", input_scale_);
    write_values(output, "input_bias", input_bias_);
    output << "layers " << layers_.size() << '\n';
    for (const auto& layer : layers_) {
        output << "layer " << layer.input_size << ' ' << layer.output_size << ' '
               << activation_name(layer.activation) << '\n';
        write_values(output, "bias", layer.biases);
        write_values(output, "weights", layer.weights);
    }
    output << "end\n";
    output.close();
    if (!output) throw std::runtime_error("failed while writing dense policy: " + temporary.string());
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
        if (error) throw std::runtime_error("cannot replace dense policy: " + error.message());
    }
}

std::vector<double> DenseQPolicy::evaluate(std::span<const double> observation) const {
    if (observation.size() != observation_size()) throw std::invalid_argument("policy observation dimension mismatch");
    std::vector<double> values(observation.size());
    for (std::size_t i = 0; i < observation.size(); ++i) {
        if (!std::isfinite(observation[i])) throw std::invalid_argument("policy observation contains a non-finite value");
        values[i] = observation[i] * input_scale_[i] + input_bias_[i];
    }
    for (const auto& layer : layers_) {
        std::vector<double> next(layer.output_size);
        for (std::size_t output = 0; output < layer.output_size; ++output) {
            double value = layer.biases[output];
            const std::size_t offset = output * layer.input_size;
            for (std::size_t input = 0; input < layer.input_size; ++input) {
                value += layer.weights[offset + input] * values[input];
            }
            next[output] = activate(value, layer.activation);
            if (!std::isfinite(next[output])) throw std::runtime_error("dense-policy inference produced a non-finite value");
        }
        values = std::move(next);
    }
    return values;
}

std::size_t DenseQPolicy::action(std::span<const double> observation) const {
    const auto values = evaluate(observation);
    return static_cast<std::size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

const std::string& DenseQPolicy::environment_id() const noexcept { return environment_id_; }
std::size_t DenseQPolicy::observation_size() const noexcept { return input_scale_.size(); }
std::size_t DenseQPolicy::action_count() const noexcept { return layers_.back().output_size; }
const std::vector<double>& DenseQPolicy::input_scale() const noexcept { return input_scale_; }
const std::vector<double>& DenseQPolicy::input_bias() const noexcept { return input_bias_; }
const std::vector<DenseLayer>& DenseQPolicy::layers() const noexcept { return layers_; }

}  // namespace gravity_lab
