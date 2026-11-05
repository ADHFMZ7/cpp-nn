#include "tensor.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <iostream>
#include <ranges>
#include <sstream>
#include <utility>

namespace {

std::string shape_to_string(const Tensor::Shape& shape);
Tensor::Shape contiguous_strides(const Tensor::Shape& shape);

} // namespace


Tensor::Tensor()
{}

Tensor::Tensor(const Shape& shape, double value)
    : Tensor {shape, std::make_shared<Storage>(numel(shape) * sizeof(double), Device::CPU)}
{
    std::fill_n(data(), size(), value);
}

Tensor::Tensor(const Shape& shape, const std::vector<double>& values)
    : Tensor {shape, std::make_shared<Storage>(numel(shape) * sizeof(double), Device::CPU)}
{
    if (values.size() != size()) {
        throw std::invalid_argument("Data size does not match shape");
    }

    std::copy(values.begin(), values.end(), data());
}

Tensor::Tensor(const Shape& shape, std::shared_ptr<Storage> storage, size_t offset)
    : offset_ {offset}
    , shape_ {shape}
    , stride_ {contiguous_strides(shape)}
    , storage_ {std::move(storage)}
{}

Tensor Tensor::zeros(const Tensor::Shape& shape) {
    return Tensor(shape, 0.0);
}

Tensor Tensor::ones(const Tensor::Shape& shape) {
    return Tensor(shape, 1.0);
}

Tensor Tensor::randn(const Tensor::Shape& shape) {

    Tensor tensor{shape, 0.0};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    for (size_t ix = 0; ix < tensor.size(); ix++) {
        tensor.data()[ix] = d(gen);
    }

    return tensor;

}


const Tensor::Shape& Tensor::shape() const { return shape_; }

size_t Tensor::ndim() const { return shape_.size(); }

size_t Tensor::size() const { return numel(shape_); }

bool Tensor::is_contiguous() const { return stride_ == contiguous_strides(shape_); }

// bool requires_grad() const;
// void set_requires_grad(bool enabled);



double* Tensor::data() { return reinterpret_cast<double*>(storage_->data()) + offset_; }
const double* Tensor::data() const { return reinterpret_cast<const double*>(storage_->data()) + offset_; }

size_t Tensor::offset_of(std::span<const size_t> ix) const {

    if (ix.size() != shape_.size()) {
        throw std::invalid_argument("Index size does not match tensor dimensions");
    }

    return std::inner_product(ix.begin(), ix.end(), stride_.begin(), size_t {0});
}

double Tensor::item() const {

    if (size() != 1) {
        throw std::invalid_argument("Tensor does not contain a single element");
    }

    return data()[0];
}


// Copies
Tensor Tensor::clone() const {

    Tensor tensor {shape_, 0.0};
    const Shape strides = contiguous_strides(shape_);

    for (size_t flat_ix = 0; flat_ix < size(); ++flat_ix) {
        tensor.data()[flat_ix] = (*this)[flat_to_idx(flat_ix, strides)];
    }

    return tensor;
}

Tensor Tensor::contiguous() const {
    return is_contiguous() ? *this : clone();
}


// Elementwise helpers: both write into a fresh tensor, so inputs are never modified
template <class Op>
Tensor Tensor::unary_op(Op op) const {

    const Tensor src = contiguous();
    Tensor tensor {shape_, 0.0};

    for (size_t ix = 0; ix < size(); ++ix) {
        tensor.data()[ix] = op(src.data()[ix]);
    }

    return tensor;
}

template <class Op>
Tensor Tensor::binary_op(const Tensor& other, Op op, const std::string& name) const {

    if (shape_ != other.shape_) {
        throw std::invalid_argument("Incompatible tensor shapes: cannot " + name + " " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    }

    const Tensor lhs = contiguous();
    const Tensor rhs = other.contiguous();
    Tensor tensor {shape_, 0.0};

    for (size_t ix = 0; ix < size(); ++ix) {
        tensor.data()[ix] = op(lhs.data()[ix], rhs.data()[ix]);
    }

    return tensor;
}


// Tensor ops
Tensor Tensor::reshape(const Shape& new_shape) const {

    if (numel(shape_) != numel(new_shape)) {
        throw std::invalid_argument("Incompatible tensor shapes: cannot reshape " + shape_to_string(shape_) + " into " + shape_to_string(new_shape));
    }

    // Only a contiguous layout can be reinterpreted in place
    const Tensor src = contiguous();

    return Tensor {new_shape, src.storage_, src.offset_};
}

Tensor Tensor::transpose(size_t dim1, size_t dim2) const {

    Tensor tensor {*this};

    std::swap(tensor.shape_[dim1], tensor.shape_[dim2]);
    std::swap(tensor.stride_[dim1], tensor.stride_[dim2]);

    return tensor;
}

Tensor Tensor::sum(int axis) const {

    const size_t ax = axis < 0 ? axis + ndim() : axis;

    Tensor::Shape new_shape {shape_};
    new_shape[ax] = 1;

    Tensor tensor {new_shape, 0.0};
    const Shape strides = contiguous_strides(shape_);

    for (size_t flat_ix = 0; flat_ix < size(); ++flat_ix) {
        Tensor::Shape ix = flat_to_idx(flat_ix, strides);
        const double x = (*this)[ix];

        // Squash axis into shape 1
        ix[ax] = 0;
        tensor[ix] += x;
    }

    return tensor;
}

Tensor Tensor::mean(int axis) const {
    // TODO: Implement keepdims for these operations

    const size_t ax = axis < 0 ? axis + ndim() : axis;

    return sum(axis) * (1.0 / shape_[ax]);
}

Tensor Tensor::relu() const {
    return unary_op([](double x) { return std::max(x, 0.0); });
}


// Arithmetic
Tensor Tensor::operator+(const Tensor& other) const {
    return binary_op(other, std::plus<>{}, "add");
}

Tensor Tensor::operator-(const Tensor& other) const {
    return binary_op(other, std::minus<>{}, "subtract");
}

Tensor Tensor::operator*(const Tensor& other) const {
    return binary_op(other, std::multiplies<>{}, "multiply");
}

Tensor Tensor::operator/(const Tensor& other) const {
    // TODO: Check for div by zero
    return binary_op(other, std::divides<>{}, "divide");
}

Tensor Tensor::operator*(double scalar) const {
    return unary_op([scalar](double x) { return x * scalar; });
}

Tensor Tensor::matmul(const Tensor& other) const {

    // Batched (ND) matrix multiplication is not implemented yet
    if (ndim() < 1 || ndim() > 2 || other.ndim() < 1 || other.ndim() > 2) {
        throw std::invalid_argument("matmul: Incompatible tensor shapes: cannot multiply " + shape_to_string(shape_) + " with " + shape_to_string(other.shape()));
    }

    // Promote 1D operands to 2D: a row vector on the left, a column vector on the right
    const Tensor lhs = ndim() == 1 ? reshape({1, shape_[0]}) : *this;
    const Tensor rhs = other.ndim() == 1 ? other.reshape({other.shape_[0], 1}) : other;

    const size_t n = lhs.shape_[0];
    const size_t k = lhs.shape_[1];
    const size_t m = rhs.shape_[1];

    if (k != rhs.shape_[0]) {
        throw std::invalid_argument("Incompatible tensor shapes: cannot multiply " + shape_to_string(shape_) + " with " + shape_to_string(other.shape()));
    }

    Tensor result {{n, m}, 0.0};
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t p = 0; p < k; ++p) {
                result[i, j] += lhs[i, p] * rhs[p, j];
            }
        }
    }

    // Drop the dimensions that were added for 1D operands
    Shape out_shape;
    if (ndim() == 2) {
        out_shape.push_back(n);
    }
    if (other.ndim() == 2) {
        out_shape.push_back(m);
    }

    return result.reshape(out_shape.empty() ? Shape {1} : out_shape);
}

void Tensor::print() const {
    if (shape_.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    const Tensor src = contiguous();
    const double* values = src.data();

    if (shape_.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < shape_[0]; ++i) {
            if (i != 0) {
                std::cout << ", ";
            }
            std::cout << values[i];
        }
        std::cout << "]" << std::endl;
        return;
    }

    size_t rows = shape_[0];
    size_t cols = shape_[1];

    std::cout << "[" << std::endl;
    for (size_t r = 0; r < rows; ++r) {
        std::cout << "  [";
        for (size_t c = 0; c < cols; ++c) {
            size_t idx = r * cols + c;
            if (c != 0) {
                std::cout << ", ";
            }
            std::cout << values[idx];
        }
        std::cout << "]";
        if (r + 1 != rows) {
            std::cout << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}


// Autograd
// void Tensor::backward() {
//
// }
//
// void Tensor::zero_grad() {
//
// }
//
// Tensor Tensor::grad() const {
//
// }



Tensor::Shape flat_to_idx(size_t flat_idx, const Tensor::Shape& strides) {
    Tensor::Shape idx(strides.size());

    for (size_t d = 0; d < strides.size(); ++d) {

        idx[d] = flat_idx / strides[d];
        flat_idx %= strides[d];
    }

    return idx;
}



size_t numel(const Tensor::Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t {1}, std::multiplies<size_t>());
}

namespace {

std::string shape_to_string(const Tensor::Shape& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

Tensor::Shape contiguous_strides(const Tensor::Shape& shape) {

    Tensor::Shape strides;
    size_t temp = numel(shape);

    for (const size_t& dim : shape) {
        temp /= dim;
        strides.push_back(temp);
    }

    return strides;
}

} // namespace
