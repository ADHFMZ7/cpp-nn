#include "tensor.h"
#include <functional>
#include <numeric>
#include <random>
#include <iostream>
#include <ranges>
#include <sstream>
#include <utility>

namespace {

std::string shape_to_string(const Tensor::Shape& shape);

} // namespace

// TODO: Include strided layout

Tensor::Tensor()
{}

Tensor::Tensor(const Shape& shape, double value) 
    : shape_ {shape}
    , data_ (numel(shape), value)
{

    size_t temp = numel(shape_);
    
    for (const size_t& dim : shape_) {
        temp /= dim;
        stride_.push_back(temp);
    }

}

Tensor::Tensor(const Shape& shape, const std::vector<double>& data) 
    : shape_ {shape}
    , data_  {data} 
{
    // TODO: Make sure data size matches passed shape
    if (data_.size() != numel(shape_)) {
        throw std::invalid_argument("Data size does not match shape");
    }


}

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

    for (auto& x: tensor.data_) {
        x = d(gen);
    }

    tensor.shape_ = shape;
    return tensor;

}


const Tensor::Shape& Tensor::shape() const { return shape_; }

size_t Tensor::ndim() const { return shape_.size(); }

size_t Tensor::size() const { return numel(shape_); }

// bool requires_grad() const;
// void set_requires_grad(bool enabled);



double* Tensor::data() { return data_.data(); }
const double* Tensor::data() const { return data_.data(); }

double& Tensor::operator[](size_t ix) {
    return data_[ix];
}

const double& Tensor::operator[](size_t ix) const {
    return data_[ix];
}

double& Tensor::operator[](std::initializer_list<size_t> ix) {
    Tensor::Shape idx(ix);
    return (*this)[idx];
}

const double& Tensor::operator[](std::initializer_list<size_t> ix) const {
    Tensor::Shape idx(ix);
    return (*this)[idx];
}

double& Tensor::operator[](Tensor::Shape& ix) {

    size_t offset {0};

    for (auto const& [i, stride] : std::views::zip(ix, stride_)) {
        offset += i * stride; 
    }

    return data_[offset];
}

const double& Tensor::operator[](Tensor::Shape& ix) const {

    size_t offset {0};

    for (auto const& [i, stride] : std::views::zip(ix, stride_)) {
        offset += i * stride; 
    }

    return data_[offset];
}

double Tensor::item() const {

    if (data_.size() != 1) {
        throw std::invalid_argument("Tensor does not contain a single element");
    }

    return data_[0];
}


// Tensor ops
Tensor Tensor::reshape(const Shape& new_shape) const {
    
    if (numel(shape_) != numel(new_shape)) {
        throw std::invalid_argument("Incompatible tensor shapes: cannot reshape " + shape_to_string(shape_) + " into " + shape_to_string(new_shape));
    }

    return Tensor {new_shape, data_};
}

Tensor Tensor::transpose(size_t dim1, size_t dim2) const {
    
    Tensor tensor {*this};

    std::swap(tensor.shape_[dim1], tensor.shape_[dim2]);
    std::swap(tensor.stride_[dim1], tensor.stride_[dim2]);

    return tensor;
}

Tensor Tensor::sum(int axis) const {

    Tensor::Shape new_shape {shape_};
    new_shape[axis] = 1;

    Tensor tensor {new_shape, 0};
    
    Tensor::Shape ix; 

    for (size_t flat_ix = 0; flat_ix < data_.size(); ++flat_ix) {
        ix = flat_to_idx(flat_ix, stride_);

        // Squash axis into shape 1
        ix[axis] = 0;
        tensor[ix] += (*this)[ix];
    }

    return tensor;
}

Tensor Tensor::mean(int axis) const {
    // TODO: Implement keepdims for these operations
    
    Tensor tensor = this->sum(axis);

    return tensor * (static_cast<double>(shape_[axis]) /numel(shape_));

}

Tensor Tensor::relu() const {
    Tensor t {*this};

    for (auto &x : t.data_) {
        x = std::max(x, 0.0);
    }

    return t;
}


// Arithmetic
Tensor Tensor::operator+(const Tensor& other) const {
    if (numel(shape_) != numel(other.shape())) {
        throw std::invalid_argument("Incompatible tensor shapes: cannot add " + shape_to_string(shape_) + " to " + shape_to_string(other.shape()));
    }

    Tensor t {*this};

    for (int ix = 0; ix < numel(t.shape()); ix++) {
        t.data_[ix] += other[ix];
    }

    return t;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (numel(shape_) != numel(other.shape())) {
        throw std::invalid_argument("Incompatible tensor shapes: cannot subtract " + shape_to_string(other.shape()) + " from " + shape_to_string(shape_)); 
    }

    Tensor t {*this};

    for (int ix = 0; ix < numel(t.shape()); ix++) {
        t.data_[ix] -= other[ix];
    }

    return t;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (numel(shape_) != numel(other.shape())) {
        throw std::invalid_argument("Incompatible tensor shapes: cannot multiply " + shape_to_string(shape_) + " with " + shape_to_string(other.shape()));
    }

    Tensor t {*this};

    for (int ix = 0; ix < numel(t.shape()); ix++) {
        t.data_[ix] *= other[ix];
    }

    return t;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (numel(shape_) != numel(other.shape())) {
        throw std::invalid_argument("Incompatible tensor shapes: cannot divide " + shape_to_string(shape_) + " by " + shape_to_string(other.shape()));
    }

    Tensor t {*this};

    for (int ix = 0; ix < numel(t.shape()); ix++) {
        // TODO: Check for div by zero
        t.data_[ix] /= other[ix];
    }
    return t;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor t {*this};

    for (auto& x : t.data_) {
        x *= scalar;
    }

    return t;
}

Tensor Tensor::matmul(const Tensor& other) const {

    // Compute the output shape

    // If both are 1D, return a scalar dot product
    if (shape_.size() == 1 && other.shape_.size() == 1) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("Incompatible tensor shapes: cannot multiply " + shape_to_string(shape_) + " with " + shape_to_string(other.shape()));
        }
        return Tensor({1}, std::vector<double>{std::inner_product(data_.begin(), data_.end(), other.data_.begin(), 0.0)});
    }

    // If one is 1D and the other is 2D, treat the 1D tensor as a row or column vector
    if (shape_.size() == 1 && other.shape_.size() == 2) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("Incompatible tensor shapes: cannot multiply " + shape_to_string(shape_) + " with " + shape_to_string(other.shape()));
        }
        Tensor result({other.shape_[1]}, std::vector<double>(other.shape_[1], 0.0));
        for (size_t j = 0; j < other.shape_[1]; ++j) {
            for (size_t i = 0; i < shape_[0]; ++i) {
                result[j] += (*this)[i] * other[{i, j}];
            }
        }
        return result;
    }

    // If one is 2D and the other is 1D, treat the 1D tensor as a row or column vector
    if (shape_.size() == 2 && other.shape_.size() == 1) {
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Incompatible tensor shapes: cannot multiply " + shape_to_string(shape_) + " with " + shape_to_string(other.shape()));
        }   

        Tensor result({shape_[0]}, std::vector<double>(shape_[0], 0.0));
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result[i] += (*this)[{i, j}] * other[j];
            }
        }
        return result;
    }

    // If both are 2D, perform standard matrix multiplication
    if (shape_.size() == 2 && other.shape_.size() == 2) {
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Incompatible tensor shapes: cannot multiply " + shape_to_string(shape_) + " with " + shape_to_string(other.shape()));
        }

        Tensor result({shape_[0], other.shape_[1]}, std::vector<double>(shape_[0] * other.shape_[1], 0.0));
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < other.shape_[1]; ++j) {
                for (size_t k = 0; k < shape_[1]; ++k) {
                    result[{i, j}] += (*this)[{i, k}] * other[{k, j}];
                }
            }
        }
        return result;
    }


    // If both are ND, perform batched matrix multiplication (not implemented yet)

    throw std::invalid_argument("matmul: Incompatible tensor shapes: cannot multiply " + shape_to_string(shape_) + " with " + shape_to_string(other.shape()));
}

void Tensor::print() const {
    if (shape_.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    if (shape_.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < shape_[0]; ++i) {
            if (i != 0) {
                std::cout << ", ";
            }
            std::cout << data_[i];
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
            std::cout << data_[idx];
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
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
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

} // namespace