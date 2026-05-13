#include "tensor.h"
#include <functional>
#include <numeric>
#include <random>
#include <iostream>

// TODO: Include strided layout

Tensor::Tensor() 
{}

Tensor::Tensor(const Shape& shape, double value) 
    : shape_ {shape}
    , data_ (numel(shape), value)
{}

Tensor::Tensor(const Shape& shape, const std::vector<double>& data) 
    : shape_ {shape}
    , data_  {data} 
{
    // TODO: Make sure data size matches passed shape
}

Tensor Tensor::zeros(const Tensor::Shape& shape) {
    return Tensor(shape, 0.0);
}

Tensor Tensor::ones(const Tensor::Shape& shape) {
    return Tensor(shape, 1.0);
}

Tensor Tensor::randn(const Tensor::Shape& shape) {

    Tensor tensor{};

    tensor.data_.resize(numel(shape));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    for (auto& x: tensor.data_) {
        x = d(gen);
        std::cout << x << std::endl;
    }

    tensor.shape_ = shape;
    return tensor;

}



const Tensor::Shape& Tensor::shape() const { return shape_; }

size_t Tensor::ndim() const { return shape_.size(); }

size_t Tensor::size() const { return numel(shape_); }

// bool requires_grad() const;
// void set_requires_grad(bool enabled);



// Raw data
double* Tensor::data() { return data_.data(); }
const double* Tensor::data() const { return data_.data(); }

// TODO: More complex indexing
double& Tensor::operator[](size_t ix) {}
const double& Tensor::operator[](size_t ix) const {}







// Tensor ops
Tensor Tensor::reshape(const Shape& new_shape) const {}
Tensor Tensor::transpose() const {}
Tensor Tensor::sum(int axis) const {}
Tensor Tensor::mean(int axis) const {}
Tensor Tensor::relu() const {}

// Arithmetic
Tensor Tensor::operator+(const Tensor& other) const {}
Tensor Tensor::operator-(const Tensor& other) const {}
Tensor Tensor::operator*(const Tensor& other) const {}
Tensor Tensor::operator/(const Tensor& other) const {}
Tensor Tensor::operator*(double scalar) const {}
Tensor Tensor::matmul(const Tensor& other) const {}

// Autograd
void Tensor::backward() {}
void Tensor::zero_grad() {}
Tensor Tensor::grad() const {}

































size_t size(const Tensor::Shape& shape) {

    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

