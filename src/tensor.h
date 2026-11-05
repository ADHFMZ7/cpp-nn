#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <initializer_list>

class Tensor {

public:

    using Shape = std::vector<size_t>;

    // Constructors
    Tensor();
    Tensor(const Shape& shape, double value);
    Tensor(const Shape& shape, const std::vector<double>& data);

    // Factory helpers
    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor randn(const Shape& shape);

    // Metadata
    const Shape& shape() const;
    size_t ndim() const;
    size_t size() const;
    // bool requires_grad() const;
    // void set_requires_grad(bool enabled);

    // Raw data
    double *data();
    const double* data() const;
    double& operator[](size_t ix);
    const double& operator[](size_t ix) const;
    double& operator[](std::initializer_list<size_t> ix);
    const double& operator[](std::initializer_list<size_t> ix) const;
    
    double item() const;

    // Indexing
    double& operator[](Shape& ix);
    const double& operator[](Shape& ix) const;

    // Tensor ops
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(size_t dim1, size_t dim2) const;
    Tensor sum(int axis = -1) const;
    Tensor mean(int axis = -1) const;
    Tensor relu() const;

    // Arithmetic
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor matmul(const Tensor& other) const;

    // Autograd
    // void backward();
    // void zero_grad();
    // Tensor grad() const;

    void print() const;

private:
    Shape shape_;
    Shape stride_;
    std::vector<double> data_;

    // bool requires_grad_ = false;
    // std::shared_ptr<Tensor> grad_;
    // std::vector<std::shared_ptr<Tensor>> parents_;
    // std::function<void()> backward_fn_;
};



size_t numel(const Tensor::Shape& shape);

Tensor::Shape flat_to_idx(size_t flat_idx, const Tensor::Shape& strides);
