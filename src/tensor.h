#pragma once

#include "storage.h"

#include <concepts>
#include <vector>
#include <span>
#include <string>
#include <memory>
#include <functional>
#include <ostream>

class Tensor {

public:

    using Shape = std::vector<size_t>;

    // Constructors
    Tensor();
    Tensor(const Shape& shape, double value);
    Tensor(const Shape& shape, const std::vector<double>& values);

    // Wraps existing storage without copying (views share the same buffer)
    Tensor(const Shape& shape, std::shared_ptr<Storage> storage, size_t offset = 0);

    // Factory helpers
    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor randn(const Shape& shape);

    // Metadata
    const Shape& shape() const;
    size_t ndim() const;
    size_t size() const;
    bool is_contiguous() const;
    // bool requires_grad() const;
    // void set_requires_grad(bool enabled);

    // Raw data, already offset to this tensor's first element
    double *data();
    const double* data() const;

    double item() const;

    // Indexing: t[i, j, ...] for fixed indices
    template <class Self, std::convertible_to<size_t>... I>
        requires (sizeof...(I) > 0)
    auto& operator[](this Self&& self, I... ix) {
        const size_t idx[] {static_cast<size_t>(ix)...};
        return self.data()[self.offset_of(idx)];
    }

    // Indexing: t[ix] for indices computed at runtime
    template <class Self>
    auto& operator[](this Self&& self, const Shape& ix) {
        return self.data()[self.offset_of(ix)];
    }

    // Copies
    Tensor clone() const;
    Tensor contiguous() const;

    // Tensor ops
    Tensor reshape(const Shape& new_shape) const;
    Tensor expand(const Shape& new_shape) const;
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
    size_t offset_ = 0;
    Shape shape_;
    Shape stride_;

    // Shared pointer allows for sharing data between tensors (e.g., views, slices)
    std::shared_ptr<Storage> storage_;

    size_t offset_of(std::span<const size_t> ix) const;

    template <class Op>
    Tensor unary_op(Op op) const;

    template <class Op>
    Tensor binary_op(const Tensor& other, Op op, const std::string& name) const;

    // bool requires_grad_ = false;
    // std::shared_ptr<Tensor> grad_;
    // std::vector<std::shared_ptr<Tensor>> parents_;
    // std::function<void()> backward_fn_;
};

std::ostream& operator<<(std::ostream& out, const Tensor::Shape& shape);


size_t numel(const Tensor::Shape& shape);

Tensor::Shape flat_to_idx(size_t flat_idx, const Tensor::Shape& strides);
