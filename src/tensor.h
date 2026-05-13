
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
    bool requires_grad() const;
    void set_requires_grad(bool enabled);

    // Raw data
    double *data();
    const double* data() const;
    double& operator[](size_t ix);
    const double& operator[](size_t ix) const;

    // Tensor ops
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose() const;
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
    void backward();
    void zero_grad();
    Tensor grad() const;

private:
    Shape shape_;
    std::vector<double> data_;

    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
    std::vector<std::shared_ptr<Tensor>> parents_;
    std::function<void()> backward_fn_;
};

size_t numel(const Tensor::Shape& shape);


