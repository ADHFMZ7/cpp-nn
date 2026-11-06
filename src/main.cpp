#include "tensor.h"

#include <vector>


int main() {

    // Broadcasting test
    Tensor tensor {{2, 3}, 0.0};
    Tensor tensor2 {{3}, {1.0, 2.0, 3.0}};

    Tensor tensor3 = tensor + tensor2;

    tensor3.print();

    // Slicing test
    Tensor tensor4 = tensor3.slice(1, 0, 3, 2);
    tensor4.print();
}
