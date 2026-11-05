#pragma once

#include "tensor.h"
#include <map>
#include <fstream>
#include <iostream>

std::map<std::string, Tensor> load_safetensors(const std::string& filename);
