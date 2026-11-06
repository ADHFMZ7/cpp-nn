#include "util.h"
#include "tensor.h"
#include <nlohmann/json.hpp>

#include <map>
#include <fstream>


std::map<std::string, Tensor> load_safetensors(const std::string& filename) {

    std::map<std::string, Tensor> tensors;

    // load 8 byte header
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    char header[8];
    file.read(header, 8);

    uint64_t json_length = *reinterpret_cast<uint64_t*>(header);

    file.seekg(8, std::ios::beg);

    std::string json_header(json_length, '\0');
    file.read(&json_header[0], json_length);

    nlohmann::json j = nlohmann::json::parse(json_header);

    size_t offset = 8 + json_length;
    size_t file_size = std::filesystem::file_size(filename);

    auto storage = Storage::from_file(filename, offset, file_size - offset, Device::CPU);

    for (auto& [key, value] : j.items()) {

        if (!value.contains("shape")) {
            continue;
        }

        size_t data_offset = value["data_offsets"][0].get<size_t>();

        // TODO: Handle different data types: value["dtype"]

        Tensor tensor {value["shape"], storage, data_offset / sizeof(double)};

        tensors[key] = tensor;
    }

    // cleanup
    file.close();
    
    return tensors;
}
