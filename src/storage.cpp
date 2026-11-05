#include "storage.h"

#include <stdexcept>
#include <utility>
#include <fstream>

StorageDeleter::StorageDeleter(Device device)
	: device_(device) {}

void StorageDeleter::operator()(std::byte* ptr) const {
	if (device_ == Device::CUDA) {
		// cudaFree(ptr);
		throw std::runtime_error("CUDA device type is not supported yet for StorageDeleter");
	} else if (device_ == Device::MPS) {
		// mpsFree(ptr);
		throw std::runtime_error("MPS device type is not supported yet for StorageDeleter");
	} else if (device_ == Device::CPU) {
		delete[] ptr;
		return;
	}
	throw std::runtime_error("Unsupported device type for StorageDeleter");
}

Storage::Storage(size_t size, Device device, std::unique_ptr<std::byte[], StorageDeleter> data)
	: data_(std::move(data)), bytes_(size), device_(device) {}

Storage::Storage(size_t size, Device device)
	: data_(Storage::allocate(size, device), StorageDeleter(device)),
	  bytes_(size),
	  device_(device) {}

std::shared_ptr<Storage> Storage::from_file(const std::string& filename, size_t offset, size_t bytes, Device device) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::byte* data_ptr = Storage::allocate(bytes, device);
    file.seekg(offset, std::ios::beg);
    file.read(reinterpret_cast<char*>(data_ptr), bytes);

    if (file.gcount() != bytes) 
        throw std::runtime_error("Failed to read the expected number of bytes from file: " + filename);

    file.close(); 

    return std::make_shared<Storage>(bytes, device, std::unique_ptr<std::byte[], StorageDeleter>(data_ptr, StorageDeleter(device)));
}

std::byte* Storage::allocate(size_t size, Device device) {
	if (device == Device::CUDA) {
		// std::byte* ptr;
		// cudaMalloc(&ptr, size);
		throw std::runtime_error("CUDA device type is not supported yet for Storage::allocate");
	} else if (device == Device::MPS) {
		// std::byte* ptr;
		// mpsMalloc(&ptr, size);
		throw std::runtime_error("MPS device type is not supported yet for Storage::allocate");
	} else if (device == Device::CPU) {
		return new std::byte[size];
	}
	throw std::runtime_error("Unsupported device type for Storage::allocate");
}

std::byte* Storage::data() { return reinterpret_cast<std::byte*>(data_.get()); }

const std::byte* Storage::data() const { return reinterpret_cast<const std::byte*>(data_.get()); }

size_t Storage::nbytes() const { return bytes_; }

Device Storage::device() const {
	return device_;
}
