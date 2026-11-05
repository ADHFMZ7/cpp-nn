#include "storage.h"

#include <stdexcept>
#include <utility>

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

Device Storage::device() const {
	return device_;
}




