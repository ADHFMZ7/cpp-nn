#pragma once

#include <memory>
#include <cstddef>
#include <stdexcept>

enum class Device {CUDA, CPU, MPS};

class StorageDeleter {

public:
    explicit StorageDeleter(Device device);
    void operator()(std::byte* ptr) const;

private:
    Device device_;
};

class Storage {

public:
    Storage(size_t size, Device device, std::unique_ptr<std::byte[], StorageDeleter> data);

    // CUDA and MPS allocations are not implemented yet.
    Storage(size_t size, Device device);

    static std::shared_ptr<Storage> from_file(const std::string& filename, size_t offset, size_t bytes, Device device);

    static std::byte* allocate(size_t size, Device device);

    Device device() const;
    size_t nbytes() const;

    std::byte* data();
    const std::byte* data() const;

private:

    std::unique_ptr<std::byte[], StorageDeleter> data_;
    size_t bytes_;
    Device device_;
    // data type is also necessary for proper memory management, but for now we will assume double precision floats.
};
