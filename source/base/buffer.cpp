#include "base/buffer.h"
#include <glog/logging.h>
#include <algorithm> // For std::min
#include <utility>   // For std::move

namespace base {

/**
 * @brief Helper function to perform the actual cross-device copy logic.
 * It simplifies the implementation of the public copy_from methods.
 */
static void PerformCopy(const std::shared_ptr<DeviceAllocator>& allocator,
                        const void* src_ptr, size_t src_byte_size, DeviceType src_device_type,
                        void* dest_ptr, size_t dest_byte_size, DeviceType dest_device_type) {
    
    CHECK(allocator != nullptr) << "Cannot copy: Allocator is null.";
    CHECK(dest_ptr != nullptr) << "Destination buffer pointer is null (allocate first).";
    CHECK(src_ptr != nullptr) << "Source buffer pointer is null.";
    
    // Determine the actual size to copy (min of source and destination sizes)
    size_t copy_size = std::min(dest_byte_size, src_byte_size);
    
    CHECK(src_device_type != DeviceType::kDeviceUnknown && 
          dest_device_type != DeviceType::kDeviceUnknown)
        << "Source or destination device type is unknown.";

    MemcpyKind kind;
    
    // Determine the MemcpyKind
    if (src_device_type == DeviceType::kDeviceCPU && dest_device_type == DeviceType::kDeviceCPU) {
        kind = MemcpyKind::kMemcpyCPU2CPU;
    } else if (src_device_type == DeviceType::kDeviceCUDA && dest_device_type == DeviceType::kDeviceCPU) {
        kind = MemcpyKind::kMemcpyCUDA2CPU;
    } else if (src_device_type == DeviceType::kDeviceCPU && dest_device_type == DeviceType::kDeviceCUDA) {
        kind = MemcpyKind::kMemcpyCPU2CUDA;
    } else {
        // Includes CUDA to CUDA
        kind = MemcpyKind::kMemcpyCUDA2CUDA;
    }
    
    // Perform the copy operation. Assumes signature is (src, dest, size, kind, ...)
    allocator->memcpy(src_ptr, dest_ptr, copy_size, kind);
}


// --- Constructor and Destructor ---

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
               bool use_external)
    // Use std::move for allocator for efficiency
    : byte_size_(byte_size),
      allocator_(std::move(allocator)), 
      ptr_(ptr),
      use_external_(use_external) 
{
    if (allocator_) {
        // Always set the device type if an allocator is provided
        device_type_ = allocator_->device_type();
    }
    
    // Auto-allocate if no external pointer was given
    if (!ptr_ && allocator_) {
        // If we allocate now, it's definitely not external memory
        use_external_ = false; 
        
        // Use the public allocate method
        if (!allocate()) {
             // Log error if initial allocation fails
             LOG(ERROR) << "Failed to allocate buffer of size " << byte_size_ << " bytes.";
        }
    }
}

Buffer::~Buffer() {
    // Only release memory if it's NOT external (i.e., we own it)
    if (!use_external_) {
        if (ptr_ && allocator_) {
            allocator_->release(ptr_);
            ptr_ = nullptr; // Crucial: clear the pointer after releasing
        }
    }
}


// --- Accessors ---

void* Buffer::ptr() { return ptr_; }

const void* Buffer::ptr() const { return ptr_; }

size_t Buffer::byte_size() const { return byte_size_; }

std::shared_ptr<DeviceAllocator> Buffer::allocator() const { return allocator_; }

DeviceType Buffer::device_type() const { return device_type_; }

void Buffer::set_device_type(DeviceType device_type) { device_type_ = device_type; }

std::shared_ptr<Buffer> Buffer::get_shared_from_this() { return shared_from_this(); }

bool Buffer::is_external() const { return this->use_external_; }


bool Buffer::allocate() {
    // If memory is already allocated and we own it, success.
    if (ptr_ != nullptr && !use_external_) {
        return true; 
    }
    
    // Only attempt allocation if we have a valid allocator and non-zero size
    if (allocator_ && byte_size_ != 0) {
        // We are taking ownership
        use_external_ = false; 
        
        ptr_ = allocator_->allocate(byte_size_);
        
        if (!ptr_) {
            LOG(ERROR) << "DeviceAllocator failed to allocate " << byte_size_ << " bytes.";
            return false;
        }
        
        // Ensure device type is set
        device_type_ = allocator_->device_type();
        return true;
    }
    
    return false;
}

void Buffer::copy_from(const Buffer& buffer) const {
    // Check that source pointer is valid
    CHECK(buffer.ptr_ != nullptr) << "Source buffer (const&) has null pointer.";

    // Use the consolidated helper
    PerformCopy(
        allocator_,
        buffer.ptr_, buffer.byte_size_, buffer.device_type(),
        this->ptr_, this->byte_size_, this->device_type()
    );
}

void Buffer::copy_from(const Buffer* buffer) const {
    // Original CHECK was complex: CHECK(buffer != nullptr || buffer->ptr_ != nullptr);
    // The robust check is two separate steps:
    CHECK(buffer != nullptr) << "Source buffer (const*) pointer is null.";
    CHECK(buffer->ptr_ != nullptr) << "Source buffer internal pointer is null.";

    // Use the consolidated helper
    PerformCopy(
        allocator_,
        buffer->ptr_, buffer->byte_size_, buffer->device_type(),
        this->ptr_, this->byte_size_, this->device_type()
    );
}

} // namespace base