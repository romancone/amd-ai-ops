#include <hip/hip_runtime.h>
#include <cstdio>
#include <string>
#include <vector>

#define HIP_OK(call) do { \
    hipError_t error = (call); \
    if (error != hipSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); \
        return 1; \
    } \
} while (false)

__global__ void add_one(float* values, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] += 1.0f;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::string expected = argv[1];
    int count = 0;
    HIP_OK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        HIP_OK(hipGetDeviceProperties(&properties, device));
        std::string target = properties.gcnArchName;
        if (target.substr(0, target.find(':')) == expected) {
            selected = device;
            break;
        }
    }
    if (selected < 0) {
        std::fprintf(stderr, "No HIP device matches %s\n", expected.c_str());
        return 1;
    }
    HIP_OK(hipSetDevice(selected));
    constexpr int elements = 1024;
    std::vector<float> values(elements);
    for (int index = 0; index < elements; ++index) values[index] = float(index);
    float* device_values = nullptr;
    HIP_OK(hipMalloc(reinterpret_cast<void**>(&device_values), elements * sizeof(float)));
    HIP_OK(hipMemcpy(device_values, values.data(), elements * sizeof(float), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(add_one, dim3(4), dim3(256), 0, 0, device_values, elements);
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(values.data(), device_values, elements * sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipFree(device_values));
    for (int index = 0; index < elements; ++index) {
        if (values[index] != float(index + 1)) {
            std::fprintf(stderr, "Incorrect result at %d\n", index);
            return 1;
        }
    }
    std::printf("PASS target=%s device=%d elements=%d\n", expected.c_str(), selected, elements);
    return 0;
}
