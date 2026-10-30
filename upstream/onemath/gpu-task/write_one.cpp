// A placeholder program: gpu-test needs a SYCL source to build. The result that matters is in
// setup.log (oneMath's own unit tests, run by run_tests.sh).
#include <sycl/sycl.hpp>
#include <fstream>
int main(int, char** argv) {
    sycl::queue q;
    float* x = sycl::malloc_shared<float>(1, q);
    q.single_task([=] { *x = 1.0f; }).wait();
    std::ofstream(argv[2], std::ios::binary).write(reinterpret_cast<const char*>(x), sizeof(float));
    sycl::free(x, q);
    return 0;
}
