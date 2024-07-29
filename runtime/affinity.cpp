#include "lockfreequeue.h"
#include "test_local.c"
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <x86intrin.h>

LockFreeQueue<SharedData> atomicQueue(M); // local to struct
int (*remote1)(int, int[], int[]) ;
// Function to set CPU affinity
void set_cpu_affinity(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Error setting CPU affinity" << std::endl;
        exit(1);
    }
}
void process_queue_item(int i) {
    while (!atomicQueue[i].valid) {
        std::this_thread::yield();
    }
    atomicQueue[i].res = remote1(atomicQueue[i].i, atomicQueue[i].a, atomicQueue[i].b);
}
// Remote thread function
void *remote_thread_func(void *arg) {
    set_cpu_affinity(64);
    void *handle = dlopen("./libremote.so", RTLD_NOW | RTLD_GLOBAL);
    printf("handle: %p\n", handle);
    if (!handle) {
        exit(-1);
    }
    dlerror();
    int (*remote1)(int, int[], int[]) = (int (*)(int, int[], int[]))dlsym(handle, "remote");

    std::vector<std::future<void>> futures;
    for (int i = 0; i < M / 4; i += 1) {
        futures.push_back(std::async(std::launch::async, process_queue_item, i));
    }
    for (auto &future : futures) {
        future.wait();
    }
    // for (int i = 0; i < M / 4; i += 1) {
    //     while (!atomicQueue[i].valid) {
    //         usleep(1); // how to make it async?
    //     }
    //     atomicQueue[i].res = (remote1(atomicQueue[i].i, atomicQueue[i].a, atomicQueue[i].b));
    // }
    return nullptr;
}

// Local thread function
void *local_thread_func(void *arg) {
    set_cpu_affinity(0);
    local_func();
    return nullptr;
}

int main() {
    // Allocate shared data structure on NUMA node 1 (remote)

    // Create threads
    pthread_t remote_thread, local_thread;
    pthread_create(&remote_thread, nullptr, remote_thread_func, nullptr);
    pthread_create(&local_thread, nullptr, local_thread_func, nullptr);

    // Wait for threads to complete
    pthread_join(remote_thread, nullptr);
    pthread_join(local_thread, nullptr);

    return 0;
}