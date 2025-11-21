// shared_protocol.h
#ifndef SHARED_PROTOCOL_H
#define SHARED_PROTOCOL_H

#include <stdint.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#ifdef __cplusplus
#include <cstring>
#else
#include <string.h>
#endif
// 共享内存名称和信号量名称
#define SHM_NAME "/btree_shared_mem"
#define SEM_REQUEST_NAME "/btree_request_sem"
#define SEM_RESPONSE_NAME "/btree_response_sem"
#define SEM_BUFFER_NAME "/btree_buffer_sem"

// BTree节点定义 (简化版)
struct BTreeNode {
    int keys[100];  // 简化为固定大小
    uint64_t children[100];  // 用ID表示子节点
    int num_keys;
    bool is_leaf;
};

// 操作类型
enum OperationType {
    OP_INSERT = 1,
    OP_DELETE = 2,
    OP_SEARCH = 3,
    OP_TERMINATE = 255  // 终止信号
};

// 请求结构
struct BTreeOpRequest {
    uint8_t op_type;     // OperationType
    int key;             // 键值
    uint64_t node_id;    // 节点ID
};

// 响应结构
struct BTreeOpResponse {
    uint8_t status;      // 状态码
    uint64_t node_id;    // 被操作的节点ID
};

// 共享内存布局
struct SharedMemory {
    // 请求队列
    BTreeOpRequest requests[256];
    uint8_t request_write_idx;
    uint8_t request_read_idx;
    
    // 响应队列
    BTreeOpResponse responses[256];
    uint8_t response_write_idx;
    uint8_t response_read_idx;
    
    // 节点数据缓冲区 - 用于传输更新的节点数据
    BTreeNode node_buffer;
    uint64_t buffer_node_id;
    bool buffer_valid;
};

// 共享内存管理类
class SharedMemoryManager {
private:
    int shm_fd;
    SharedMemory* shared_mem;
    sem_t* request_sem;
    sem_t* response_sem;
    sem_t* buffer_sem;
    bool is_owner;

public:
    // 创建/打开共享内存
    SharedMemoryManager(bool create_new = false) : is_owner(create_new) {
        if (create_new) {
            // 创建共享内存
            shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
            ftruncate(shm_fd, sizeof(SharedMemory));
            
            // 创建信号量
            request_sem = sem_open(SEM_REQUEST_NAME, O_CREAT, 0666, 1);
            response_sem = sem_open(SEM_RESPONSE_NAME, O_CREAT, 0666, 1);
            buffer_sem = sem_open(SEM_BUFFER_NAME, O_CREAT, 0666, 1);
        } else {
            // 打开已存在的共享内存
            shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
            
            // 打开已存在的信号量
            request_sem = sem_open(SEM_REQUEST_NAME, 0);
            response_sem = sem_open(SEM_RESPONSE_NAME, 0);
            buffer_sem = sem_open(SEM_BUFFER_NAME, 0);
        }
        
        // 映射共享内存
        shared_mem = (SharedMemory*)mmap(NULL, sizeof(SharedMemory), 
                                         PROT_READ | PROT_WRITE, MAP_SHARED, 
                                         shm_fd, 0);
        
        // 如果是创建者，初始化共享内存
        if (create_new) {
            memset(shared_mem, 0, sizeof(SharedMemory));
        }
    }
    
    // 发送请求
    bool send_request(const BTreeOpRequest& req) {
        sem_wait(request_sem);
        
        // 检查队列是否已满
        if ((shared_mem->request_write_idx + 1) % 256 == shared_mem->request_read_idx) {
            sem_post(request_sem);
            return false;
        }
        
        // 写入请求
        shared_mem->requests[shared_mem->request_write_idx] = req;
        shared_mem->request_write_idx = (shared_mem->request_write_idx + 1) % 256;
        
        sem_post(request_sem);
        return true;
    }
    
    // 接收请求
    bool receive_request(BTreeOpRequest& req) {
        sem_wait(request_sem);
        
        // 检查队列是否为空
        if (shared_mem->request_read_idx == shared_mem->request_write_idx) {
            sem_post(request_sem);
            return false;
        }
        
        // 读取请求
        req = shared_mem->requests[shared_mem->request_read_idx];
        shared_mem->request_read_idx = (shared_mem->request_read_idx + 1) % 256;
        
        sem_post(request_sem);
        return true;
    }
    
    // 发送响应
    bool send_response(const BTreeOpResponse& resp) {
        sem_wait(response_sem);
        
        // 检查队列是否已满
        if ((shared_mem->response_write_idx + 1) % 256 == shared_mem->response_read_idx) {
            sem_post(response_sem);
            return false;
        }
        
        // 写入响应
        shared_mem->responses[shared_mem->response_write_idx] = resp;
        shared_mem->response_write_idx = (shared_mem->response_write_idx + 1) % 256;
        
        sem_post(response_sem);
        return true;
    }
    
    // 接收响应
    bool receive_response(BTreeOpResponse& resp) {
        sem_wait(response_sem);
        
        // 检查队列是否为空
        if (shared_mem->response_read_idx == shared_mem->response_write_idx) {
            sem_post(response_sem);
            return false;
        }
        
        // 读取响应
        resp = shared_mem->responses[shared_mem->response_read_idx];
        shared_mem->response_read_idx = (shared_mem->response_read_idx + 1) % 256;
        
        sem_post(response_sem);
        return true;
    }
    
    // 推送节点数据到缓冲区
    bool push_node_to_buffer(uint64_t node_id, const BTreeNode& node) {
        sem_wait(buffer_sem);
        
        // 如果缓冲区已有数据且未被处理，返回失败
        if (shared_mem->buffer_valid) {
            sem_post(buffer_sem);
            return false;
        }
        
        // 写入节点数据
        shared_mem->node_buffer = node;
        shared_mem->buffer_node_id = node_id;
        shared_mem->buffer_valid = true;
        
        sem_post(buffer_sem);
        return true;
    }
    
    // 从缓冲区获取节点数据
    bool get_node_from_buffer(uint64_t& node_id, BTreeNode& node) {
        sem_wait(buffer_sem);
        
        // 如果缓冲区没有数据，返回失败
        if (!shared_mem->buffer_valid) {
            sem_post(buffer_sem);
            return false;
        }
        
        // 读取节点数据
        node = shared_mem->node_buffer;
        node_id = shared_mem->buffer_node_id;
        shared_mem->buffer_valid = false;
        
        sem_post(buffer_sem);
        return true;
    }
    
    ~SharedMemoryManager() {
        // 解除内存映射
        munmap(shared_mem, sizeof(SharedMemory));
        close(shm_fd);
        
        // 关闭信号量
        sem_close(request_sem);
        sem_close(response_sem);
        sem_close(buffer_sem);
        
        // 如果是创建者，删除共享内存和信号量
        if (is_owner) {
            shm_unlink(SHM_NAME);
            sem_unlink(SEM_REQUEST_NAME);
            sem_unlink(SEM_RESPONSE_NAME);
            sem_unlink(SEM_BUFFER_NAME);
        }
    }
};

#endif // SHARED_PROTOCOL_H
