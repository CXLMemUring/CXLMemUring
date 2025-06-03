// main_riscv.cpp
#include "shared_protocol.h"
#include <iostream>
#include <map>
#include <thread>
#include <chrono>

// RISCV端的内存管理
class MemoryManager {
private:
    std::map<uint64_t, BTreeNode> node_map;
    
public:
    // 获取节点引用，如果不存在则创建
    BTreeNode& get_or_create_node(uint64_t node_id) {
        auto it = node_map.find(node_id);
        if (it != node_map.end()) {
            return it->second;
        }
        
        // 创建新节点
        BTreeNode node;
        node.is_leaf = true; // 默认设置
        node.num_keys = 0;
        node_map[node_id] = node;
        return node_map[node_id];
    }
};

// RISCV端的BTree操作实现
void insert(BTreeNode& node, int key) {
    // 简化的插入实现
    if (node.is_leaf) {
        // 叶节点操作
        int i = node.num_keys - 1;
        while (i >= 0 && key < node.keys[i]) {
            node.keys[i + 1] = node.keys[i];
            i--;
        }
        node.keys[i + 1] = key;
        node.num_keys++;
    } else {
        // 非叶节点操作 - 简化
        // 在实际实现中需要处理子节点
    }
}

int main() {
    // 创建共享内存管理器
    SharedMemoryManager shm(true); // 创建者
    
    // 创建内存管理器
    MemoryManager mem_manager;
    
    // 处理来自x86的请求
    BTreeOpRequest req;
    BTreeOpResponse resp;
    bool running = true;
    
    std::cout << "RISCV processor started" << std::endl;
    
    while (running) {
        // 接收请求
        if (shm.receive_request(req)) {
            // 处理请求
            if (req.op_type == OP_TERMINATE) {
                running = false;
                continue;
            } else if (req.op_type == OP_INSERT) {
                // 获取或创建节点
                BTreeNode& node = mem_manager.get_or_create_node(req.node_id);
                
                // 执行插入操作
                insert(node, req.key);
                
                // 将更新后的节点推送到共享内存
                while (!shm.push_node_to_buffer(req.node_id, node)) {
                    std::this_thread::yield();
                }
                
                // 发送响应
                resp.status = 0; // 成功
                resp.node_id = req.node_id;
                while (!shm.send_response(resp)) {
                    std::this_thread::yield();
                }
            }
        }
        
        // 避免过度占用CPU
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    std::cout << "RISCV processor terminating" << std::endl;
    return 0;
}