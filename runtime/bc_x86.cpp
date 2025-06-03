#include "shared_protocol.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <thread>
#include <unistd.h>
#include <vector>
#include <wait.h>
#define N 100000000
#define M 100000000
#define K 100

// 本地的BTree节点管理
class BTreeManager {
private:
    std::map<uint64_t, BTreeNode *> nodes;
    uint64_t next_id = 1;

public:
    BTreeNode *create_node(bool is_leaf) {
        BTreeNode *node = new BTreeNode();
        node->is_leaf = is_leaf;
        node->num_keys = 0;

        uint64_t id = next_id++;
        nodes[id] = node;
        return node;
    }

    uint64_t get_node_id(BTreeNode *node) {
        for (const auto &pair : nodes) {
            if (pair.second == node) {
                return pair.first;
            }
        }
        return 0; // 未找到
    }

    BTreeNode *get_node(uint64_t id) {
        auto it = nodes.find(id);
        if (it != nodes.end()) {
            return it->second;
        }
        return nullptr;
    }

    // 更新节点数据
    void update_node(uint64_t id, const BTreeNode &data) {
        auto it = nodes.find(id);
        if (it != nodes.end()) {
            *(it->second) = data;
        }
    }
};

// 异步任务
class AsyncTask {
private:
    SharedMemoryManager &shm;
    int key;
    uint64_t node_id;
    bool completed = false;

public:
    AsyncTask(SharedMemoryManager &s, int k, uint64_t n) : shm(s), key(k), node_id(n) {}

    bool execute() {
        if (!completed) {
            BTreeOpRequest req;
            req.op_type = OP_INSERT;
            req.key = key;
            req.node_id = node_id;

            if (shm.send_request(req)) {
                BTreeOpResponse resp;
                if (shm.receive_response(resp) && resp.node_id == node_id) {
                    completed = true;
                    return true;
                }
            }
            return false;
        }
        return true;
    }

    bool is_completed() const { return completed; }
};

int main() {
    // 启动RISCV进程
    pid_t child_pid = fork();

    if (child_pid == 0) {
        // 子进程 - 模拟RISCV
        execl("./riscv_btree", "riscv_btree", NULL);
        exit(EXIT_FAILURE); // 如果execl失败
    }

    // 主进程 - x86
    // 等待一段时间让RISCV进程初始化
    sleep(1);

    // 创建共享内存管理器
    SharedMemoryManager shm(false); // 非创建者

    // 创建BTree管理器
    BTreeManager btree_mgr;

    // 开始执行BTree操作
    auto start = std::chrono::high_resolution_clock::now();

    BTreeNode *root = btree_mgr.create_node(true);
    uint64_t root_id = btree_mgr.get_node_id(root);

    std::vector<AsyncTask> tasks;

    // 创建任务
    for (int i = 0; i < M; i += K) {
        tasks.emplace_back(shm, i, root_id);
    }

    // 执行任务
    bool all_completed;
    do {
        all_completed = true;
        for (auto &task : tasks) {
            if (!task.is_completed()) {
                all_completed = false;
                task.execute();
            }
        }

        // 检查并应用节点更新
        uint64_t updated_id;
        BTreeNode updated_node;
        if (shm.get_node_from_buffer(updated_id, updated_node)) {
            btree_mgr.update_node(updated_id, updated_node);
        }

        std::this_thread::yield();
    } while (!all_completed);

    // 发送终止信号
    BTreeOpRequest term_req;
    term_req.op_type = OP_TERMINATE;
    shm.send_request(term_req);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    // 等待子进程结束
    waitpid(child_pid, NULL, 0);

    return 0;
}