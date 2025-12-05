// Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
// 
// Maintainer: Wjliu (mcmillantac@163.com)
// Algorithm of paper: < AutoPipe: A Fast Pipeline Parallelism Approach 
// with Balanced Partitioning and Micro-batch Slicing >
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Algorithm for auto pipeline partition according to critical path for synchronized pipeline.
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

namespace torchpipe {

// 常量定义
constexpr long long kCommunicationOverhead = 0;
constexpr long long kMaxLongLong = std::numeric_limits<long long>::max();
constexpr int kMaxInt32 = std::numeric_limits<int32_t>::max();

// 前向声明
class PipelinePartitioner {
public:
    static vector<int> merak_pipe(
        const vector<long long>& forward_times,
        const vector<long long>& backward_times,
        int num_stages
    );
    
private:
    struct PartitionResult {
        vector<vector<int>> partition;
        long long cost;
        int critical_stage;
    };
    
    // 核心算法函数
    static vector<vector<int>> block_partition_algorithm(
        const vector<int>& model,
        int num_stages,
        const vector<vector<long long>>& block_time_mapping
    );
    
    static void reconstruct_partitions(
        const vector<int>& model,
        const vector<long long>& prefix_sum,
        const vector<vector<long long>>& dp,
        int remaining_blocks,
        int remaining_partitions,
        vector<vector<int>>& partition
    );
    
    static pair<long long, int> calculate_training_time(
        const vector<vector<int>>& partition,
        const vector<vector<long long>>& block_time_mapping
    );
    
    static void calculate_stage_times(
        const vector<vector<int>>& partition,
        const vector<vector<long long>>& block_time_mapping,
        vector<long long>& forward_time,
        vector<long long>& backward_time,
        vector<int>& last_microbatch
    );
    
    static pair<long long, int> calculate_steady_phase(
        const vector<int>& last_batch,
        const vector<long long>& forward_time,
        const vector<long long>& backward_time
    );
    
    static long long calculate_cooldown_phase(
        int num_stages,
        int critical_stage,
        long long last_forward_start,
        const vector<long long>& forward_time,
        const vector<long long>& backward_time
    );
    
    static PartitionResult find_best_partition(
        const vector<vector<long long>>& block_time_mapping,
        int num_stages,
        const vector<vector<int>>& initial_partition,
        const vector<long long>& prefix_sum,
        const vector<vector<long long>>& dp_array
    );
    
    static void calculate_prefix_sum_and_dp(
        const vector<int>& model,
        int num_stages,
        const vector<vector<long long>>& block_time_mapping,
        vector<long long>& prefix_sum,
        vector<vector<long long>>& dp_array
    );
};

// 实现部分
void PipelinePartitioner::calculate_prefix_sum_and_dp(
    const vector<int>& model,
    int num_stages,
    const vector<vector<long long>>& block_time_mapping,
    vector<long long>& prefix_sum,
    vector<vector<long long>>& dp_array
) {
    int num_blocks = model.size();
    int max_partitions = min(num_blocks, num_stages);
    
    // 计算前缀和
    prefix_sum.clear();
    prefix_sum.reserve(num_blocks + 1);
    prefix_sum.push_back(0);
    
    for (int i = 0; i < num_blocks; ++i) {
        int block = model[i];
        prefix_sum.push_back(prefix_sum.back() + 
                            block_time_mapping[0][block] + 
                            block_time_mapping[1][block]);
    }
    
    // 动态规划数组
    dp_array.assign(num_blocks + 1, vector<long long>(max_partitions + 1, kMaxLongLong));
    dp_array[0][0] = 0;
    
    // 动态规划计算
    for (int blocks = 1; blocks <= num_blocks; ++blocks) {
        int max_p = min(blocks, max_partitions);
        for (int partitions = 1; partitions <= max_p; ++partitions) {
            long long min_val = kMaxLongLong;
            for (int prev_blocks = 0; prev_blocks < blocks; ++prev_blocks) {
                long long val = max(dp_array[prev_blocks][partitions - 1],
                                   prefix_sum[blocks] - prefix_sum[prev_blocks]);
                min_val = min(min_val, val);
                if (min_val == 0) break;
            }
            dp_array[blocks][partitions] = min_val;
        }
    }
}

vector<vector<int>> PipelinePartitioner::block_partition_algorithm(
    const vector<int>& model,
    int num_stages,
    const vector<vector<long long>>& block_time_mapping
) {
    vector<long long> prefix_sum;
    vector<vector<long long>> dp_array;
    
    calculate_prefix_sum_and_dp(model, num_stages, block_time_mapping, prefix_sum, dp_array);
    
    vector<vector<int>> partition;
    reconstruct_partitions(model, prefix_sum, dp_array, 
                          model.size(), num_stages, partition);
    reverse(partition.begin(), partition.end());
    
    return partition;
}

void PipelinePartitioner::reconstruct_partitions(
    const vector<int>& model,
    const vector<long long>& prefix_sum,
    const vector<vector<long long>>& dp_array,
    int remaining_blocks,
    int remaining_partitions,
    vector<vector<int>>& partition
) {
    if (remaining_blocks == 0 && remaining_partitions == 0) return;
    
    if (remaining_blocks <= 0 || remaining_partitions <= 0 || 
        remaining_blocks < remaining_partitions) {
        throw runtime_error("Error during partition reconstruction");
    }
    
    int prev_end = 0;
    while (prev_end < remaining_blocks && 
           dp_array[remaining_blocks][remaining_partitions] != 
           max(dp_array[prev_end][remaining_partitions - 1], 
               prefix_sum[remaining_blocks] - prefix_sum[prev_end])) {
        ++prev_end;
    }
    
    vector<int> current_partition;
    current_partition.reserve(remaining_blocks - prev_end);
    for (int i = prev_end + 1; i <= remaining_blocks; ++i) {
        current_partition.push_back(model[i - 1]);
    }
    partition.push_back(move(current_partition));
    
    reconstruct_partitions(model, prefix_sum, dp_array, prev_end, 
                          remaining_partitions - 1, partition);
}

void PipelinePartitioner::calculate_stage_times(
    const vector<vector<int>>& partition,
    const vector<vector<long long>>& block_time_mapping,
    vector<long long>& forward_time,
    vector<long long>& backward_time,
    vector<int>& last_microbatch
) {
    int num_stages = partition.size();
    int num_microbatches = num_stages * 2;
    
    // 构建最后微批次数组
    for (int i = 0; i < num_stages; ++i) {
        last_microbatch[i] = num_microbatches - num_stages + i;
    }
    
    // 计算每个阶段的前向和后向时间
    for (int i = 1; i <= num_stages; ++i) {
        long long forward_sum = 0, backward_sum = 0;
        for (int block_type : partition[i - 1]) {
            forward_sum += block_time_mapping[0][block_type];
            backward_sum += block_time_mapping[1][block_type];
        }
        forward_time[i] = forward_sum;
        backward_time[i] = backward_sum;
    }
}

pair<long long, int> PipelinePartitioner::calculate_steady_phase(
    const vector<int>& last_batch,
    const vector<long long>& forward_time,
    const vector<long long>& backward_time
) {
    int num_stages = last_batch.size();
    int num_microbatches = num_stages * 2;
    
    // 动态规划数组
    vector<vector<vector<long long>>> dp(num_stages + 2, 
                                        vector<vector<long long>>(num_microbatches, 
                                        vector<long long>(2, 0)));
    
    // 初始化
    long long initial_backward_start = 0;
    for (int stage = 0; stage < num_stages; ++stage) {
        initial_backward_start += forward_time[stage + 1];
        if (stage != num_stages - 1) initial_backward_start += kCommunicationOverhead;
    }
    
    for (int stage = num_stages - 1; stage >= 0; --stage) {
        dp[stage + 1][0][0] = kMaxLongLong;
        dp[stage + 1][0][1] = initial_backward_start;
        initial_backward_start += backward_time[stage + 1] + kCommunicationOverhead;
    }
    
    // 计算稳态阶段
    for (int microbatch = 1; microbatch < num_microbatches; ++microbatch) {
        // 前向计算
        for (int stage = 0; stage < num_stages; ++stage) {
            if (microbatch <= last_batch[stage]) {
                dp[stage + 1][microbatch][0] = max(
                    dp[stage][microbatch - 1][0] + forward_time[stage],
                    dp[stage + 1][microbatch - 1][1] + backward_time[stage + 1]
                );
                if (stage != 0) dp[stage + 1][microbatch][0] += kCommunicationOverhead;
            }
        }
        
        // 后向计算
        for (int stage = num_stages - 1; stage >= 0; --stage) {
            if (microbatch <= last_batch[stage]) {
                dp[stage + 1][microbatch][1] = max(
                    dp[stage + 2][microbatch][1] + backward_time[stage + 2],
                    dp[stage + 1][microbatch][0] + forward_time[stage + 1]
                );
                if (stage != num_stages - 1) dp[stage + 1][microbatch][1] += kCommunicationOverhead;
            }
        }
    }
    
    // 寻找关键路径阶段
    int critical_stage = num_stages - 1;
    while (critical_stage >= 0) {
        int microbatch;
        long long forward_comm = (critical_stage != 0) ? kCommunicationOverhead : 0;
        long long backward_comm = (critical_stage != num_stages - 1) ? kCommunicationOverhead : 0;
        
        for (microbatch = 1; microbatch <= last_batch[critical_stage]; ++microbatch) {
            if (dp[critical_stage + 1][microbatch][0] != 
                dp[critical_stage + 1][microbatch - 1][1] + 
                backward_time[critical_stage + 1] + forward_comm) {
                break;
            }
            
            if (dp[critical_stage + 1][microbatch][1] != 
                dp[critical_stage + 1][microbatch][0] + 
                forward_time[critical_stage + 1] + backward_comm) {
                break;
            }
        }
        
        if (microbatch == last_batch[critical_stage] + 1) break;
        --critical_stage;
    }
    
    if (critical_stage < 0) {
        throw runtime_error("Failed to determine critical stage");
    }
    
    return make_pair(dp[critical_stage + 1][last_batch[critical_stage]][0], 
                     critical_stage);
}

long long PipelinePartitioner::calculate_cooldown_phase(
    int num_stages,
    int critical_stage,
    long long last_forward_start,
    const vector<long long>& forward_time,
    const vector<long long>& backward_time
) {
    int vector_size = num_stages - critical_stage;
    if (vector_size <= 0) return last_forward_start;
    
    vector<vector<long long>> dp(vector_size, vector<long long>(vector_size, 0));
    long long backward_start = last_forward_start;
    
    // 初始化
    for (int i = 0; i < vector_size; ++i) {
        backward_start += forward_time[critical_stage + 1 + i];
        if (critical_stage + i != num_stages - 1) {
            backward_start += kCommunicationOverhead;
        }
        int j = vector_size - 1 - i;
        dp[i][j] = backward_start;
    }
    
    // 运行动态规划
    for (int col = vector_size - 2; col >= 0; --col) {
        for (int row = vector_size - col - 2; row >= 0; --row) {
            long long option1 = dp[row][col + 1] + 
                              backward_time[critical_stage + 1 + row] + 
                              kCommunicationOverhead;
            long long option2 = dp[row + 1][col] + 
                              backward_time[critical_stage + 1 + row + 1] + 
                              kCommunicationOverhead;
            dp[row][col] = max(option1, option2);
            
            if (row > 0) {
                dp[row][col] = max(dp[row][col], dp[row - 1][col + 1]);
            }
        }
    }
    
    return dp[0][0];
}

pair<long long, int> PipelinePartitioner::calculate_training_time(
    const vector<vector<int>>& partition,
    const vector<vector<long long>>& block_time_mapping
) {
    int num_stages = partition.size();
    int num_microbatches = num_stages * 2;
    
    vector<int> last_microbatch(num_stages);
    vector<long long> forward_time(num_stages + 2, 0);
    vector<long long> backward_time(num_stages + 2, 0);
    
    // 计算阶段时间
    for (int i = 0; i < num_stages; ++i) {
        last_microbatch[i] = num_microbatches - num_stages + i;
        
        long long forward_sum = 0, backward_sum = 0;
        for (int block : partition[i]) {
            forward_sum += block_time_mapping[0][block];
            backward_sum += block_time_mapping[1][block];
        }
        forward_time[i + 1] = forward_sum;
        backward_time[i + 1] = backward_sum;
    }
    
    auto steady_result = calculate_steady_phase(last_microbatch, 
                                                forward_time, 
                                                backward_time);
    
    long long last_forward_start = steady_result.first;
    int critical_stage = steady_result.second;
    
    if (last_forward_start == kMaxLongLong) {
        throw runtime_error("Failed to calculate steady phase");
    }
    
    long long last_backward_start = calculate_cooldown_phase(
        num_stages, critical_stage, last_forward_start, 
        forward_time, backward_time);
    
    long long pipeline_flush_time = last_backward_start;
    for (int stage = critical_stage; stage > 0; --stage) {
        pipeline_flush_time += backward_time[stage + 1] + kCommunicationOverhead;
    }
    pipeline_flush_time += backward_time[1];
    
    return make_pair(pipeline_flush_time, critical_stage);
}

PipelinePartitioner::PartitionResult PipelinePartitioner::find_best_partition(
    const vector<vector<long long>>& block_time_mapping,
    int num_stages,
    const vector<vector<int>>& initial_partition,
    const vector<long long>& prefix_sum,
    const vector<vector<long long>>& dp_array
) {
    // 哈希函数用于unordered_set
    struct VectorHash {
        size_t operator()(const vector<vector<int>>& v) const {
            size_t hash = 0;
            for (const auto& inner : v) {
                for (int val : inner) {
                    hash ^= hash << 13;
                    hash ^= hash >> 7;
                    hash ^= hash << 17;
                    hash ^= val + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
            }
            return hash;
        }
    };
    
    struct VectorEqual {
        bool operator()(const vector<vector<int>>& a, const vector<vector<int>>& b) const {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i].size() != b[i].size()) return false;
                for (size_t j = 0; j < a[i].size(); ++j) {
                    if (a[i][j] != b[i][j]) return false;
                }
            }
            return true;
        }
    };
    
    vector<int> last_microbatch(num_stages, 0);
    vector<long long> forward_time(num_stages + 2, 0);
    vector<long long> backward_time(num_stages + 2, 0);
    
    // 初始化最优结果
    PartitionResult best_result;
    best_result.cost = kMaxLongLong;
    best_result.critical_stage = kMaxInt32;
    
    // 记录已处理的分区
    unordered_set<vector<vector<int>>, VectorHash, VectorEqual> visited;
    queue<vector<vector<int>>> partitions_queue;
    partitions_queue.push(initial_partition);
    visited.insert(initial_partition);
    
    while (!partitions_queue.empty()) {
        vector<vector<int>> current_partition = partitions_queue.front();
        partitions_queue.pop();
        
        // 计算当前分区的时间
        calculate_stage_times(current_partition, block_time_mapping, 
                             forward_time, backward_time, last_microbatch);
        
        auto time_result = calculate_training_time(current_partition, 
                                                  block_time_mapping);
        long long current_cost = time_result.first;
        int current_critical = time_result.second;
        
        // 更新最优结果
        if (current_cost < best_result.cost) {
            best_result.partition = current_partition;
            best_result.cost = current_cost;
            best_result.critical_stage = current_critical;
        }
        
        // 尝试调整分区（简化版，原逻辑较复杂）
        if (current_critical > 0) {
            // 尝试移动关键路径前的块
            vector<int> blocks_before;
            for (int stage = 0; stage < current_critical; ++stage) {
                blocks_before.insert(blocks_before.end(),
                                   current_partition[stage].begin(),
                                   current_partition[stage].end());
            }
            
            // 添加关键路径的第一个块
            blocks_before.push_back(current_partition[current_critical][0]);
            
            // 重新分区
            vector<vector<int>> new_partition;
            reconstruct_partitions(blocks_before, prefix_sum, dp_array,
                                 blocks_before.size(), current_critical,
                                 new_partition);
            reverse(new_partition.begin(), new_partition.end());
            blocks_before.pop_back();
            
            // 完成剩余分区
            for (int stage = current_critical; stage < current_partition.size(); ++stage) {
                new_partition.push_back(current_partition[stage]);
            }
            new_partition[current_critical].erase(new_partition[current_critical].begin());
            
            // 添加到队列
            if (visited.find(new_partition) == visited.end()) {
                partitions_queue.push(new_partition);
                visited.insert(new_partition);
            }
        }
    }
    
    return best_result;
}

vector<int> PipelinePartitioner::merak_pipe(
    const vector<long long>& forward_times,
    const vector<long long>& backward_times,
    int num_stages
) {
    // 输入验证
    if (forward_times.empty() || backward_times.empty()) {
        throw invalid_argument("Input vectors cannot be empty");
    }
    
    if (forward_times.size() != backward_times.size()) {
        throw invalid_argument("Forward and backward vectors must have same size");
    }
    
    if (num_stages <= 0 || num_stages > static_cast<int>(forward_times.size())) {
        throw invalid_argument("Invalid number of pipeline stages");
    }
    
    // 准备数据
    vector<vector<long long>> block_time_mapping = {forward_times, backward_times};
    vector<int> model(forward_times.size());
    iota(model.begin(), model.end(), 0);
    
    // 执行算法
    vector<vector<int>> initial_partition = block_partition_algorithm(
        model, num_stages, block_time_mapping);
    
    vector<long long> prefix_sum;
    vector<vector<long long>> dp_array;
    calculate_prefix_sum_and_dp(model, num_stages, block_time_mapping, 
                               prefix_sum, dp_array);
    
    PartitionResult best_result = find_best_partition(
        block_time_mapping, num_stages, initial_partition, 
        prefix_sum, dp_array);
    
    // 返回每个分区的第一个块索引
    vector<int> result;
    for (const auto& partition : best_result.partition) {
        result.push_back(partition[0]);
    }
    
    return result;
}

} // namespace torchpipe

// Python绑定
PYBIND11_MODULE(autopipe, m) {
    m.doc() = "AutoPipe pipeline partition generator";
    
    m.def("pipeline", &torchpipe::PipelinePartitioner::merak_pipe, 
          "Generate pipeline partition",
          py::arg("forward_times"),
          py::arg("backward_times"),
          py::arg("num_stages"));
}