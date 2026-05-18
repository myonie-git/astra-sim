/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMON_HH__
#define __COMMON_HH__

#include <cstdint>
#include <string>

namespace AstraSim {

typedef unsigned long long Tick;

constexpr uint64_t CLOCK_PERIOD = 1;           // 1ns
constexpr uint64_t FREQ = 1000 * 1000 * 1000;  // 1GHz

enum time_type_e { SE = 0, MS, US, NS, FS };

enum req_type_e { UINT8 = 0, BFLOAT16, FP32 };

struct timespec_t {
    time_type_e time_res;
    long double time_val;
};

struct sim_request {
    uint32_t srcRank;
    uint32_t dstRank;
    uint32_t tag;
    req_type_e reqType;
    uint64_t reqCount;
    uint32_t vnet;
    uint32_t layerNum;
};

class MetaData {
  public:
    timespec_t timestamp;
};

enum class ComType {
    None = 0,
    Reduce_Scatter,
    All_Gather,
    All_Reduce,
    All_to_All,
    All_Reduce_All_to_All
};

enum class CollectiveOptimization { Baseline = 0, LocalBWAware };//LocalBWAware：本地BW aware，AllReduce使用

enum class CollectiveBarrier { Blocking = 0, Non_Blocking };

enum class SchedulingPolicy { LIFO = 0, FIFO, EXPLICIT, None };

enum class IntraDimensionScheduling {
    FIFO = 0,
    RG,
    SmallestFirst,
    LessRemainingPhaseFirst
};

enum class InterDimensionScheduling {
    Ascending = 0, //升序传输
    OnlineGreedy, //在线贪心，一般是all reduce的组织方式
    RoundRobin, // 维度轮转，维度优先级[0,1,2]->[1,2,0]->[2,0,1]->[0,1,2]
    OfflineGreedy, //预先规划
    OfflineGreedyFlex //在OffinelineGreedy的基础上，允许动态调整chunk大小，更均衡
};

enum class InjectionPolicy { //用于控制集合通信算法向底层网络发包的激进程度
    Infinite = 0, //无限制注入数据，有包就发（Astra-sim没实现）
    Aggressive, //一次性拔该节点发给所有其他节点的数据块全部并发注入，详见
    SemiAggressive, //预留
    ExtraAggressive, //预留
    Normal //每次只允许发送 1 个 数据块。等前一个阶段（或者上一个数据块）完成并释放了资源后，才开始发送下一个。
};

enum class PacketRouting { Hardware = 0, Software };

enum class BusType { Both = 0, Shared, Mem };

enum class StreamState {
    Created = 0,
    Transferring,
    Ready,
    Executing,
    Zombie,
    Dead
};

enum class EventType {
    CallEvents = 0,
    General,
    RendezvousSend,
    RendezvousRecv,
    PacketReceived,
    PacketSent,
    Rec_Finished,
    Send_Finished,
    Processing_Finished,
    NPU_to_MA,
    MA_to_NPU,
    Consider_Process,
    Consider_Retire,
    Consider_Send_Back,
    StreamInit,
    CommProcessingFinished,
    CollectiveCommunicationFinished,
    CompFinished,
    MemLoadFinished,
    MemStoreFinished
};

}  // namespace AstraSim

#endif /* __COMMON_HH__ */
