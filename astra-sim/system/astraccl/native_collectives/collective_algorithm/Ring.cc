/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.hh"

#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"

using namespace AstraSim;

Ring::Ring(ComType type,
           int id,
           RingTopology* ring_topology,
           uint64_t data_size,
           RingTopology::Direction direction,
           InjectionPolicy injection_policy)
    : Algorithm() {
    this->comType = type; //通信类型：如All_gather
    this->id = id; //当前rank
    this->logical_topo = ring_topology;
    this->data_size = data_size; 
    this->direction = direction; //传输，顺时针还是逆时针
    this->nodes_in_ring = ring_topology->get_nodes_in_ring(); //环上节点数
    this->curr_receiver = ring_topology->get_receiver(id, direction); //当前rank的接收者
    this->curr_sender = ring_topology->get_sender(id, direction); //当前rank的发送者
    this->parallel_reduce = 1; //默认并行度
    this->injection_policy = injection_policy; //注入策略
    this->total_packets_sent = 0;
    this->total_packets_received = 0;
    this->free_packets = 0;
    this->zero_latency_packets = 0;
    this->non_zero_latency_packets = 0;
    this->toggle = false;
    this->name = Name::Ring;
    if (ring_topology->get_dimension() == RingTopology::Dimension::Local) {//根据是Local还是Remote选择Membus的建模方式 
        transmition = MemBus::Transmition::Fast; //固定10个tick的延迟
    } else {
        transmition = MemBus::Transmition::Usual; //用配置的communication_delay
    }
    switch (type) {
    case ComType::All_Reduce:
        stream_count = 2 * (nodes_in_ring - 1); //ring all reduce = reduce_scatter + all gather
        break;
    case ComType::All_to_All:
        this->stream_count = ((nodes_in_ring - 1) * nodes_in_ring) / 2;
        switch (injection_policy) {
        case InjectionPolicy::Aggressive:
            this->parallel_reduce = nodes_in_ring - 1;
            break;
        case InjectionPolicy::Normal:
            this->parallel_reduce = 1;
            break;
        default:
            this->parallel_reduce = 1;
            break;
        }
        break;
    default:
        stream_count = nodes_in_ring - 1;
    }
    if (type == ComType::All_to_All || type == ComType::All_Gather) {
        max_count = 0; //max_count 是一个“批次计数器/节流计数器”：用来控制 什么时候把已经锁住的 packet（locked_packets）打包并真正发到 MemBus 上 https://www.yuque.com/u953085/fk8874/ymu7s3t9x2cflwic/edit?toc_node_uuid=M9QMtTMWZxVC8TyF
    } else {
        max_count = nodes_in_ring - 1;
    }
    remained_packets_per_message = 1;
    remained_packets_per_max_count = 1;
    switch (type) {
    case ComType::All_Reduce:
        this->final_data_size = data_size;  //每个Rank上的数据量
        this->msg_size = data_size / nodes_in_ring; // Msg_size: send/recv的字节数
        break;
    case ComType::All_Gather:
        this->final_data_size = data_size * nodes_in_ring;
        this->msg_size = data_size;
        break;
    case ComType::Reduce_Scatter:
        this->final_data_size = data_size / nodes_in_ring;
        this->msg_size = data_size / nodes_in_ring;
        break;
    case ComType::All_to_All:
        this->final_data_size = data_size;
        this->msg_size = data_size / nodes_in_ring;
        break;
    default:;
    }
}

int Ring::get_non_zero_latency_packets() {
    return (nodes_in_ring - 1) * parallel_reduce * 1;
}

void Ring::run(EventType event, CallData* data) {
    if (event == EventType::General) { //General事件，可以多发一个包了
        free_packets += 1;
        ready();
        iteratable();
    } else if (event == EventType::PacketReceived) { //收到上游来的包了
        total_packets_received++;
        insert_packet(nullptr);
    } else if (event == EventType::StreamInit) { // 首次启动
        for (int i = 0; i < parallel_reduce; i++) {
            insert_packet(nullptr);//插入parallel_reduce个包，由InjectionPolicy设置，Normal为1 
        }
    }
}

void Ring::release_packets() {//把刚才在 insert_packet 里生成并“锁住（locked）”的数据包，打包成一个 PacketBundle，并交给本地的内存总线（MemBus）去模拟数据在节点内部的搬运和计算延迟，并触发General，返回run。
    for (auto packet : locked_packets) {
        packet->set_notifier(this);
    }
    if (NPU_to_MA == true) {
        (new PacketBundle(stream->owner, stream, locked_packets, processed,
                          send_back, msg_size, transmition))
            ->send_to_MA();
    } else {
        (new PacketBundle(stream->owner, stream, locked_packets, processed,
                          send_back, msg_size, transmition))
            ->send_to_NPU();
    }
    locked_packets.clear();
}

void Ring::process_stream_count() {
    if (remained_packets_per_message > 0) {
        remained_packets_per_message--;
    }
    if (id == 0) {
    }
    if (remained_packets_per_message == 0 && stream_count > 0) {
        stream_count--;
        if (stream_count > 0) {
            remained_packets_per_message = 1;
        }
    }
    if (remained_packets_per_message == 0 && stream_count == 0 &&
        stream->state != StreamState::Dead) {
        stream->changeState(StreamState::Zombie);
    }
}

void Ring::process_max_count() {
    if (remained_packets_per_max_count > 0) {
        remained_packets_per_max_count--;
    }
    if (remained_packets_per_max_count == 0) {
        max_count--;
        release_packets();
        remained_packets_per_max_count = 1;
    }
}

void Ring::reduce() {
    process_stream_count();
    packets.pop_front();
    free_packets--;
    total_packets_sent++;
}

bool Ring::iteratable() {
    if (stream_count == 0 &&
        free_packets == (parallel_reduce * 1)) {  // && not_delivered==0
        exit();
        return false;
    }
    return true;
}

void Ring::insert_packet(Callable* sender) {
    if (zero_latency_packets == 0 && non_zero_latency_packets == 0) { //起始包 & 中继包，在本轮中的额度都为0了，轮的概念：https://www.yuque.com/u953085/fk8874/rnva9d3ze3115zf8/edit?toc_node_uuid=M9QMtTMWZxVC8TyF
        zero_latency_packets = parallel_reduce * 1; //重新配置两种包的额度
        non_zero_latency_packets =
            get_non_zero_latency_packets();  //(nodes_in_ring-1)*parallel_reduce*1;
        toggle = !toggle;
    }
    if (zero_latency_packets > 0) {
        packets.push_back(MyPacket(//创建一个新包，并指定接收者和发送者
            stream->current_queue_id, curr_sender,
            curr_receiver));  // vnet Must be changed for alltoall topology
        packets.back().sender = sender; //.back() 最后一个元素
        locked_packets.push_back(&packets.back());
        processed = false;
        send_back = false;
        NPU_to_MA = true;
        process_max_count();
        zero_latency_packets--;
        return;
    } else if (non_zero_latency_packets > 0) {
        packets.push_back(MyPacket(
            stream->current_queue_id, curr_sender,
            curr_receiver));  // vnet Must be changed for alltoall topology
        packets.back().sender = sender;
        locked_packets.push_back(&packets.back());
        if (comType == ComType::Reduce_Scatter ||
            (comType == ComType::All_Reduce && toggle)) {
            processed = true;
        } else {
            processed = false;
        }
        if (non_zero_latency_packets <= parallel_reduce * 1) {
            send_back = false;
        } else {
            send_back = true;
        }
        NPU_to_MA = false;
        process_max_count();
        non_zero_latency_packets--;
        return;
    }
    Sys::sys_panic("should not inject nothing!");
}

bool Ring::ready() {
    if (stream->state == StreamState::Created ||
        stream->state == StreamState::Ready) { //stream正式进行通信，将状态改为ready
        stream->changeState(StreamState::Executing);
    }
    if (packets.size() == 0 || stream_count == 0 || free_packets == 0) {
        return false; //如果无法继续推进包的发送，就退出
    }
    MyPacket packet = packets.front();
    sim_request snd_req;
    snd_req.srcRank = id;
    snd_req.dstRank = packet.preferred_dest;
    snd_req.tag = stream->stream_id;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    stream->owner->front_end_sim_send( //调用sim_send发包
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_dest,
        stream->stream_id, &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent,
        nullptr);  // stream_id+(packet.preferred_dest*50)
    sim_request rcv_req;//构造recv请求
    rcv_req.vnet = this->stream->current_queue_id;
    RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
        stream, stream->owner->id, EventType::PacketReceived,
        packet.preferred_vnet, packet.stream_id);
    stream->owner->front_end_sim_recv(
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_src,
        stream->stream_id, &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent,
        ehd);  // stream_id+(owner->id*50)
    reduce(); ///状态推进
    return true;
}

void Ring::exit() {
    if (packets.size() != 0) {
        packets.clear();
    }
    if (locked_packets.size() != 0) {
        locked_packets.clear();
    }
    stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
    return;
}
