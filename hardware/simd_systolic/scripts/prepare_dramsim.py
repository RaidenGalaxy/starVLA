#!/usr/bin/env python3
"""Apply the pinned, reviewable DRAMSim3 drain/physical-burst extension."""
from pathlib import Path
import difflib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
API='''
    // SIMD evaluation extension: posted write acknowledgements are unchanged.
    void RequestDrain();
    bool IsDrained() const;
    uint64_t GetReadBursts() const;
    uint64_t GetWriteBursts() const;
'''

def prepare():
    dep=ROOT/'.deps/dramsim3';lock=json.loads((ROOT/'dependencies.lock.json').read_text())['dramsim3']
    if not dep.exists():
        subprocess.run(['git','clone',lock['url'],str(dep)],check=True)
        subprocess.run(['git','checkout',lock['commit']],cwd=dep,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dep,text=True).strip()
    if actual!=lock['commit']:raise RuntimeError('DRAMSim3 revision does not match lock')
    if 'SIMD evaluation extension' in (dep/'src/controller.h').read_text():return dep
    old={n:(dep/'src'/n).read_text() for n in ['controller.h','controller.cc','dram_system.h','dram_system.cc','memory_system.h','memory_system.cc','dramsim3.h']}
    new=old.copy()
    new['controller.h']=new['controller.h'].replace('    void ClockTick();',API+'    void ClockTick();',1)
    new['controller.h']=new['controller.h'].replace('    uint64_t clk_;','''    bool force_drain_ = false;
    uint64_t last_data_end_ = 0, physical_reads_ = 0, physical_writes_ = 0;
    uint64_t clk_;''',1)
    new['controller.cc']=new['controller.cc'].replace('#include <limits>','#include <limits>\n#include <algorithm>')
    new['controller.cc']=new['controller.cc'].replace('    trans.added_cycle = clk_;','    force_drain_ = false;\n    trans.added_cycle = clk_;',1)
    new['controller.cc']=new['controller.cc'].replace('(write_buffer_.size() > 8 && cmd_queue_.QueueEmpty()))',
        '(write_buffer_.size() > 8 && cmd_queue_.QueueEmpty()) ||\n            (force_drain_ && !write_buffer_.empty()))',1)
    new['controller.cc']=new['controller.cc'].replace('    if (cmd.IsRead()) {','''    if (cmd.IsRead()) {
        ++physical_reads_;
        last_data_end_ = std::max(last_data_end_, clk_ + uint64_t(config_.read_delay));''',1)
    new['controller.cc']=new['controller.cc'].replace('    } else if (cmd.IsWrite()) {','''    } else if (cmd.IsWrite()) {
        ++physical_writes_;
        last_data_end_ = std::max(last_data_end_, clk_ + uint64_t(config_.write_delay));''',1)
    methods='''
void Controller::RequestDrain() { force_drain_ = true; }
bool Controller::IsDrained() const {
    return read_queue_.empty() && write_buffer_.empty() && unified_queue_.empty()
        && pending_rd_q_.empty() && pending_wr_q_.empty() && return_queue_.empty()
        && cmd_queue_.QueueEmpty() && clk_ >= last_data_end_;
}
uint64_t Controller::GetReadBursts() const { return physical_reads_; }
uint64_t Controller::GetWriteBursts() const { return physical_writes_; }
'''
    new['controller.cc']=new['controller.cc'].replace('}  // namespace dramsim3',methods+'\n}  // namespace dramsim3')
    new['dram_system.h']=new['dram_system.h'].replace('    void ResetStats();','    void ResetStats();'+API,1)
    methods='''
void BaseDRAMSystem::RequestDrain() { for (auto c : ctrls_) c->RequestDrain(); }
bool BaseDRAMSystem::IsDrained() const {
    for (auto c : ctrls_) if (!c->IsDrained()) return false;
    return true;
}
uint64_t BaseDRAMSystem::GetReadBursts() const {
    uint64_t n=0; for (auto c : ctrls_) n+=c->GetReadBursts(); return n;
}
uint64_t BaseDRAMSystem::GetWriteBursts() const {
    uint64_t n=0; for (auto c : ctrls_) n+=c->GetWriteBursts(); return n;
}
'''
    new['dram_system.cc']=new['dram_system.cc'].replace('}  // namespace dramsim3',methods+'\n}  // namespace dramsim3')
    for n in ['memory_system.h','dramsim3.h']:
        new[n]=new[n].replace('    void ResetStats();','    void ResetStats();'+API,1)
    methods='''
void MemorySystem::RequestDrain() { dram_system_->RequestDrain(); }
bool MemorySystem::IsDrained() const { return dram_system_->IsDrained(); }
uint64_t MemorySystem::GetReadBursts() const { return dram_system_->GetReadBursts(); }
uint64_t MemorySystem::GetWriteBursts() const { return dram_system_->GetWriteBursts(); }
'''
    new['memory_system.cc']=new['memory_system.cc'].replace('}  // namespace dramsim3',methods+'\n}  // namespace dramsim3')
    patch=''
    for n in old:
        if old[n]==new[n]:raise RuntimeError('patch did not modify '+n)
        patch+=''.join(difflib.unified_diff(old[n].splitlines(True),new[n].splitlines(True),fromfile='a/src/'+n,tofile='b/src/'+n))
        (dep/'src'/n).write_text(new[n])
    (ROOT/'vendor/patches/dramsim3-drain.patch').write_text(patch)
    return dep

if __name__=='__main__':
    dep=prepare();out=ROOT/'.build/dramsim3';out.mkdir(parents=True,exist_ok=True)
    subprocess.run(['cmake','-S',str(dep),'-B',str(out),'-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(out),'--target','dramsim3','-j8'],check=True)
