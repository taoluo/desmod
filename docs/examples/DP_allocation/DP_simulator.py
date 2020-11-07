"""Model grocery store checkout lanes.

A grocery store checkout system is modeled. Each grocery store has one or more
checkout lanes. Each lane has a cashier that scans customers' items. Zero or
more baggers bag items after the cashier scans them. Cashiers will also bag
items if there is no bagger helping at their lane.

Several bagger assignment policies are implemented. This model helps determine
the optimal policy under various conditions. The model is also useful for
estimating bagger, checkout lane, and cashier resources needed for various
customer profiles.

"""
WITH_PROFILE = 0
import time
import heapq as hq
from argparse import ArgumentParser
from datetime import timedelta
from functools import partial
from itertools import count, tee, chain, repeat
import simpy
import sys
import timeit
from datetime import datetime

if WITH_PROFILE:
    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)

from max_min_fairness import max_min_fair_allocation
# from simpy import Container, Resource, Store, FilterStore, Event
from simpy.resources import base
from simpy.events import PENDING, EventPriority, URGENT
from simpy.resources import store
from vcd.gtkw import GTKWSave

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Union,
)
from simpy.core import BoundClass, Environment
import random
from pyDigitalWaveTools.vcd.parser import VcdParser
import yaml
from desmod.component import Component
from desmod.config import apply_user_overrides, parse_user_factors
from desmod.dot import generate_dot
from desmod.queue import Queue, FilterQueue, QueueGetEvent
from desmod.pool import Pool
from functools import wraps
from desmod.simulation import simulate, simulate_factors, simulate_many
from collections import OrderedDict
import pprint as pp
import simpy as sim

from desmod.simulation import (
    SimEnvironment,
    SimStopEvent, )

ALLOCATION_SUCCESS = "V"
ALLOCATION_FAIL = "F"
ALLOCATION_REQUEST = "allocation_request"
NEW_TASK = "new_task_created"

DP_POLICY_FCFS = "fcfs2342"
DP_POLICY_RATE_LIMIT = "rate123"
DP_POLICY_DPF = "DPF_N_234"  # task arrival-based quota accumulation
DP_POLICY_DPF_T = "DPF_T_234"  # task time-based quota accumulation
DP_POLICY_DPF_A = "DPF_A_234"  # adaptive task arrival based quota accumulation
TIMEOUT_VAL = "timeout_triggered543535"
NUMERICAL_DELTA = 1e-8
TIMEOUT_TOLERATE = 0.01
DRS_DEFAULT = 'default_def_DRS'
DRS_L2 = 'L2_norm_def_DRS'
DRS_L1 = 'L1_norm_def_DRS'
DRS_L_INF = 'L_inf_norm_def_DRS'


class NoneBlockingPutMixin(object):
    def put(self, *args, **kwargs):
        event = super(NoneBlockingPutMixin, self).put(*args, **kwargs)
        assert event.ok
        return event


class LazyAnyFilterQueue(FilterQueue):
    LAZY: EventPriority = EventPriority(99)

    def _trigger_when_at_least(self, *args, **kwargs) -> None:
        super()._trigger_when_at_least(priority=self.LAZY, *args, **kwargs)


class SingleGetterMixin(object):
    def get(self, *args, **kwargs):
        event = super(SingleGetterMixin, self).get(*args, **kwargs)
        assert len(self._get_waiters) <= 1
        return event


class NoneBlockingGetMixin(object):
    def get(self, *args, **kwargs):
        event = super(NoneBlockingGetMixin, self).get(*args, **kwargs)
        assert event.ok
        return event


def defuse(event):
    def set_defused(evt):
        evt.defused = True

    if not event.processed:
        event.callbacks.append(set_defused)


class DummyPutPool(SingleGetterMixin, NoneBlockingPutMixin, Pool):
    pass


class DummyPool(NoneBlockingGetMixin, NoneBlockingPutMixin, Pool):
    pass


class DummyPutQueue(SingleGetterMixin, NoneBlockingPutMixin, Queue):
    pass


class DummyPutLazyAnyFilterQueue(SingleGetterMixin, NoneBlockingPutMixin, LazyAnyFilterQueue):
    pass


class DummyQueue(NoneBlockingPutMixin, NoneBlockingGetMixin, Queue):
    pass


class DummyFilterQueue(NoneBlockingPutMixin, NoneBlockingGetMixin, FilterQueue):
    pass


class InsufficientDpException(RuntimeError):
    pass


class RejectDpPermissionError(RuntimeError):
    pass


class StopReleaseDpError(RuntimeError):
    pass


class DpBlockRetiredError(RuntimeError):
    pass


class ResourceAllocFail(RuntimeError):
    pass


class TaskPreemptedError(RuntimeError):
    pass


class RejectResourcePermissionError(RuntimeError):
    pass


class Top(Component):
    """The top-level component of the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = Tasks(self)
        self.resource_master = ResourceMaster(self)
        tick_seconds = self.env.config['resource_master.clock.tick_seconds']
        if self.env.config['resource_master.block.lifetime'] and self.env.config[
            'resource_master.clock.dpf_adaptive_tick']:
            tick_seconds = self.env.config['resource_master.block.lifetime'] * self.env.config['task.demand.epsilon.mice'] * 0.70
        # tick_seconds = 0.5
        self.global_clock = Clock(tick_seconds, self)
        self.add_process(self.timeout_stop)

    def timeout_stop(self):
        t0 = timeit.default_timer()
        while timeit.default_timer() - t0 < self.env.config['sim.runtime.timeout'] * 60:
            yield self.env.timeout(20)
        raise Exception('Simulation timeout %d min ' % self.env.config['sim.runtime.timeout'])

    def connect_children(self):
        self.connect(self.tasks, 'resource_master')
        self.connect(self.tasks, 'global_clock')
        self.connect(self.resource_master, 'global_clock')

    @classmethod
    def pre_init(cls, env):
        # Compose a GTKWave save file that lays-out the various VCD signals in
        # a meaningful manner. This must be done at pre-init time to allow
        # sim.gtkw.live to work.
        analog_kwargs = {
            # 'datafmt': 'dec',
            'color': 'cycle',
            'extraflags': ['analog_step'],
        }
        with open(env.config['sim.gtkw.file'], 'w') as gtkw_file:
            gtkw = GTKWSave(gtkw_file)
            gtkw.dumpfile(env.config['sim.vcd.dump_file'], abspath=False)
            gtkw.treeopen('dp_sim')
            gtkw.signals_width(300)
            with gtkw.group(f'task'):
                scope = 'tasks'
                gtkw.trace(f'{scope}.active_count', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.completion_count', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.fail_count', datafmt='dec', **analog_kwargs)

            # for i in range(env.config['grocery.num_lanes']):
            with gtkw.group(f'resource'):
                scope = 'resource_master'
                gtkw.trace(f'{scope}.cpu_pool', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.gpu_pool', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.memory_pool', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.unused_dp', datafmt='real', **analog_kwargs)
                gtkw.trace(f'{scope}.committed_dp', datafmt='real', **analog_kwargs)

    def elab_hook(self):
        # We generate DOT representations of the component hierarchy. It is
        # only after elaboration that the component tree is fully populated and
        # connected, thus generate_dot() is called here in elab_hook().
        generate_dot(self)


class Clock(Component):
    base_name = 'clock'

    def __init__(self, per_tick_seconds, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tick_period = per_tick_seconds
        self.ticking_proc = self.env.process(self.ticking())

    def ticking(self):
        while True:
            yield self.env.timeout(self.tick_period)

    @property
    def next_tick(self):
        return self.ticking_proc.target


class ResourceMaster(Component):
    """Model a grocery store with checkout lanes, cashiers, and baggers."""

    base_name = 'resource_master'
    _DP_HANDLER_INTERRUPT_MSG = "interrupted_by_dp_hanlder"
    _RSC_HANDLER_INTERRUPT_MSG = "interrupted_by_resource_hanlder"

    _RESRC_HANDLER_INTERRUPT_MSG = 'interrupted_by_resource_hanlder'
    _RESRC_RELEASE = "released_resource"
    _RESRC_PERMITED_FAIL_TO_ALLOC = "new_task_arrival"
    _RESRC_TASK_ARRIVAL = "new_task_arrival"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_connections('global_clock')
        self._retired_blocks = set()
        self.dp_policy = self.env.config["resource_master.dp_policy"]
        self.is_dp_policy_fcfs = self.dp_policy == DP_POLICY_FCFS
        self.is_dp_policy_dpf = self.dp_policy == DP_POLICY_DPF
        self.is_dp_policy_dpft = self.dp_policy == DP_POLICY_DPF_T
        self.is_dp_policy_dpfa = self.dp_policy == DP_POLICY_DPF_A

        self.is_dp_policy_rate = self.dp_policy == DP_POLICY_RATE_LIMIT

        self.unused_dp = DummyPool(self.env)

        self.auto_probe('unused_dp', vcd={'var_type': 'real'})
        self.init_blocks_ready = self.env.event()
        self.committed_dp = DummyPool(self.env)

        self.auto_probe('committed_dp', vcd={'var_type': 'real'})

        self.block_dp_storage = DummyPutQueue(self.env, capacity=float("inf"))

        self.is_cpu_needed_only = self.env.config['resource_master.is_cpu_needed_only']
        self.cpu_pool = DummyPool(self.env, capacity=self.env.config["resource_master.cpu_capacity"],
                                  init=self.env.config["resource_master.cpu_capacity"], hard_cap=True)
        self.auto_probe('cpu_pool', vcd={})

        self.memory_pool = DummyPool(self.env, capacity=self.env.config["resource_master.memory_capacity"],
                                     init=self.env.config["resource_master.memory_capacity"], hard_cap=True)
        self.auto_probe('memory_pool', vcd={})

        self.gpu_pool = DummyPool(self.env, capacity=self.env.config["resource_master.gpu_capacity"],
                                  init=self.env.config["resource_master.gpu_capacity"], hard_cap=True)
        self.auto_probe('gpu_pool', vcd={})

        self.mail_box = DummyPutQueue(self.env)
        # make sure get event happens at the last of event queue at current epoch.
        self.resource_sched_mail_box = DummyPutLazyAnyFilterQueue(self.env)
        # two types of item in mail box:
        # 1. a list of block ids whose quota get incremented,
        # 2. new arrival task id
        self.dp_sched_mail_box = DummyPutLazyAnyFilterQueue(self.env)

        self.task_state = dict()  # {task_id:{...},...,}
        # for rate limiting dp scheduling, distributed dp schedulers init in loop
        self.add_processes(self.generate_datablocks_loop)
        self.add_processes(self.allocator_frontend_loop)
        self.debug("dp allocation policy %s" % self.dp_policy)
        # waiting for dp permission
        self.dp_waiting_tasks = DummyFilterQueue(self.env,
                                                 capacity=float(
                                                     "inf"))  # {tid: DRS },  put task id and state to cal order

        # waiting for resource permission
        self.resource_waiting_tasks = DummyFilterQueue(self.env, capacity=float("inf"))

        self.auto_probe('resource_waiting_tasks', vcd={})
        self.auto_probe('dp_waiting_tasks', vcd={})

        if self.is_dp_policy_dpf:
            self.denom = self.env.config['resource_master.dp_policy.dpf.denominator']
        # for quota based policy
        if self.is_dp_policy_dpf or self.is_dp_policy_dpft or self.is_dp_policy_dpfa:
            self.add_processes(self.scheduling_dp_loop)

        self.add_processes(self.scheduling_resources_loop)

    def scheduling_resources_loop(self):

        def _permit_resource(request_tid, idle_resources):
            # non blocking
            # warning, resource allocation may fail/abort after permitted.
            # e.g. when resource handler is interrupted

            self.resource_waiting_tasks.get(filter=lambda x: x == request_tid)
            idle_resources['cpu_level'] -= self.task_state[request_tid]['resource_request']['cpu']
            if not self.is_cpu_needed_only:
                idle_resources['gpu_level'] -= self.task_state[request_tid]['resource_request']['gpu']
                idle_resources['memory_level'] -= self.task_state[request_tid]['resource_request']['memory']
            self.task_state[request_tid]['resource_permitted_event'].succeed()

        # fixme coverage
        def _reject_resource(request_tid):

            self.resource_waiting_tasks.get(filter=lambda x: x == request_tid)
            self.task_state[request_tid]['resource_permitted_event'].fail(RejectResourcePermissionError('xxxx'))

        while True:

            yield self.resource_sched_mail_box.when_any()
            # ensure the scheduler is really lazy to process getter
            assert self.env.peek() != self.env.now or self.env._queue[0][1] == LazyAnyFilterQueue.LAZY
            # ignore fake door bell, listen again
            if len(self.resource_sched_mail_box.items) == 0:
                continue

            mail_box = self.resource_sched_mail_box
            # HACK using get is slow
            msgs, mail_box.items = mail_box.items, []
            # msgs = list(mail_box.get(filter=lambda x: True).value for _ in range(len(mail_box.items)))

            resrc_release_msgs = []
            new_arrival_msgs = []
            fail_alloc_msgs = []

            for msg in msgs:
                if msg['msg_type'] == self._RESRC_TASK_ARRIVAL:
                    new_arrival_msgs.append(msg)
                elif msg['msg_type'] == self._RESRC_RELEASE:
                    resrc_release_msgs.append(msg)
                # fixme coverage
                elif msg['msg_type'] == self._RESRC_PERMITED_FAIL_TO_ALLOC:
                    fail_alloc_msgs.append(msg)
                else:
                    raise Exception('cannot identify message type')

            new_arrival_tid = [m['task_id'] for m in new_arrival_msgs]
            # should be a subset

            assert set(new_arrival_tid) <= set(self.resource_waiting_tasks.items)

            task_sched_order = None
            # optimization for case with only new arrival task(s), fcfs
            if len(new_arrival_msgs) == len(msgs):
                task_sched_order = new_arrival_tid
            # otherwise, iterate over all sleeping tasks to sched.
            else:
                task_sched_order = copy.deepcopy(self.resource_waiting_tasks.items)

            this_epoch_idle_resources = {"cpu_level": self.cpu_pool.level,
                                         "gpu_level": self.gpu_pool.level,
                                         "memory_level": self.memory_pool.level
                                         }
            # save, sched later
            fcfs_sleeping_dp_waiting_tasks = []

            for sleeping_tid in task_sched_order:
                if not self.task_state[sleeping_tid]['dp_committed_event'].triggered:
                    # will schedule dp_waiting task later
                    fcfs_sleeping_dp_waiting_tasks.append(sleeping_tid)

                # sched dp granted tasks
                # first round: sched dp-granted tasks
                elif self.task_state[sleeping_tid]['dp_committed_event'].ok:
                    if self._is_idle_resource_enough(sleeping_tid, this_epoch_idle_resources):
                        _permit_resource(sleeping_tid, this_epoch_idle_resources)
                # fixme coverage
                else:
                    assert not self.task_state[sleeping_tid]['dp_committed_event'].ok
                    raise Exception(
                        "impossible to see dp rejected task in resource_waiting_tasks. This should already happen: failed dp commit -> "
                        "interrupt resoruce handler -> dequeue resource_waiting_tasks")

            sleeping_dp_waiting_sched_order = None
            # sched dp waiting tasks
            if self.is_dp_policy_fcfs or self.is_dp_policy_rate:
                sleeping_dp_waiting_sched_order = fcfs_sleeping_dp_waiting_tasks
            elif self.is_dp_policy_dpf or self.is_dp_policy_dpft or self.is_dp_policy_dpfa:
                # smallest dominant_resource_share task first
                sleeping_dp_waiting_sched_order = sorted(fcfs_sleeping_dp_waiting_tasks, reverse=False,
                                                         key=lambda t_id: self.task_state[t_id][
                                                             'dominant_resource_share'])

            # second round: sched dp ungranted
            for sleeping_tid in sleeping_dp_waiting_sched_order:
                if self._is_idle_resource_enough(sleeping_tid, this_epoch_idle_resources):
                    _permit_resource(sleeping_tid, this_epoch_idle_resources)

    def _is_idle_resource_enough(self, tid, idle_resources):
        if idle_resources['cpu_level'] < self.task_state[tid]['resource_request']['cpu']:
            return False
        if not self.is_cpu_needed_only:
            if idle_resources['gpu_level'] < self.task_state[tid]['resource_request']['gpu']:
                return False
            if idle_resources['memory_level'] < self.task_state[tid]['resource_request'][
                'memory']:
                return False

        return True

    def scheduling_dp_loop(self):
        # calculate DR share, match, allocate,
        # update DRS if new quota has over lap with tasks
        while True:

            yield self.dp_sched_mail_box.when_any()
            dp_processed_task_idx = []
            # ensure the scheduler is really lazy to process getter, wait for all quota incremented
            assert self.env.peek() != self.env.now or self.env._queue[0][1] == LazyAnyFilterQueue.LAZY
            # ignore fake door bell, listen again
            if len(self.dp_sched_mail_box.items) == 0:
                continue
            # HACK, avoid calling slow get()
            msgs, self.dp_sched_mail_box.items = self.dp_sched_mail_box.items, []
            # mail_box = self.dp_sched_mail_box
            # msgs = list(mail_box.get(filter=lambda x: True).value for _ in range(len(mail_box.items)))

            new_arrival_tid = []
            incremented_quota_idx = set()
            msgs_amount = len(msgs)
            for m in msgs:
                if isinstance(m, int):
                    # assert m in self.dp_waiting_tasks.items
                    idx = self.dp_waiting_tasks.items.index(m, -msgs_amount - 10)
                    new_arrival_tid.append((idx, m))
                else:
                    assert isinstance(m, list)
                    incremented_quota_idx.update(m)

            this_epoch_unused_quota = [block['dp_quota'].level for block in self.block_dp_storage.items]
            # new task arrived
            for _, new_task_id in new_arrival_tid:
                assert self.task_state[new_task_id]['dominant_resource_share'] is None

            has_quota_increment = True if len(incremented_quota_idx) > 0 else False
            # update DRS of tasks if its demands has any incremented quota, or new comming tasks.
            quota_incre_upper_bound = max(incremented_quota_idx) if has_quota_increment else -1
            quota_incre_lower_bound = min(incremented_quota_idx) if has_quota_increment else -1

            if self.env.config[
                'resource_master.dp_policy.dpf_family.dominant_resource_share'] == DRS_DEFAULT and has_quota_increment:
                # iterate from newest task
                for t_id in reversed(self.dp_waiting_tasks.items):
                    # update DRS
                    this_task = self.task_state[t_id]
                    this_request = this_task['resource_request']
                    request_blk_upper_bound = this_request['block_idx'][-1]
                    request_blk_lower_bound = this_request['block_idx'][0]
                    if not (
                            request_blk_upper_bound < quota_incre_lower_bound or request_blk_lower_bound > quota_incre_upper_bound) \
                            or this_task['dominant_resource_share'] is None:
                        # optimization
                        if self.is_dp_policy_dpft:
                            # only care newest blk, has least quota
                            i = this_request['block_idx'][-1]
                            quota_amount = self.env.config["resource_master.block.init_epsilon"] - \
                                           self.block_dp_storage.items[i]['dp_container'].level
                            if quota_amount:
                                this_task['dominant_resource_share'] = this_request['epsilon'] / quota_amount
                            else:
                                this_task['dominant_resource_share'] = float('inf')

                        else:
                            resource_shares = []
                            for i in this_request['block_idx']:
                                quota_amount = self.env.config["resource_master.block.init_epsilon"] - \
                                               self.block_dp_storage.items[i]['dp_container'].level
                                if quota_amount:
                                    rs = this_request['epsilon'] / quota_amount
                                    assert rs > 0
                                    resource_shares.append(rs)
                                else:
                                    self.task_state[t_id]['dominant_resource_share'] = float('inf')
                                    break
                            else:
                                self.task_state[t_id]['dominant_resource_share'] = max(resource_shares)

            elif self.env.config[
                'resource_master.dp_policy.dpf_family.dominant_resource_share'] == DRS_DEFAULT and not has_quota_increment:
                for _, new_task_id in new_arrival_tid:
                    this_task = self.task_state[new_task_id]
                    this_request = this_task["resource_request"]
                    # block_num = len(this_request['block_idx'])
                    # omit sqrt operation for performance, maintain the correctness/monotonicity.
                    # this_task['dominant_resource_share'] = (this_request['epsilon'] ** 2) * block_num
                    resource_shares = []
                    for i in this_request['block_idx']:
                        quota_amount = self.block_dp_storage.items[i]['dp_quota'].level
                        if quota_amount:
                            rs = this_request['epsilon'] / quota_amount
                        else:
                            rs = float('inf')
                        assert rs > 0
                        resource_shares.append(rs)
                    this_task['dominant_resource_share'] = max(resource_shares)

            # other DRS are independent from arrival task
            elif self.env.config['resource_master.dp_policy.dpf_family.dominant_resource_share'] == DRS_L2:
                for _, new_task_id in new_arrival_tid:
                    this_task = self.task_state[new_task_id]
                    this_request = this_task['resource_request']
                    block_num = len(this_request['block_idx'])
                    # omit sqrt operation for performance, maintain the correctness/monotonicity.
                    this_task['dominant_resource_share'] = (this_request['epsilon'] ** 2) * block_num

            elif self.env.config['resource_master.dp_policy.dpf_family.dominant_resource_share'] == DRS_L_INF:
                for _, new_task_id in new_arrival_tid:
                    this_task = self.task_state[new_task_id]
                    this_task['dominant_resource_share'] = this_task["resource_request"]['epsilon']

            elif self.env.config['resource_master.dp_policy.dpf_family.dominant_resource_share'] == DRS_L1:
                for _, new_task_id in new_arrival_tid:
                    this_task = self.task_state[new_task_id]
                    this_request = this_task['resource_request']
                    block_num = len(this_request['block_idx'])
                    # omit sqrt operation for performance, maintain the correctness/monotonicity.
                    this_task['dominant_resource_share'] = this_request['epsilon'] * block_num

            if has_quota_increment:
                permit_dp_task_order = sorted(enumerate(self.dp_waiting_tasks.items), reverse=False,
                                              key=lambda x: self.task_state[x[1]]['dominant_resource_share'])
            # optimization for no new quota case
            elif len(new_arrival_tid) != 0:
                permit_dp_task_order = sorted(new_arrival_tid, reverse=False,
                                              key=lambda x: self.task_state[x[1]]['dominant_resource_share'])

            # iterate over tasks ordered by DRS, match quota, allocate.
            dp_permitted_task_ids = set()
            dp_permitted_blk_ids = set()
            should_grant_top_small = self.env.config['resource_master.dp_policy.dpf_family.grant_top_small']
            are_leading_tasks_ok = True
            for idx, t_id in permit_dp_task_order:
                if should_grant_top_small and (not are_leading_tasks_ok):
                    break
                this_task = self.task_state[t_id]
                this_request = this_task["resource_request"]
                # drs = this_task['dominant_resource_share']
                # assert drs is not None
                task_demand_block_idx = this_request['block_idx']

                task_demand_epsilon = this_request['epsilon']

                for b_idx in task_demand_block_idx:
                    if this_epoch_unused_quota[b_idx] + NUMERICAL_DELTA < task_demand_epsilon:
                        are_leading_tasks_ok = False
                        break
                # task is permitted
                else:
                    drs = this_task['dominant_resource_share']
                    self.debug(t_id, "DP permitted, Dominant resource share: %.3f" % drs)
                    for i in task_demand_block_idx:
                        this_epoch_unused_quota[i] -= task_demand_epsilon
                    this_task["dp_permitted_event"].succeed()
                    dp_permitted_task_ids.add(t_id)
                    dp_permitted_blk_ids.update(task_demand_block_idx)
                    dp_processed_task_idx.append(idx)

            # reject tasks after allocation
            dp_rejected_task_ids = []
            if has_quota_increment:
                for idx, t_id in enumerate(self.dp_waiting_tasks.items):
                    if t_id not in dp_permitted_task_ids:
                        this_task = self.task_state[t_id]
                        this_request = this_task["resource_request"]
                        task_demand_block_idx = this_request['block_idx']
                        task_demand_epsilon = this_request['epsilon']

                        # HACK, this is an approximation for rejection performance
                        old_demand_blk, new_demand_blk = task_demand_block_idx[0], task_demand_block_idx[-1]
                        item = self.block_dp_storage.items[old_demand_blk]
                        if item['is_retired']:
                            if this_epoch_unused_quota[b_idx] < task_demand_epsilon:
                                this_task["dp_permitted_event"].fail(DpBlockRetiredError())
                                dp_rejected_task_ids.append(t_id)
                                dp_processed_task_idx.append(idx)

                        if not self.is_dp_policy_dpft and new_demand_blk != old_demand_blk:
                            item = self.block_dp_storage.items[new_demand_blk]
                            if item['is_retired']:
                                if this_epoch_unused_quota[b_idx] < task_demand_epsilon:
                                    this_task["dp_permitted_event"].fail(DpBlockRetiredError())
                                    dp_rejected_task_ids.append(t_id)
                                    dp_processed_task_idx.append(idx)

            # dequeue all permitted and rejected waiting tasks
            # HACK avoid calling get()
            dp_processed_task_idx.sort(reverse=True)
            for i in dp_processed_task_idx:
                del self.dp_waiting_tasks.items[i]
            # .remove(tid)
            # self.dp_waiting_tasks.get(filter=lambda tid_item: tid_item == tid)

    if WITH_PROFILE:
        scheduling_dp_loop = profile(scheduling_dp_loop)

    def allocator_frontend_loop(self):
        while True:
            # loop only blocks here
            yield self.mail_box.when_any()

            for i in range(self.mail_box.size):
                get_evt = self.mail_box.get()
                msg = get_evt.value

                if msg["message_type"] == NEW_TASK:
                    assert msg["task_id"] not in self.task_state
                    self.task_state[msg["task_id"]] = dict()
                    self.task_state[msg["task_id"]]["task_proc"] = msg["task_process"]

                if msg["message_type"] == ALLOCATION_REQUEST:
                    assert msg["task_id"] in self.task_state
                    self.task_state[msg["task_id"]] = dict()
                    self.task_state[msg["task_id"]]["resource_request"] = msg
                    self.task_state[msg["task_id"]]["resource_allocate_timestamp"] = None
                    self.task_state[msg["task_id"]]["dp_commit_timestamp"] = None
                    self.task_state[msg["task_id"]]["task_completion_timestamp"] = None
                    self.task_state[msg["task_id"]]["task_publish_timestamp"] = None

                    self.task_state[msg["task_id"]]["is_dp_granted"] = False

                    self.task_state[msg["task_id"]]["resource_allocated_event"] = msg.pop("resource_allocated_event")
                    self.task_state[msg["task_id"]]["dp_committed_event"] = msg.pop("dp_committed_event")

                    # following two events are controlled by scheduling policy
                    self.task_state[msg["task_id"]]["dp_permitted_event"] = self.env.event()
                    self.task_state[msg["task_id"]]["resource_permitted_event"] = self.env.event()
                    self.task_state[msg["task_id"]]["resource_released_event"] = self.env.event()

                    # self.task_state[msg["task_id"]]["retired_blocks_in_demand"] = [] if self.is_dp_policy_dpf else None
                    self.task_state[msg["task_id"]]["dominant_resource_share"] = None

                    self.task_state[msg["task_id"]]["execution_proc"] = msg.pop("execution_proc")
                    self.task_state[msg["task_id"]]["waiting_for_dp_proc"] = msg.pop("waiting_for_dp_proc")

                    ## trigger allocation
                    self.task_state[msg["task_id"]]["handler_proc_dp"] = self.env.process(
                        self.task_dp_handler(msg["task_id"]))
                    self.task_state[msg["task_id"]]["handler_proc_resource"] = self.env.process(
                        self.task_resources_handler(msg["task_id"]))

                    self.task_state[msg["task_id"]]["accum_getters"] = dict()  # blk_idx: getter

                    msg['task_init_event'].succeed()

    def task_dp_handler(self, task_id):
        self.debug(task_id, "Task DP handler created")
        this_task = self.task_state[task_id]
        dp_committed_event = this_task["dp_committed_event"]

        resource_demand = this_task["resource_request"]
        # peek remaining DP, reject if DP is already insufficient
        for i in resource_demand["block_idx"]:
            capacity = self.block_dp_storage.items[i]["dp_container"].capacity
            if capacity + NUMERICAL_DELTA < resource_demand["epsilon"]:
                self.debug(task_id,
                           "DP is insufficient before asking dp scheduler, Block ID: %d, remain epsilon: %.3f" % (
                               i, capacity))
                if this_task["handler_proc_resource"].is_alive:
                    this_task["handler_proc_resource"].interrupt(self._DP_HANDLER_INTERRUPT_MSG)
                # inform user's dp waiting task
                dp_committed_event.fail(InsufficientDpException(
                    "DP request is rejected by handler admission control, Block ID: %d, remain epsilon: %.3f" % (
                        i, capacity)))
                return

        # getevent -> blk_idx

        if self.is_dp_policy_fcfs:
            for i in resource_demand["block_idx"]:
                # after admission control check, only need to handle numerical accuracy
                self.block_dp_storage.items[i]["dp_container"].get(
                    min(resource_demand["epsilon"], self.block_dp_storage.items[i]["dp_container"].level))

        # appliable to other policies controlled by another (centralized) processes
        if self.is_dp_policy_dpf or self.is_dp_policy_dpfa or self.is_dp_policy_dpft:
            if self.is_dp_policy_dpf or self.is_dp_policy_dpfa:

                if self.is_dp_policy_dpf:
                    self.debug(task_id, "fairshare epsilon: %0.2f" % (
                            self.env.config['resource_master.block.init_epsilon'] / self.env.config[
                        'resource_master.dp_policy.dpf.denominator']))
                this_task_retired_blocks = []
                # quota increment
                quota_increment_idx = []
                for i in resource_demand["block_idx"]:
                    self.block_dp_storage.items[i]["arrived_task_num"] += 1
                    if not self.block_dp_storage.items[i]["is_retired"]:
                        if self.is_dp_policy_dpf:
                            quota_increment = self.env.config["resource_master.block.init_epsilon"] / self.denom
                            assert quota_increment < self.block_dp_storage.items[i][
                                'dp_container'].level + NUMERICAL_DELTA
                            if -NUMERICAL_DELTA < quota_increment - self.block_dp_storage.items[i][
                                'dp_container'].level < NUMERICAL_DELTA:
                                get_amount = self.block_dp_storage.items[i]['dp_container'].level
                            else:
                                get_amount = quota_increment

                            if self.block_dp_storage.items[i]["arrived_task_num"] == self.denom:
                                self.block_dp_storage.items[i]["is_retired"] = True
                                self._retired_blocks.add(i)
                                this_task_retired_blocks.append(i)



                        elif self.is_dp_policy_dpfa:
                            # last_arrival_time = self.block_dp_storage.items[i]["last_task_arrival_time"]
                            self.block_dp_storage.items[i]["last_task_arrival_time"] = self.env.now
                            age = self.env.now - self.block_dp_storage.items[i]["create_time"]
                            if age < self.env.config['resource_master.block.lifetime']:
                                target_quota1 = age / self.env.config['resource_master.block.lifetime'] * \
                                                self.env.config[
                                                    "resource_master.block.init_epsilon"]
                                x = age / self.env.config['resource_master.block.lifetime']
                                target_quota = ((x - 1) / (-0. * x + 1) + 1) * self.env.config[
                                    "resource_master.block.init_epsilon"]
                                released_quota = self.env.config["resource_master.block.init_epsilon"] - \
                                                 self.block_dp_storage.items[i]['dp_container'].level
                                # assert -NUMERICAL_DELTA < target_quota - target_quota1 < NUMERICAL_DELTA
                                get_amount = target_quota - released_quota
                            else:
                                assert age == self.env.config['resource_master.block.lifetime']
                                # let block's callback handle
                                continue

                        if get_amount > 0:
                            self.block_dp_storage.items[i]['dp_container'].get(get_amount)
                            self.block_dp_storage.items[i]['dp_quota'].put(get_amount)

                        else:
                            assert get_amount > -NUMERICAL_DELTA
                        quota_increment_idx.append(i)
                if len(this_task_retired_blocks):
                    self.debug(task_id, 'blocks No. %s get retired.' % this_task_retired_blocks.__repr__())
                self.dp_sched_mail_box.put(quota_increment_idx)

            # dp_policy_dpft only needs enqueue
            self.dp_waiting_tasks.put(task_id)
            self.dp_sched_mail_box.put(task_id)

            unused_dp = []
            for i, block in enumerate(self.block_dp_storage.items):
                if i in resource_demand["block_idx"]:
                    unused_dp.append(-round(block['dp_quota'].level, 2))
                else:
                    unused_dp.append(round(block['dp_quota'].level, 2))
            if self.env.config['workload_test.enabled']:
                self.debug(task_id,
                           "unused dp quota after arrival: %s (negative sign denotes demanded block)" % pp.pformat(
                               unused_dp))

            try:
                t0 = self.env.now
                yield self.task_state[task_id]["dp_permitted_event"]
                self.debug(task_id, "grant_dp_permitted after ", timedelta(seconds=(self.env.now - t0)))
                for i in resource_demand["block_idx"]:
                    # get_evt_block_mapping[
                    self.block_dp_storage.items[i]["dp_quota"].get(
                        min(resource_demand["epsilon"], self.block_dp_storage.items[i]["dp_quota"].level))

            except DpBlockRetiredError as err:
                self.debug(task_id, "policy=%s, fail to acquire dp: %s" % (self.dp_policy, err.__repr__()))

                # should not issue get to quota
                assert not self.task_state[task_id]["dp_permitted_event"].ok

                # interrupt dp_waiting_proc
                if this_task["handler_proc_resource"].is_alive:
                    this_task["handler_proc_resource"].interrupt(self._DP_HANDLER_INTERRUPT_MSG)
                dp_committed_event.fail(err)
                return

        # attach containers to demanded blocks , handling rejection, return dp etc.
        elif self.is_dp_policy_rate:

            for i in resource_demand["block_idx"]:
                if self.block_dp_storage.items[i]['is_dead']:
                    # interrupt dp_waiting_proc
                    if this_task["handler_proc_resource"].is_alive:
                        this_task["handler_proc_resource"].interrupt(self._DP_HANDLER_INTERRUPT_MSG)
                    dp_committed_event.fail(
                        StopReleaseDpError("task %d rejected dp, due to block %d has stopped release" % (task_id, i)))
                    return
            else:
                for i in resource_demand["block_idx"]:
                    # need to wait get on own accum containers
                    accum_cn = DummyPutPool(self.env, capacity=resource_demand["epsilon"], init=0.0)
                    this_task["accum_getters"][i] = accum_cn.get(resource_demand["epsilon"])
                    self.block_dp_storage.items[i]["accum_containers"][task_id] = accum_cn

            all_getter = self.env.all_of(list(this_task["accum_getters"].values()))

            try:
                yield all_getter
                self.debug(task_id, "get all dp from blocks")

            except (StopReleaseDpError, InsufficientDpException) as err:
                self.debug(task_id, "policy=%s, fail to acquire dp due to" % self.dp_policy, err.__repr__())
                # interrupt dp_waiting_proc
                if this_task["handler_proc_resource"].is_alive:
                    this_task["handler_proc_resource"].interrupt(self._DP_HANDLER_INTERRUPT_MSG)
                dp_committed_event.fail(err)

                for blk_idx, get_event in this_task["accum_getters"].items():  # get_evt_block_mapping.items():
                    get_event.cancel()
                    get_event.defused = True

                    accum_container = self.block_dp_storage.items[blk_idx]['accum_containers'][task_id]
                    dp_container = self.block_dp_storage.items[blk_idx]["dp_container"]

                    if get_event.triggered and get_event.ok:

                        if dp_container.level + get_event.amount < dp_container.capacity:
                            dp_container.put(get_event.amount)
                        else:
                            # fill to full
                            # tolerate small numerical inaccuracy
                            assert dp_container.level + get_event.amount < dp_container.capacity + NUMERICAL_DELTA
                            dp_container.put(dp_container.capacity - dp_container.level)

                return

        self._commit_dp_allocation(resource_demand["block_idx"], epsilon=resource_demand["epsilon"])
        self.task_state[task_id]["is_dp_granted"] = True
        assert not dp_committed_event.triggered
        dp_committed_event.succeed()

    def task_resources_handler(self, task_id):
        self.debug(task_id, "Task resource handler created")
        # add to resources wait queue
        self.resource_waiting_tasks.put(task_id)
        self.resource_sched_mail_box.put({"msg_type": self._RESRC_TASK_ARRIVAL, "task_id": task_id})
        this_task = self.task_state[task_id]
        resource_allocated_event = this_task["resource_allocated_event"]
        resource_demand = this_task["resource_request"]
        resource_permitted_event = this_task["resource_permitted_event"]

        success_resrc_get_events = []

        try:

            yield resource_permitted_event
            get_cpu_event = self.cpu_pool.get(resource_demand["cpu"])
            success_resrc_get_events.append(get_cpu_event)

            if not self.is_cpu_needed_only:
                get_memory_event = self.memory_pool.get(resource_demand["memory"])
                success_resrc_get_events.append(get_memory_event)

                get_gpu_event = self.gpu_pool.get(resource_demand["gpu"])
                success_resrc_get_events.append(get_gpu_event)

            resource_allocated_event.succeed()
            self.task_state[task_id]["resource_allocate_timestamp"] = self.env.now

        # fixme coverage, maybe add another exception handling chain: interrupt dp handler....
        except RejectResourcePermissionError as err:
            # sched find task's dp is rejected, then fail its resource handler.
            resource_allocated_event.fail(ResourceAllocFail(err))

            return

        except simpy.Interrupt as err:
            assert err.args[0] == self._DP_HANDLER_INTERRUPT_MSG
            assert len(success_resrc_get_events) == 0
            resource_allocated_event.fail(ResourceAllocFail("Abort resource request: %s" % err))
            defuse(resource_permitted_event)
            # interrupted while permitted
            # more likely
            if not resource_permitted_event.triggered:
                assert task_id in self.resource_waiting_tasks.items
                self.resource_waiting_tasks.get(filter=lambda x: x == task_id)

                for i in self.resource_sched_mail_box.items:
                    if (i['task_id'] == task_id) and (i['msg_type'] == self._RESRC_TASK_ARRIVAL):
                        self.resource_sched_mail_box.get(
                            filter=lambda x: (x['task_id'] == task_id) and (x['msg_type'] == self._RESRC_TASK_ARRIVAL))
                pass
            # fixme coverage
            else:
                assert resource_permitted_event.ok and (not resource_permitted_event.processed)
                self.debug(task_id, "warning: resource permitted but abort to allocate due to interrupt")
                self.resource_sched_mail_box.put({"msg_type": self._RESRC_PERMITED_FAIL_TO_ALLOC, "task_id": task_id})

            return

        exec_proc = this_task['execution_proc']
        try:
            # yield task_completion_event
            yield exec_proc

        except simpy.Interrupt as err:
            assert err.args[0] == self._DP_HANDLER_INTERRUPT_MSG
            if exec_proc.is_alive:
                defuse(exec_proc)
                exec_proc.interrupt(self._DP_HANDLER_INTERRUPT_MSG)
                v = yield exec_proc | self.env.timeout(TIMEOUT_TOLERATE, TIMEOUT_VAL)
                # exec_proc should exit immeidately after interrupt
                assert v != TIMEOUT_VAL

        self.cpu_pool.put(resource_demand["cpu"])
        if not self.is_cpu_needed_only:
            self.gpu_pool.put(resource_demand["gpu"])
            self.memory_pool.put(resource_demand["memory"])

        self.debug(task_id, "Resource released")
        self.resource_sched_mail_box.put({"msg_type": self._RESRC_RELEASE, "task_id": task_id})

        return

    ## for dpft policy
    def scheduling_dp_subloop_release_quota(self, block_id):
        self.debug('block_id %d release quota subloop start at %.3f' % (block_id, self.env.now))
        this_block = self.block_dp_storage.items[block_id]
        # wait first to sync clock
        yield self.global_clock.next_tick
        if this_block["end_of_life"] <= self.env.now:
            total_dp = this_block["dp_container"].level
            this_block["dp_container"].get(total_dp)
            this_block["dp_quota"].put(total_dp)
            self.dp_sched_mail_box.put([block_id])
            this_block["is_retired"] = True
            self._retired_blocks.add(block_id)
            return

        rate_per_sec = this_block["dp_container"].level / (this_block["end_of_life"] - self.env.now)
        rate = rate_per_sec * self.global_clock.tick_period
        is_release_done = False
        while self.env.now < this_block["end_of_life"]:

            if self.env.now + self.global_clock.tick_period >= this_block["end_of_life"]:
                rate_new = rate_per_sec * (this_block["end_of_life"] - self.env.now)
                assert rate_new < this_block["dp_container"].level + NUMERICAL_DELTA
                if -NUMERICAL_DELTA < this_block["dp_container"].level - rate_new < NUMERICAL_DELTA:
                    get_amount = this_block["dp_container"].level
                    is_release_done = True
                else:
                    get_amount = rate_new
            else:
                assert rate < this_block["dp_container"].level + NUMERICAL_DELTA
                if -NUMERICAL_DELTA < this_block["dp_container"].level - rate < NUMERICAL_DELTA:
                    get_amount = this_block["dp_container"].level
                    is_release_done = True
                else:
                    get_amount = rate

            this_block["dp_container"].get(get_amount)
            assert this_block["dp_quota"].level + get_amount < this_block["dp_quota"].capacity + NUMERICAL_DELTA
            put_amount = min(get_amount, this_block["dp_quota"].capacity - this_block["dp_quota"].level)
            this_block["dp_quota"].put(put_amount)
            self.debug('block_id %d release %.3f at %.3f' % (block_id, get_amount, self.env.now))
            self.dp_sched_mail_box.put([block_id])
            if not is_release_done:
                yield self.global_clock.next_tick
            else:
                break

        assert this_block["dp_container"].level == 0
        this_block["is_retired"] = True
        self._retired_blocks.add(block_id)
        # HACK quotad increment msg, to trigger waiting task exceptions
        self.dp_sched_mail_box.put([block_id])
        self.debug('block_id %d retired with 0 DP left' % block_id)

    ## for rate limit policy
    def scheduling_dp_subloop_rate_release(self, block_id):
        is_active = False
        rate = 0
        this_block = self.block_dp_storage.items[block_id]

        # due to discretization, shift release during tick seceonds period to the start of this second.
        # therefore, last release should happen before end of life
        # wait first to sync clock
        yield self.global_clock.next_tick
        while self.env.now < this_block["end_of_life"]:

            waiting_task_cn_mapping = {tid: cn for tid, cn in this_block["accum_containers"].items()
                                       if (len(cn._get_waiters) != 0 and not cn._get_waiters[0].triggered)}

            # active waiting task may get reduced
            # copy, to avoid iterate over a changing dictionary
            for task_id in list(waiting_task_cn_mapping.keys()):
                accum_cn = waiting_task_cn_mapping[task_id]
                this_task = self.task_state[task_id]
                # compare remaining capacity and demand, conservative reject
                # fixme coverage
                if this_task["resource_request"]["epsilon"] - accum_cn.level > this_block["dp_container"].capacity:
                    waiting_task_cn_mapping.pop(task_id)
                    waiting_evt = cn._get_waiters.pop(0)
                    waiting_evt.fail(InsufficientDpException(
                        "block %d remaining uncommitted DP is insufficient for remaining ungranted dp of task %d" % (
                            block_id, task_id)))

            if len(waiting_task_cn_mapping) > 0:
                # activate block/ renew rate
                if not is_active:
                    rate_per_sec = this_block["dp_container"].level / (this_block["end_of_life"] - self.env.now)
                    rate = rate_per_sec * self.global_clock.tick_period
                    is_active = True

                # should return after wakeup
                if self.env.now + self.global_clock.tick_period >= this_block["end_of_life"]:
                    rate = rate_per_sec * (this_block["end_of_life"] - self.env.now)
                    assert rate < this_block["dp_container"].level + NUMERICAL_DELTA
                    if -NUMERICAL_DELTA < this_block["dp_container"].level - rate < NUMERICAL_DELTA:
                        get_amount = this_block["dp_container"].level
                    else:
                        get_amount = rate
                else:
                    get_amount = rate

                this_block["dp_container"].get(get_amount)

                desired_dp = {tid: cn.capacity - cn.level for tid, cn in waiting_task_cn_mapping.items()}
                # self.debug(block_id, "call max_min_fair_allocation")
                fair_allocation = max_min_fair_allocation(desired=desired_dp, capacity=get_amount)

                # all waiting task is granted by this block, return back unused dp
                if sum(fair_allocation.values()) < get_amount:
                    this_block["dp_container"].put(get_amount - sum(fair_allocation.values()))
                    is_active = False

                for tid, dp_alloc_amount in fair_allocation.items():
                    cn = this_block["accum_containers"][tid]
                    get_amount = cn._get_waiters[0].amount
                    # fill to trigger get
                    if dp_alloc_amount < get_amount - cn.level < dp_alloc_amount + NUMERICAL_DELTA:
                        cn.put(get_amount - cn.level)
                    else:
                        cn.put(dp_alloc_amount)
                    # finish put all
                    if dp_alloc_amount == desired_dp[tid]:
                        # getter should be triggered when putter is processed
                        assert cn.level == cn._get_waiters[0].amount
            else:
                is_active = False
            yield self.global_clock.next_tick

        # now >= end of life
        # wait for dp getter event processed, reject untriggered get
        yield self.env.timeout(delay=0)
        waiting_task_cn_mapping = {tid: cn for tid, cn in this_block["accum_containers"].items()
                                   if (len(cn._get_waiters) != 0 and not cn._get_waiters[0].triggered)}

        if len(waiting_task_cn_mapping) != 0:
            self.debug("block %d last period of lifetime, with waiting tasks: " % block_id,
                       list(waiting_task_cn_mapping.keys()))
            for task_id, cn in waiting_task_cn_mapping.items():
                assert len(cn._get_waiters) == 1
                # avoid getter triggered by cn
                waiting_evt = cn._get_waiters.pop(0)
                waiting_evt.fail(StopReleaseDpError(
                    "task %d rejected dp, due to block %d has stopped release" % (task_id, block_id)))
        else:
            self.debug("block %d out of life, with NO waiting task" % block_id)

        this_block["is_dead"] = True

        return

    def generate_datablocks_loop(self):
        cur_block_nr = 0
        block_id = count()
        is_static_blocks = self.env.config["resource_master.block.is_static"]
        init_amount = self.env.config["resource_master.block.init_amount"]
        while True:
            if cur_block_nr > init_amount:
                yield self.env.timeout(self.env.config["resource_master.block.arrival_interval"])

            elif cur_block_nr < init_amount:
                cur_block_nr += 1

            elif cur_block_nr == init_amount:
                cur_block_nr += 1
                self.init_blocks_ready.succeed()
                if is_static_blocks:
                    self.debug('epsilon initial static data blocks: %s' % pp.pformat(
                        [blk['dp_container'].capacity for blk in self.block_dp_storage.items]))
                    return
                else:
                    yield self.env.timeout(self.env.config["resource_master.block.arrival_interval"])

            # generate block_id
            cur_block_id = next(block_id)
            total_dp = self.env.config['resource_master.block.init_epsilon']
            new_block = DummyPool(self.env, capacity=total_dp, init=total_dp, name=cur_block_id, hard_cap=True)

            if self.is_dp_policy_dpf or self.is_dp_policy_dpft or self.is_dp_policy_dpfa:
                new_quota = DummyPool(self.env, capacity=total_dp, init=0, name=cur_block_id, hard_cap=True)
                is_retired = False
                arrived_task_num = 0

            else:
                new_quota = None
                is_retired = None
                arrived_task_num = None

            if self.is_dp_policy_rate:
                EOL = self.env.now + self.env.config['resource_master.block.lifetime']
                accum_cn_dict = dict()
                is_dead = False
            elif self.is_dp_policy_dpft or self.is_dp_policy_dpfa:
                EOL = self.env.now + self.env.config['resource_master.block.lifetime']
                accum_cn_dict = None
                is_dead = None
            else:
                EOL = None
                accum_cn_dict = None
                is_dead = None

            block_item = {
                "dp_container": new_block,
                "dp_quota": new_quota,  # for dpf policy
                # lifetime :=  # of periods from born to end
                "end_of_life": EOL,
                "accum_containers": accum_cn_dict,  # task_id: container, for rate limiting policy
                'is_dead': is_dead,
                'is_retired': is_retired,
                'arrived_task_num': arrived_task_num,
                'last_task_arrival_time': None,
                'create_time': self.env.now,
            }

            def eol_callback_gen(b_idx):
                cn = self.block_dp_storage.items[b_idx]['dp_container']
                quota = self.block_dp_storage.items[b_idx]['dp_quota']

                def eol_callback(eol):
                    lvl = cn.level
                    cn.get(lvl)
                    assert quota.level + lvl < quota.capacity + NUMERICAL_DELTA
                    lvl = min(lvl, quota.capacity - quota.level)
                    quota.put(lvl)
                    assert cn.level == 0
                    self.dp_sched_mail_box.put([b_idx])
                    self.block_dp_storage.items[b_idx]['is_retired'] = True
                    self._retired_blocks.add(b_idx)
                    self.debug('block %d EOF, move remaining dp from container to quota' % b_idx)

                return eol_callback

            self.block_dp_storage.put(block_item)
            self.unused_dp.put(total_dp)
            self.debug("new data block %d created" % cur_block_id)

            if self.is_dp_policy_rate:
                self.env.process(self.scheduling_dp_subloop_rate_release(cur_block_id))
            elif self.is_dp_policy_dpft:
                self.env.process(self.scheduling_dp_subloop_release_quota(cur_block_id))
            if self.is_dp_policy_dpfa:
                eol_event = self.env.timeout(self.env.config['resource_master.block.lifetime'])
                eol_event.callbacks.append(
                    eol_callback_gen(self.block_dp_storage.items.index(block_item)))

    def _commit_dp_allocation(self, block_idx: List[int], epsilon: float):
        """
        each block's capacity is uncommitted DP, commit by deducting capacity by epsilon.
        Args:
            block_idx:
            epsilon:

        Returns:

        """
        assert len(block_idx) > 0
        # verify able to commit
        for i in block_idx:
            this_container = self.block_dp_storage.items[i]["dp_container"]
            assert epsilon <= this_container.capacity or (
                    epsilon - this_container.capacity < NUMERICAL_DELTA and this_container.level == 0)
            # tolerate small numerical inaccuracy
            assert (this_container.level + epsilon) < this_container.capacity + NUMERICAL_DELTA

        for i in block_idx:
            this_container = self.block_dp_storage.items[i]["dp_container"]
            this_container.capacity = max(this_container.capacity - epsilon, this_container.level)
        committed_amount = min(epsilon * len(block_idx), self.unused_dp.level)
        self.unused_dp.get(committed_amount)
        self.committed_dp.put(committed_amount)
        unused_dp = []

        if self.is_dp_policy_dpf or self.is_dp_policy_dpft or self.is_dp_policy_dpfa:
            for i, block in enumerate(self.block_dp_storage.items):
                if i in block_idx:
                    unused_dp.append(-round(block['dp_quota'].level, 2))
                else:
                    unused_dp.append(round(block['dp_quota'].level, 2))
            if self.env.config['workload_test.enabled']:
                self.debug(
                    "unused dp quota after commit: %s (negative sign denotes committed block)" % pp.pformat(unused_dp))
        else:
            for i, block in enumerate(self.block_dp_storage.items):
                # uncommitted dp
                if i in block_idx:
                    unused_dp.append(-round(block['dp_container'].capacity, 2))
                else:
                    unused_dp.append(round(block['dp_container'].capacity, 2))
            if self.env.config['workload_test.enabled']:
                self.debug("unused dp after commit: %s (negative sign denotes committed block)" % pp.pformat(unused_dp))

    def get_result_hook(self, result):
        if self.env.tracemgr.vcd_tracer.enabled:
            cpu_capacity = self.env.config['resource_master.cpu_capacity']
            with open(self.env.config["sim.vcd.dump_file"]) as vcd_file:

                if vcd_file.read(1) == '':
                    return

            with open(self.env.config["sim.vcd.dump_file"]) as vcd_file:
                from functools import reduce
                vcd = VcdParser()
                vcd.parse(vcd_file)
                root_data = vcd.scope.toJson()
                assert root_data['children'][0]['children'][2]['name'] == "cpu_pool"
                # at least 11 sample
                if len(root_data['children'][0]['children'][2]['data']) == 0:
                    result['CPU_utilization%'] = 0
                    return
                elif len(root_data['children'][0]['children'][2]['data']) <= 10:
                    self.debug("WARNING: CPU change sample size <= 10")
                idle_cpu_record = map(lambda t: (t[0], eval('0' + t[1])),
                                      root_data['children'][0]['children'][2]['data'])

                idle_cpu_record = list(idle_cpu_record)
                # record should start at time 0
                if idle_cpu_record[0][0] != 0:
                    idle_cpu_record = [(0, cpu_capacity)] + idle_cpu_record

                assert self.env.config['sim.timescale'] == self.env.config['sim.duration'].split(' ')[1]
                end_tick = int(self.env.config['sim.duration'].split(' ')[0]) - 1
                if idle_cpu_record[-1][0] != end_tick:
                    idle_cpu_record = idle_cpu_record + [(end_tick, idle_cpu_record[-1][1])]

                t1, t2 = tee(idle_cpu_record)
                next(t2)
                busy_cpu_time = map(lambda t: (cpu_capacity - t[0][1]) * (t[1][0] - t[0][0]), zip(t1, t2))

            # cal over start and end
            result['CPU_utilization%'] = 100 * sum(busy_cpu_time) / (end_tick + 1) / cpu_capacity


class Tasks(Component):
    base_name = 'tasks'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_rand = self.env.rand
        self.add_connections('resource_master')
        self.add_connections('global_clock')

        self.add_process(self.generate_tasks_loop)

        self.task_unpublished_count = DummyPool(self.env)
        self.auto_probe('task_unpublished_count', vcd={})

        self.task_published_count = DummyPool(self.env)
        self.auto_probe('task_published_count', vcd={})

        self.task_sleeping_count = DummyPool(self.env)
        self.auto_probe('task_sleeping_count', vcd={})

        self.task_running_count = DummyPool(self.env)
        self.auto_probe('task_running_count', vcd={})

        self.task_completed_count = DummyPool(self.env)
        self.auto_probe('task_completed_count', vcd={})

        self.task_abort_count = DummyPool(self.env)
        self.auto_probe('task_abort_count', vcd={})

        self.task_ungranted_count = DummyPool(self.env)
        self.auto_probe('task_ungranted_count', vcd={})

        self.task_granted_count = DummyPool(self.env)
        self.auto_probe('task_granted_count', vcd={})

        self.task_dp_rejected_count = DummyPool(self.env)
        self.auto_probe('task_dp_rejected_count', vcd={})

        if self.env.tracemgr.sqlite_tracer.enabled:
            self.db = self.env.tracemgr.sqlite_tracer.db
            self.db.execute(
                'CREATE TABLE tasks '
                '(task_id INTEGER PRIMARY KEY,'
                ' start_block_id INTEGER,'
                ' num_blocks INTEGER,'
                ' epsilon REAL,'
                ' cpu INTEGER,'
                ' gpu INTEGER,'
                ' memory REAL,'
                ' start_timestamp REAL,'
                ' dp_commit_timestamp REAL,'
                ' resource_allocation_timestamp REAL,'
                ' completion_timestamp REAL,'
                ' publish_timestamp REAL'
                ')'
            )
        else:
            self.db = None

        # todo use setdefault()
        num_cpu_min = 1 if "task.demand.num_cpu.min" not in self.env.config else self.env.config[
            'task.demand.num_cpu.min']
        if self.env.config.get('task.demand.num_cpu.constant') is not None:
            # self.debug("task cpu demand is fix %d " % self.env.config['task.demand.num_cpu.constant'])
            assert isinstance(self.env.config['task.demand.num_cpu.constant'], int)
            self.cpu_dist = lambda: self.env.config['task.demand.num_cpu.constant']
        else:
            self.cpu_dist = partial(self.load_rand.randint, num_cpu_min, self.env.config['task.demand.num_cpu.max'])

        size_memory_min = 1 if "task.demand.size_memory.min" not in self.env.config else self.env.config[
            'task.demand.size_memory.min']
        self.memory_dist = partial(self.load_rand.randint, size_memory_min,
                                   self.env.config['task.demand.size_memory.max'])

        num_gpu_min = 1 if "task.demand.num_gpu.min" not in self.env.config else self.env.config[
            'task.demand.num_gpu.min']
        self.gpu_dist = partial(self.load_rand.randint, num_gpu_min, self.env.config['task.demand.num_gpu.max'])

        completion_time_min = 1 if "task.completion_time.min" not in self.env.config else self.env.config[
            'task.completion_time.min']
        if self.env.config.get('task.completion_time.constant') is not None:
            # self.debug("task completion time is fixed %d" % self.env.config['task.completion_time.constant'])
            self.completion_time_dist = lambda: self.env.config['task.completion_time.constant']
        else:
            self.completion_time_dist = partial(self.load_rand.randint, completion_time_min,
                                                self.env.config['task.completion_time.max'])
        choose_one = lambda *kargs, **kwargs: self.load_rand.choices(*kargs, **kwargs)[0]
        e_mice_fraction = self.env.config['task.demand.epsilon.mice_percentage'] / 100

        self.epsilon_dist = partial(choose_one, (self.env.config['task.demand.epsilon.mice'] , self.env.config['task.demand.epsilon.elephant'] ), (e_mice_fraction, 1 - e_mice_fraction))
        # self.load_rand.uniform, 0, self.env.config['resource_master.block.init_epsilon'] / self.env.config[
        #     'task.demand.epsilon.mean_tasks_per_block'] * 2
        # )

        # num_blocks_mu = self.env.config['task.demand.num_blocks.mu']
        # num_blocks_sigma = self.env.config['task.demand.num_blocks.sigma']
        block_mice_fraction = self.env.config['task.demand.num_blocks.mice_percentage'] / 100
        self.num_blocks_dist = partial(choose_one, (self.env.config['task.demand.num_blocks.mice'] , self.env.config['task.demand.num_blocks.elephant'] ), (block_mice_fraction, 1 - block_mice_fraction))
        # self.load_rand.normalvariate, num_blocks_mu, num_blocks_sigma
        # )

    def generate_tasks_loop(self):
        """Generate grocery store customers.

        Various configuration parameters determine the distribution of customer
        arrival times as well as the number of items each customer will shop
        for.

        """
        task_id = count()
        arrival_interval_dist = partial(
            self.load_rand.expovariate, 1 / self.env.config['task.arrival_interval']
        )

        ## wait for generating init blocks
        def init_one_task(task_id, start_block_idx, end_block_idx, epsilon, completion_time, cpu_demand, gpu_demand,
                          memory_demand):

            task_process = self.env.process(
                self.task(task_id, start_block_idx, end_block_idx, epsilon, completion_time, cpu_demand, gpu_demand,
                          memory_demand))
            new_task_msg = {"message_type": NEW_TASK,
                            "task_id": task_id,
                            "task_process": task_process,
                            }

            self.resource_master.mail_box.put(new_task_msg)

        yield self.resource_master.init_blocks_ready
        if not self.env.config['workload_test.enabled']:
            while True:
                yield self.env.timeout(arrival_interval_dist())
                t_id = next(task_id)
                # query existing data blocks
                num_stored_blocks = len(self.resource_master.block_dp_storage.items)
                assert (num_stored_blocks > 0)
                num_blocks_demand = min(max(1, round(self.num_blocks_dist())), num_stored_blocks)
                epsilon = self.epsilon_dist()

                init_one_task(t_id, start_block_idx=num_stored_blocks - num_blocks_demand,
                              end_block_idx=num_stored_blocks - 1, epsilon=epsilon,
                              completion_time=self.completion_time_dist(), cpu_demand=self.cpu_dist(),
                              gpu_demand=self.gpu_dist(),
                              memory_demand=self.memory_dist())

        else:
            assert self.env.config['workload_test.workload_trace_file']
            with open(self.env.config['workload_test.workload_trace_file']) as f:
                tasks = yaml.load(f, Loader=yaml.FullLoader)
                for t in sorted(tasks, key=lambda x: x['arrival_time']):
                    assert t['arrival_time'] - self.env.now >= 0
                    yield self.env.timeout(t['arrival_time'] - self.env.now)
                    t_id = next(task_id)
                    init_one_task(task_id=t_id, start_block_idx=t['start_block_index'],
                                  end_block_idx=t['end_block_index'],
                                  epsilon=t['epsilon'],
                                  completion_time=t['completion_time'],
                                  cpu_demand=t['cpu_demand'],
                                  gpu_demand=t['gpu_demand'],
                                  memory_demand=t['memory_demand'])

    def task(self, task_id, start_block_idx, end_block_idx, epsilon, completion_time, cpu_demand, gpu_demand,
             memory_demand):
        num_blocks_demand = end_block_idx - start_block_idx + 1
        if self.env.config['workload_test.enabled']:
            self.debug(task_id, 'DP demand epsilon=%.2f for blocks No. %s ' % (
                epsilon, list(range(start_block_idx, end_block_idx + 1))))
        else:
            self.debug(task_id, 'DP demand epsilon=%.2f for blocks No. %s ' % (
                epsilon, range(start_block_idx, end_block_idx).__repr__()))
        self.task_unpublished_count.put(1)
        self.task_ungranted_count.put(1)
        self.task_sleeping_count.put(1)

        t0 = self.env.now

        resource_allocated_event = self.env.event()
        dp_committed_event = self.env.event()
        task_init_event = self.env.event()

        def run_task(task_id, resource_allocated_event):

            assert not resource_allocated_event.triggered
            try:

                yield resource_allocated_event
                resource_alloc_time = self.resource_master.task_state[task_id]["resource_allocate_timestamp"]
                assert resource_alloc_time is not None
                resrc_allocation_wait_duration = resource_alloc_time - t0
                self.debug(task_id, 'INFO: Resources allocated after',
                           timedelta(seconds=resrc_allocation_wait_duration))

                self.task_sleeping_count.get(1)
                self.task_running_count.put(1)
            except ResourceAllocFail as err:
                task_abort_timestamp = self.resource_master.task_state[task_id][
                    "task_completion_timestamp"] = -self.env.now
                # note negative sign here
                task_preempted_duration = - task_abort_timestamp - t0
                self.debug(task_id, 'WARNING: Resource Allocation fail after',
                           timedelta(seconds=task_preempted_duration))
                self.task_sleeping_count.get(1)
                self.task_abort_count.put(1)
                return 1
            core_running_task = self.env.timeout(resource_request_msg["completion_time"])

            def post_completion_callback(event):
                task_completion_timestamp = self.resource_master.task_state[task_id][
                    "task_completion_timestamp"] = self.env.now
                task_completion_duration = task_completion_timestamp - t0
                self.debug(task_id, 'Task completed after', timedelta(seconds=task_completion_duration))
                self.task_running_count.get(1)
                self.task_completed_count.put(1)

            try:
                # running task
                yield core_running_task
                post_completion_callback(core_running_task)

                return 0

            except simpy.Interrupt as err:
                assert err.args[0] == self.resource_master._RESRC_HANDLER_INTERRUPT_MSG
                # triggered but not porocessed
                if core_running_task.triggered:
                    # same as post completion_event handling
                    assert not core_running_task.processed
                    post_completion_callback(core_running_task)

                # fixme coverage
                else:
                    task_abort_timestamp = self.resource_master.task_state[task_id][
                        "task_completion_timestamp"] = -self.env.now
                    # note negative sign here
                    task_preempted_duration = - task_abort_timestamp - t0
                    self.debug(task_id, 'Task preempted while running after',
                               timedelta(seconds=task_preempted_duration))

                    self.task_running_count.get(1)
                    self.task_abort_count.put(1)
                return 1

        def wait_for_dp(task_id, dp_committed_event):

            assert not dp_committed_event.triggered
            t0 = self.env.now
            try:
                yield dp_committed_event
                dp_committed_time = self.resource_master.task_state[task_id]["dp_commit_timestamp"] = self.env.now
                dp_committed_duration = dp_committed_time - t0
                self.debug(task_id, 'INFO: DP committed after', timedelta(seconds=dp_committed_duration))

                self.task_ungranted_count.get(1)
                self.task_granted_count.put(1)
                return 0


            except (InsufficientDpException, StopReleaseDpError, DpBlockRetiredError) as err:
                assert not dp_committed_event.ok
                dp_rejected_timestamp = self.resource_master.task_state[task_id]["dp_commit_timestamp"] = -self.env.now
                allocation_rej_duration = - dp_rejected_timestamp - t0

                self.debug(task_id, 'WARNING: DP commit fails after', timedelta(seconds=allocation_rej_duration),
                           err.__repr__())
                self.task_dp_rejected_count.put(1)
                return 1

        # listen, wait for allocation
        running_task = self.env.process(run_task(task_id, resource_allocated_event))
        waiting_for_dp = self.env.process(wait_for_dp(task_id, dp_committed_event))

        # prep allocation request,
        resource_request_msg = {
            "message_type": ALLOCATION_REQUEST,
            "task_id": task_id,
            "cpu": cpu_demand,
            "memory": memory_demand,
            "gpu": gpu_demand,
            "epsilon": epsilon,
            # todo maybe exclude EOL blocks?

            "block_idx": list(range(start_block_idx, end_block_idx + 1)),  # choose latest num_blocks_demand
            "completion_time": completion_time,
            "resource_allocated_event": resource_allocated_event,
            "dp_committed_event": dp_committed_event,
            # 'task_completion_event': task_completion_event,
            'task_init_event': task_init_event,
            "user_id": None,
            "model_id": None,
            'execution_proc': running_task,
            'waiting_for_dp_proc': waiting_for_dp,
        }
        # send allocation request, note, do it when child process is already listening
        self.resource_master.mail_box.put(resource_request_msg)
        t0 = self.env.now
        dp_grant, task_exec = yield self.env.all_of([waiting_for_dp, running_task])

        if dp_grant.value == 0:
            # verify, if dp granted, then task must be completed.
            assert task_exec.value == 0
            self.resource_master.task_state[task_id]["task_publish_timestamp"] = self.env.now
            self.task_unpublished_count.get(1)
            self.task_published_count.put(1)
            publish_duration = self.env.now - t0
            self.debug(task_id, "INFO: task get published after ", timedelta(seconds=publish_duration))
        else:
            assert dp_grant.value == 1
            publish_fail_duration = self.env.now - t0
            self.debug(task_id, "WARNING: task fail to publish after ", timedelta(seconds=publish_fail_duration))
            self.resource_master.task_state[task_id]["task_publish_timestamp"] = None

        if self.db:
            # verify iff cp commit fail <=> no publish
            if self.resource_master.task_state[task_id]["task_publish_timestamp"] is None:
                assert self.resource_master.task_state[task_id]["dp_commit_timestamp"] <= 0

            if self.resource_master.task_state[task_id]["dp_commit_timestamp"] < 0:
                assert self.resource_master.task_state[task_id]["task_publish_timestamp"] is None

            NoneType = type(None)
            assert isinstance(task_id, int)
            assert isinstance(resource_request_msg["block_idx"][0], int)
            assert isinstance(num_blocks_demand, int)
            assert isinstance(resource_request_msg["epsilon"], float)
            assert isinstance(resource_request_msg["cpu"], int)
            assert isinstance(resource_request_msg["gpu"], (int, NoneType))
            assert isinstance(resource_request_msg["memory"], (int, NoneType))
            assert isinstance(t0, (float, int))
            assert isinstance(self.resource_master.task_state[task_id]["dp_commit_timestamp"], (float, int))
            assert isinstance(self.resource_master.task_state[task_id]["resource_allocate_timestamp"],
                              (float, NoneType, int))
            assert isinstance(self.resource_master.task_state[task_id]["task_completion_timestamp"], (float, int))
            assert isinstance(self.resource_master.task_state[task_id]["task_publish_timestamp"],
                              (float, NoneType, int))

            def insert_db():
                self.db.execute(
                    'INSERT INTO tasks '
                    '(task_id, start_block_id, num_blocks, epsilon, cpu, gpu, memory, '
                    'start_timestamp, dp_commit_timestamp, resource_allocation_timestamp, completion_timestamp, publish_timestamp ) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (task_id, resource_request_msg["block_idx"][0], num_blocks_demand, resource_request_msg["epsilon"],
                     resource_request_msg["cpu"], resource_request_msg["gpu"], resource_request_msg["memory"], t0,
                     self.resource_master.task_state[task_id]["dp_commit_timestamp"],
                     self.resource_master.task_state[task_id]["resource_allocate_timestamp"],
                     self.resource_master.task_state[task_id]["task_completion_timestamp"],
                     self.resource_master.task_state[task_id]["task_publish_timestamp"]),
                )

            try:
                insert_db()
            except Exception as e:
                time.sleep(0.3)
                insert_db()
        return

    def get_result_hook(self, result):

        if not self.db:
            return
        result['succeed_tasks_total'] = self.db.execute(
            'SELECT COUNT() FROM tasks  WHERE  dp_commit_timestamp >=0'
        ).fetchone()[0]
        result['succeed_tasks_per_hour'] = result['succeed_tasks_total'] / (
                self.env.time() / 3600
        )

        # WARN not exact cal for median
        sql_duration_percentile = """
        with nt_table as
         (
             select (%s - start_timestamp) AS dp_allocation_duration, ntile(%d) over (order by (dp_commit_timestamp - start_timestamp)  desc) ntile
             from tasks
             WHERE dp_commit_timestamp >=0
         )

select avg(a)from (
         select min(dp_allocation_duration) a
         from nt_table
         where ntile = 1

         union
         select max(dp_allocation_duration) a
         from nt_table
         where ntile = 2
     )"""
        result['dp_allocation_duration_avg'] = self.db.execute(
            'SELECT AVG(dp_commit_timestamp - start_timestamp) as dur FROM tasks WHERE dp_commit_timestamp >=0 '
        ).fetchone()[0]

        result['dp_allocation_duration_min'] = self.db.execute(
            'SELECT MIN(dp_commit_timestamp - start_timestamp) as dur FROM tasks WHERE dp_commit_timestamp >=0'
        ).fetchone()[0]

        result['dp_allocation_duration_max'] = self.db.execute(
            'SELECT MAX(dp_commit_timestamp - start_timestamp) as dur FROM tasks WHERE  dp_commit_timestamp >=0'
        ).fetchone()[0]

        if result['succeed_tasks_total'] >= 2:
            result['dp_allocation_duration_Median'] = self.db.execute(
                sql_duration_percentile % ('dp_commit_timestamp', 2)
            ).fetchone()[0]

        if result['succeed_tasks_total'] >= 100:
            result['dp_allocation_duration_P99'] = self.db.execute(
                sql_duration_percentile % ('dp_commit_timestamp', 100)
            ).fetchone()[0]

        if result['succeed_tasks_total'] >= 1000:
            result['dp_allocation_duration_P999'] = self.db.execute(
                sql_duration_percentile % ('dp_commit_timestamp', 1000)
            ).fetchone()[0]

        if result['succeed_tasks_total'] >= 10000:
            result['dp_allocation_duration_P9999'] = self.db.execute(
                sql_duration_percentile % ('dp_commit_timestamp', 10000)
            ).fetchone()[0]


if __name__ == '__main__':
    import copy

    run_test_single = False
    run_test_many = False
    run_test_parallel = False
    run_factor = True
    config = {
        'workload_test.enabled': False,
        'workload_test.workload_trace_file': '/home/tao2/desmod/docs/examples/DP_allocation/workloads.yaml',
        'task.arrival_interval': 10,
        'task.demand.num_blocks.mice_percentage': 100.0,
        'task.demand.num_blocks.mice': 1,
        'task.demand.num_blocks.elephant': 20,

        'task.demand.num_blocks.mu': 20,
        'task.demand.num_blocks.sigma': 10,
        # num_blocks * 1/mean_tasks_per_block = 1/10
        'task.demand.epsilon.mice_percentage': 50.0,
        'task.demand.epsilon.mean_tasks_per_block': 15,
        'task.demand.epsilon.mice': 1e-2,
        'task.demand.epsilon.elephant': 2e-1,

        'task.completion_time.constant': 0,  # finish immediately
        # max : half of capacity
        'task.completion_time.max': 100,
        # 'task.completion_time.min': 10,
        'task.demand.num_cpu.constant': 1,  # int, [min, max]
        'task.demand.num_cpu.max': 80,
        'task.demand.num_cpu.min': 1,

        'task.demand.size_memory.max': 412,
        'task.demand.size_memory.min': 1,

        'task.demand.num_gpu.max': 3,
        'task.demand.num_gpu.min': 1,
        'resource_master.block.init_epsilon': 1.0,
        'resource_master.block.arrival_interval': 10,
        'resource_master.block.is_static': False,
        'resource_master.block.init_amount': 1,  # for block elephant demand
        # 'resource_master.dp_policy': DP_POLICY_RATE_LIMIT,
        'resource_master.block.lifetime': 10 * 10,  # policy level param
        'resource_master.dp_policy.dpf_family.dominant_resource_share': DRS_DEFAULT,  # DRS_L2
        'resource_master.dp_policy.dpf_family.grant_top_small': False,
        # only continous leading small tasks in queue are granted

        # 'resource_master.dp_policy': FCFS_POLICY,
        'resource_master.dp_policy': DP_POLICY_DPF,
        'resource_master.dp_policy.dpf.denominator': 5,
        # https://cloud.google.com/compute/docs/gpus
        # V100 VM instance
        'resource_master.is_cpu_needed_only': True,
        'resource_master.cpu_capacity': sys.maxsize,  # number of cores
        'resource_master.memory_capacity': 624,  # in GB, assume granularity is 1GB
        'resource_master.gpu_capacity': 8,  # in cards
        'resource_master.clock.tick_seconds': 25,
        'resource_master.clock.dpf_adaptive_tick': True,

        'sim.db.enable': True,
        'sim.db.persist': False,
        'sim.dot.colorscheme': 'blues5',
        'sim.dot.enable': False,
        'sim.duration': '700 s',
        'sim.runtime.timeout': 20,  # in min
        # 'sim.duration': '10 s',
        'sim.gtkw.file': 'sim.gtkw',
        'sim.gtkw.live': False,
        'sim.log.enable': False,
        "sim.log.level": "DEBUG",
        'sim.progress.enable': True,
        'sim.result.file': 'result.json',
        'sim.seed': 1234,
        'sim.timescale': 's',
        'sim.vcd.dump_file': 'sim_dp.vcd',
        'sim.vcd.enable': False,
        'sim.vcd.persist': False,
        'sim.workspace': 'trade_off_analysis/workspace_%s' % datetime.now().strftime("%m-%d-%HH-%M-%S"),
        'sim.workspace.overwrite': True,

    }
    is_single_block = False
    sample_val = (1,2,3,4,5, ) #6, 7,8,9,10)  # in 1 - 10
    b_genintvl = config['resource_master.block.arrival_interval']  # 10 sec
    # load: contention from low to high
    # 2 ^ (-1.5 ^ x)
    # mice >~= half

    # option 2
    if is_single_block:
        blk_nr_mice_pct = [100]  # all block mice
    else:
        # 99.5 - 0.27 %
        # blk_nr_mice_pct = [ (1- 2 ** (- 1.5** i )) for i in (5,4,3,2,1,0,-1,-2)]
        blk_nr_mice_pct = [95, 75, 50, 25, 5]  # all block mice

    # epsilon_mice_pct = [ (1 - 2 ** (- 1.5** i ) )for i in (5,4,3,2,1,0,-1,-2)]
    epsilon_mice_pct = [95, 75, 50, 25, 5]

    if is_single_block:
        # option 2
        t_intvl = [1, ]  # treat as time unit
    else:
        # [2, 4, 16, 64, 128, 256, 512, 1024] per b_genintvl
        t_intvl = [b_genintvl * (2 ** -i) for i in ( 1, 2, 4, 6, 7, 8,9, 10)]

    def load_filter(conf):
        # assert stress_factor in ("blk_nr","epsilon","task_arrival" )
        blk_nr_filter = lambda c: c['task.demand.epsilon.mice_percentage'] == epsilon_mice_pct[0] and c[
            'task.arrival_interval'] == t_intvl[0]
        epsilon_filter = lambda c: c['task.demand.num_blocks.mice_percentage'] == blk_nr_mice_pct[0] and c[
            'task.arrival_interval'] == t_intvl[0]
        task_arrival_filter = lambda c: c['task.demand.epsilon.mice_percentage'] == epsilon_mice_pct[0] and c[
            'task.demand.num_blocks.mice_percentage'] == blk_nr_mice_pct[0]

        # filters = {"blk_nr":blk_nr_filter, "epsilon":epsilon_filter, "task_arrival":task_arrival_filter}
        return blk_nr_filter(conf) or epsilon_filter(conf) or task_arrival_filter(conf)


    flip_coin = random.Random(x=23425453)


    def sparse_load_filter(conf):

        # idx_sum = sum(blk_nr_mice_pct.index(conf[ 'task.demand.num_blocks.mice_percentage' ]) + epsilon_mice_pct.index(conf[ 'task.demand.epsilon.mice_percentage' ]) + t_intvl.index(conf[ 'task.arrival_interval']))
        return flip_coin.randint(1, 10) in sample_val


    if is_single_block:
        # assume 1 sec interarrival
        # option 2
        #   [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        b_lifeintvl = [int(2 ** i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]
    else:
        # policy
        # [0.25, 1, 4, 16, 64, 256, // 1024]
        b_lifeintvl = [b_genintvl * (4 ** i) for i in (-1, 0, 1, 2, 3, 4)] # 5)]

    # policy, T, N
    dpf_t_factors = zip(repeat(DP_POLICY_DPF_T), b_lifeintvl, repeat(None))

    if is_single_block:
        # option 2
        #   [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        b_N_total = [int(2 ** i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]
    else:
        # [1, 4, 16, 64, 256, 1024, 4096]
        b_N_total = [(4 ** i) for i in (0, 1, 2, 3, 4, 5, 6)]

    if is_single_block:
        is_static_block = True
        init_blocks = 1
    else:
        is_static_block = False
        init_blocks =  config['task.demand.num_blocks.elephant']

    dpf_n_factors = zip(repeat(DP_POLICY_DPF), repeat(None), b_N_total)
    if is_single_block:
        sim_duration = round(max(b_lifeintvl) * 1.1)
    else:
        #  10 * 20 * 10 sec 
        sim_duration = 6 * config['task.demand.num_blocks.elephant'] * b_genintvl   

    if is_single_block:
        load_filter = lambda x: True
    else:
        load_filter = sparse_load_filter

    factors = [(['sim.duration'], [['%d s' % sim_duration]]),
               (['resource_master.block.is_static'], [[is_static_block]]),
               (['resource_master.block.init_amount'], [[init_blocks]]),
               (['task.demand.num_blocks.mice_percentage'], [[i] for i in blk_nr_mice_pct]),
               (['task.demand.epsilon.mice_percentage'], [[i] for i in epsilon_mice_pct]),
               (['task.arrival_interval'], [[i] for i in t_intvl]),
               (['resource_master.dp_policy', 'resource_master.block.lifetime',
                 'resource_master.dp_policy.dpf.denominator'],
                list(chain([[DP_POLICY_FCFS, None, None], ], dpf_t_factors, dpf_n_factors))),  #
               ]

    parser = ArgumentParser()
    parser.add_argument(
        '--set',
        '-s',
        nargs=2,
        metavar=('KEY', 'VALUE'),
        action='append',
        default=[],
        dest='config_overrides',
        help='Override config KEY with VALUE expression',
    )
    parser.add_argument(
        '--factor',
        '-f',
        nargs=2,
        metavar=('KEYS', 'VALUES'),
        action='append',
        default=[],
        dest='factors',
        help='Add multi-factor VALUES for KEY(S)',
    )
    args = parser.parse_args()
    apply_user_overrides(config, args.config_overrides)
    # factors = parse_user_factors(config, args.factors)
    if factors and run_factor:
        simulate_factors(config, factors, Top, config_filter=load_filter)

    if run_test_single:
        pp.pprint(config)
        simulate(copy.deepcopy(config), Top)

    task_configs = {}
    scheduler_configs = {}
    config1 = copy.deepcopy(config)
    # use rate limit by default
    config1["resource_master.dp_policy"] = DP_POLICY_RATE_LIMIT
    # use random
    config1['task.completion_time.constant'] = None
    config1['task.demand.num_cpu.constant'] = None
    config1["resource_master.is_cpu_needed_only"] = False

    demand_block_num_baseline = config1['task.demand.epsilon.mean_tasks_per_block'] * config1['task.arrival_interval'] / \
                                config1['resource_master.block.arrival_interval']
    demand_block_num_low_factor = 1
    task_configs["high_cpu_low_dp"] = {'task.demand.num_cpu.max': config1["resource_master.cpu_capacity"],
                                       'task.demand.num_cpu.min': 2,
                                       'task.demand.epsilon.mean_tasks_per_block': 200,
                                       'task.demand.num_blocks.mu': demand_block_num_baseline * demand_block_num_low_factor,
                                       # 3
                                       'task.demand.num_blocks.sigma': demand_block_num_baseline * demand_block_num_low_factor,
                                       }
    task_configs["low_cpu_high_dp"] = {'task.demand.num_cpu.max': 2,
                                       'task.demand.num_cpu.min': 1,
                                       'task.demand.epsilon.mean_tasks_per_block': 8,
                                       'task.demand.num_blocks.mu': demand_block_num_baseline * demand_block_num_low_factor * 4,
                                       # 45
                                       'task.demand.num_blocks.sigma': demand_block_num_baseline * demand_block_num_low_factor * 4 / 3,
                                       # 5
                                       }

    scheduler_configs["fcfs_policy"] = {'resource_master.dp_policy': DP_POLICY_FCFS}

    scheduler_configs["rate_policy_slow_release"] = {'resource_master.dp_policy': DP_POLICY_RATE_LIMIT,
                                                     'resource_master.block.lifetime': config1[
                                                                                           'task.arrival_interval'] * 10 * 5
                                                     # 500
                                                     }
    scheduler_configs["rate_policy_fast_release"] = {'resource_master.dp_policy': DP_POLICY_RATE_LIMIT,
                                                     'resource_master.block.lifetime': config1[
                                                                                           'task.arrival_interval'] * 5}  # 50
    dp_factor_names = set()
    for sched_conf_k, sched_conf_v in scheduler_configs.items():
        for conf_factor in sched_conf_v:
            dp_factor_names.add(conf_factor)

    for task_conf_k, task_conf_v in task_configs.items():
        for conf_factor in task_conf_v:
            dp_factor_names.add(conf_factor)
    dp_factor_names = list(dp_factor_names)
    dp_factor_values = []

    configs = []

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_fcfs"
    config2["resource_master.dp_policy"] = DP_POLICY_FCFS
    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_dpf"
    config2["resource_master.dp_policy"] = DP_POLICY_DPF
    config2["resource_master.dp_policy.dpf.denominator"] = 30  # inter arrival is 10 sec
    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_dpft"
    config2["resource_master.dp_policy"] = DP_POLICY_DPF_T
    config2["resource_master.block.lifetime"] = 300
    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_rate_limiting"
    config2["resource_master.dp_policy"] = DP_POLICY_RATE_LIMIT
    config2["resource_master.block.lifetime"] = 300
    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_dpfa"
    config2["resource_master.dp_policy"] = DP_POLICY_DPF_A
    config2["resource_master.block.lifetime"] = 300
    configs.append(config2)

    if run_test_many:
        for c in configs:
            # c['sim.seed'] = time.time()
            pp.pprint(c)
            simulate(copy.deepcopy(c), Top)

    if run_test_parallel:
        simulate_many(copy.deepcopy(configs), Top)

    for sched_conf_k, sched_conf_v in scheduler_configs.items():
        for task_conf_k, task_conf_v in task_configs.items():
            new_config = copy.deepcopy(config1)
            new_config.update(sched_conf_v)
            new_config.update(task_conf_v)
            workspace_name = "workspace_%s-%s" % (sched_conf_k, task_conf_k)
            new_config["sim.workspace"] = workspace_name
            configs.append(new_config)

    for i, c in enumerate(configs):
        try:
            # simulate(c, Top)
            pass
        except:
            print(i)
            print(c)

    # debug a config
    # for cfg in configs:
    #     # if "slow_release-low_cpu_high_dp"in cfg["sim.workspace"]:
    #     #     simulate(cfg, Top)
    #
    #     if "fast_release-low_cpu_high_dp"in cfg["sim.workspace"]:
    #         simulate(cfg, Top)
