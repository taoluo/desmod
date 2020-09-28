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
from argparse import ArgumentParser
from datetime import timedelta
from functools import partial
from itertools import count, tee
import simpy
from simpy import Container, Resource, Store, FilterStore, Event
from simpy.events import PENDING
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

from desmod.component import Component
from desmod.config import apply_user_overrides, parse_user_factors
from desmod.dot import generate_dot
from desmod.queue import Queue, FilterQueue
from desmod.pool import Pool
from functools import wraps
from desmod.simulation import simulate, simulate_factors, simulate_many
from collections import OrderedDict
from pprint import pprint

ALLOCATION_SUCCESS = "V"
ALLOCATION_FAIL = "F"
ALLOCATION_REQUEST = "allocation_request"
NEW_TASK = "new_task_created"
# TASK_COMPLETION="task_completion"
TASK_RESOURCE_RELEASED = "resource_release"

POLICY_FCFS = "fcfs2342"
POLICY_RATE_LIMIT = "rate123"
POLICY_DPF = "dynamic_DPF234"
TIMEOUT_VAL = "timeout_triggered543535"
NUMERICAL_DELTA = 1e-12
TIMEOUT_TOLERATE = 0.01

class Top(Component):
    """The top-level component of the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = Tasks(self)
        self.resource_master = ResourceMaster(self)

    def connect_children(self):
        self.connect(self.tasks, 'resource_master')

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


class InsufficientDpException(Exception):
    pass


class TaskPreemptedError(Exception):
    pass


class RejectAllocException(Exception):
    pass


class ShouldTriggeredImmediatelyException(Exception):
    pass


class ResourceMaster(Component):
    """Model a grocery store with checkout lanes, cashiers, and baggers."""

    base_name = 'resource_master'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unused_dp = Pool(self.env)
        self.auto_probe('unused_dp', vcd={'var_type': 'real'})
        self.init_blocks_ready = self.env.event()
        self.committed_dp = Pool(self.env)

        self.auto_probe('committed_dp', vcd={'var_type': 'real'})

        self.block_dp_storage = Queue(self.env, capacity=float("inf"))
        # self.block_waiting_containers = [] # [[],[],...,[] ], put each task's waiting DP container in sublist
        self.is_cpu_needed_only = self.env.config['resource_master.is_cpu_needed_only']
        self.cpu_pool = Pool(self.env, capacity=self.env.config["resource_master.cpu_capacity"],
                             init=self.env.config["resource_master.cpu_capacity"], hard_cap=True)
        self.auto_probe('cpu_pool', vcd={})

        self.memory_pool = Pool(self.env, capacity=self.env.config["resource_master.memory_capacity"],
                                init=self.env.config["resource_master.memory_capacity"], hard_cap=True)
        self.auto_probe('memory_pool', vcd={})

        self.gpu_pool = Pool(self.env, capacity=self.env.config["resource_master.gpu_capacity"],
                             init=self.env.config["resource_master.gpu_capacity"], hard_cap=True)
        self.auto_probe('gpu_pool', vcd={})

        self.mail_box = Queue(self.env)
        self.task_state = dict()  # {task_id:{...},...,}
        self.add_processes(self.generate_datablocks_loop)
        self.add_processes(self.allocator_init_loop)
        self.period = 1
        self.policy = self.env.config["resource_master.dp_policy"]
        self.debug("dp allocation policy %s" % self.policy)
        # get by task id
        self.dp_waiting_tasks = FilterQueue(self.env,
                                            capacity=float("inf"))  # {tid: DRS },  put task id and state to cal order
        self.resource_waiting_tasks = FilterQueue(self.env, capacity=float("inf"))
        self.auto_probe('resource_waiting_tasks', vcd={})
        self.auto_probe('dp_waiting_tasks', vcd={})

        self.is_policy_fcfs = self.policy == POLICY_FCFS
        self.is_policy_dpf = self.policy == POLICY_DPF
        self.is_policy_rate = self.policy == POLICY_RATE_LIMIT
        if self.is_policy_dpf:
            self.add_processes(self.scheduling_dp_loop)
            self.denom = self.env.config['resource_master.dp_policy.dpf.denominator']

        self.add_processes(self.scheduling_resources_loop)

    def clock_tick(self):
        return self.env.timeout(self.period)

    def scheduling_resources_loop(self):
        # permitted is a task state after its resources are permitted, before its resources are released
        # {tid,...,}
        resource_permitted_tasks = set()
        dp_waiting_tasks = set()
        # dp_committed_events = dict()
        resource_released_events = dict()
        resrc_request_arrival_event = self.env.event()
        self.resource_waiting_tasks._put_hook = lambda: resrc_request_arrival_event.succeed()

        def _grant_resource_permission(request_tid):
            # todo more tear down operation
            resource_permitted_tasks.add(request_tid)
            # if permitted task is dp waiting, remove from dp waiting list
            dp_waiting_tasks.discard(request_tid)
            # assert self.task_state[request_tid]['resource_released_event'] not in resource_released_events
            # resource_released_events[self.task_state[request_tid]['resource_released_event']] = request_tid
            self.task_state[request_tid]['resource_permitted_event'].succeed()
            yield from _should_return_immediately(self.env, self.task_state[request_tid]['resource_allocated_event'])
            yield from _should_return_immediately(self.env, self.resource_waiting_tasks.get(filter=lambda x: x == request_tid))



        while True:
            sched_listened_events = dict()

            # new demand
            sched_listened_events[resrc_request_arrival_event] = None

            # existing demand state/priority change
            # only care about sleeping tasks' dp state change
            dp_committed_events = {self.task_state[tid]['dp_committed_event']: tid for tid in
                                   self.resource_waiting_tasks.items if
                                   not self.task_state[tid]['dp_committed_event'].triggered}
            # todo alternative is filter by dp_waiting_tasks
            dp_waiting_tasks2 = set( (tid for evt,tid in dp_committed_events.items()) )
            ## fixme !!! condition is violated for long simulation.
            assert dp_waiting_tasks2 == dp_waiting_tasks

            sched_listened_events.update(dp_committed_events)

            # new supply
            resource_released_events = {self.task_state[tid]['resource_released_event']: tid for tid in
                                        resource_permitted_tasks }
            sched_listened_events.update(resource_released_events)

            sched_listener = self.env.any_of(sched_listened_events)

            try:
                succeed_events = yield sched_listener
                succeed_dp_committed_events = []
                succeed_arrival_events = []
                succeed_resource_released_events = []
                for evt in succeed_events.events:
                    assert evt in sched_listened_events
                    if evt in resource_released_events:
                        succeed_resource_released_events.append(evt)
                    elif evt is resrc_request_arrival_event:
                        succeed_arrival_events.append(evt)
                    else:
                        assert evt in dp_committed_events
                        succeed_dp_committed_events.append(evt)
                assert len(succeed_arrival_events) <= 1
                # overview of events-handling order:
                # step1: if any, update dp_waiting/dp_granted state of waiting tasks
                # step2: if there is a new task arrival and no resource was released, try to schedule new task.
                # step3: after step1 and step2, scheduler's state is updated and ready. if resource was released, \
                #        then make scheduling decision here.


                # step1: if any, update dp_waiting/dp_granted state of waiting tasks
                # new dp granted, do nothing
                # todo maybe interrupt some ungranted running tasks to trigger resrc release event
                for evt in succeed_dp_committed_events:

                    dp_committed_tid = dp_committed_events[evt]
                    dp_waiting_tasks.remove(dp_committed_tid)

                # step2: if there is a new task arrival and no resource was released, try to schedule new task.
                # secondary scheduling decision branch, FCFS when new task arrives and idle resource is sufficient.
                if len(succeed_arrival_events) != 0:
                    assert succeed_arrival_events[0] is resrc_request_arrival_event
                    resrc_request_arrival_event = self.env.event()
                    self.resource_waiting_tasks._put_hook = lambda: resrc_request_arrival_event.succeed()

                    # check if there exists enough resources to be scheduled
                    # assert self.resource_waiting_tasks.items[-1]['dp_state'] is None
                    request_tid = self.resource_waiting_tasks.items[-1]

                    should_try_sched = True if len(succeed_resource_released_events)==0 else False
                    if self.task_state[request_tid]['dp_committed_event'].triggered:
                        if not self.task_state[request_tid]['dp_committed_event'].ok:
                            # dp rejected, dequeue and do nothing
                            yield from _should_return_immediately(self.env, self.resource_waiting_tasks.get(filter=lambda x: x == request_tid))
                            self.debug(request_tid, 'dp request is rejected upon resource request arrival, resource sched ignore the task')
                        # dp already granted
                        elif should_try_sched and self._is_idle_resource_enough(request_tid):
                            yield from _grant_resource_permission(request_tid)

                    # dp waiting
                    else:
                        if should_try_sched and self._is_idle_resource_enough(request_tid):
                            yield from _grant_resource_permission(request_tid)
                        else:
                            dp_waiting_tasks.add(request_tid)

                # step3: after step1 and step2, scheduler's state is updated and ready. if resource was released, \
                #        then make scheduling decision here.
                for evt in succeed_resource_released_events:
                    released_tid = resource_released_events[evt]
                    # resource_released_events.pop(evt)
                    resource_permitted_tasks.remove(released_tid)

                # primary scheduling decision, when resource is released
                if len(succeed_resource_released_events) != 0:
                    dp_waiting_tasks_temp = set()
                    for sleeping_tid in copy.deepcopy(self.resource_waiting_tasks.items):

                        if not self.task_state[sleeping_tid]['dp_committed_event'].triggered:
                            # will schedule dp_waiting task later
                            dp_waiting_tasks_temp.add(sleeping_tid)

                        # sched dp granted tasks
                        elif self.task_state[sleeping_tid]['dp_committed_event'].ok:

                            if self._is_idle_resource_enough(sleeping_tid):
                                yield from _grant_resource_permission(sleeping_tid)

                        else:
                            raise Exception("should already be dequeued in caught exception.")

                    assert dp_waiting_tasks == dp_waiting_tasks_temp
                    resource_waiting_tid_set = set(self.resource_waiting_tasks.items)

                    # sched dp waiting tasks
                    for t in sorted(self.dp_waiting_tasks.items, reverse=False, key=lambda x: x['dominant_resource_share']):
                        dp_waiting_tid = t['task_id']
                        if dp_waiting_tid in resource_waiting_tid_set and self._is_idle_resource_enough(sleeping_tid):
                            yield from _grant_resource_permission(dp_waiting_tid)

            except (InsufficientDpException, RejectAllocException) as err:
                dp_rejected_tid = err.args[0]
                dp_waiting_tasks.remove(dp_rejected_tid)
                yield from _should_return_immediately(self.env, self.resource_waiting_tasks.get(filter=lambda x: x == dp_rejected_tid))
                self.debug(dp_rejected_tid, "dp request is rejected, get dequeued from waiting resource requests")

    def _is_idle_resource_enough(self, tid):
        if self.cpu_pool.level < self.task_state[tid]['resource_request']['cpu']:
            return False
        if not self.is_cpu_needed_only:
            if self.gpu_pool.level < self.task_state[tid]['resource_request']['gpu']:
                return False
            if self.memory_pool.level < self.task_state[tid]['resource_request'][
                'memory']:
                return False

        return True


    def scheduling_dp_loop(self):
        # calculate DR share, match, allocate,
        # update DRS if new quota has over lap with tasks
        while True:
            dp_request_arrival_event = self.env.event()
            self.dp_waiting_tasks._put_hook = lambda: dp_request_arrival_event.succeed()
            yield dp_request_arrival_event

            # new task arrived
            new_task_dict = self.dp_waiting_tasks.items[-1]
            assert new_task_dict['dominant_resource_share'] is None
            new_task_id = new_task_dict['task_id']

            incremented_quota_idx = set(self.task_state[new_task_id]["resource_request"]['block_idx']) - set(
                self.task_state[new_task_id]['retired_blocks'])

            # should update DRS of last new task if it demands any non-retired block
            for t in self.dp_waiting_tasks.items:
                t_id = t['task_id']
                # update DRS
                if set(self.task_state[t_id]["resource_request"]['block_idx']) & incremented_quota_idx:
                    resource_shares = []
                    for i in self.task_state[new_task_id]["resource_request"]['block_idx']:
                        rs = self.task_state[t_id]["resource_request"]['epsilon'] / (
                                self.env.config["resource_master.block.init_epsilon"] -
                                self.block_dp_storage.items[i]['dp_container'].level)
                        resource_shares.append(rs)
                t['dominant_resource_share'] = max(resource_shares)

            # iterate over tasks ordered by DRS, match quota, allocate.
            interrupted_task_ids = []
            # ordered_DRS = copy.deepcopy(sorted(self.dp_waiting_tasks.items, reverse=False, key=lambda x: x['dominant_resource_share']))
            for t in sorted(self.dp_waiting_tasks.items, reverse=False, key=lambda x: x['dominant_resource_share']):
                assert t['dominant_resource_share'] is not None
                t_id = t['task_id']
                this_task = self.task_state[t['task_id']]
                # self.debug(t_id, "DRS: %.3f" % t['dominant_resource_share'])
                for b_idx in range(len(self.block_dp_storage.items)):
                    if self.block_dp_storage.items[b_idx]['dp_quota'].level < self.task_state[t_id]["resource_request"][
                        'epsilon']:
                        self.debug(t_id, "current quota is insufficient ")
                        if self.block_dp_storage.items[b_idx]['is_retired'] :
                            assert not this_task["handler_proc_dp"].triggered
                            this_task["handler_proc_dp"].interrupt(
                                "dp allocation is rejected, because retired block %d has %.3f DP left < demanded epsilon %.3f" % (b_idx,self.block_dp_storage.items[b_idx]['dp_quota'].level, self.task_state[t_id]["resource_request"][
                            'epsilon']))
                            interrupted_task_ids.append(t['task_id'])

                        break

                else:
                    self.debug(t_id, "DP granted, DRS: %.3f" % t['dominant_resource_share'])
                    yield self.dp_waiting_tasks.get(filter=lambda item: item['task_id'] == t['task_id'])
                    this_task["grant_dp_permitted_event"].succeed()
                    # break
            for tid in interrupted_task_ids:
                yield self.dp_waiting_tasks.get(filter=lambda item: item['task_id'] == tid)

    def allocator_init_loop(self):
        while True:
            # loop only blocks here
            msg = yield self.mail_box.get()
            if msg["message_type"] == NEW_TASK:
                assert msg["task_id"] not in self.task_state
                self.task_state[msg["task_id"]] = dict()
                self.task_state[msg["task_id"]]["task_proc"] = msg["task_process"]

            elif msg["message_type"] == ALLOCATION_REQUEST:
                assert msg["task_id"] in self.task_state
                self.task_state[msg["task_id"]]["resource_request"] = msg
                self.task_state[msg["task_id"]]["resource_allocate_timestamp"] = None
                self.task_state[msg["task_id"]]["dp_commit_timestamp"] = None
                self.task_state[msg["task_id"]]["task_completion_timestamp"] = None
                self.task_state[msg["task_id"]]["task_publish_timestamp"] = None

                self.task_state[msg["task_id"]]["is_dp_granted"] = False

                self.task_state[msg["task_id"]]["resource_allocated_event"] = msg.pop("resource_allocated_event")
                self.task_state[msg["task_id"]]["dp_committed_event"] = msg.pop("dp_committed_event")

                self.task_state[msg["task_id"]]["task_completion_event"] = msg.pop("task_completion_event")
                # following two events are controlled by scheduling policy
                self.task_state[msg["task_id"]]["grant_dp_permitted_event"] = self.env.event()
                self.task_state[msg["task_id"]]["resource_permitted_event"] = self.env.event()
                self.task_state[msg["task_id"]]["resource_released_event"] = self.env.event()

                self.task_state[msg["task_id"]]["retired_blocks"] = [] if self.is_policy_dpf else None

                self.task_state[msg["task_id"]]["execution_proc"] = msg.pop("execution_proc")
                self.task_state[msg["task_id"]]["waiting_for_dp_proc"] = msg.pop("waiting_for_dp_proc")

                ## trigger allocation
                self.task_state[msg["task_id"]]["handler_proc_dp"] = self.env.process(
                    self.task_dp_handler(msg["task_id"]))
                self.task_state[msg["task_id"]]["handler_proc_resource"] = self.env.process(
                    self.task_resources_handler(msg["task_id"]))

                self.task_state[msg["task_id"]]["accum_containers"] = dict()  # blk_idx: container

                msg['task_init_event'].succeed()
            elif msg["message_type"] == TASK_RESOURCE_RELEASED:
                assert msg["task_id"] in self.task_state

    def task_dp_handler(self, task_id):
        self.debug(task_id, "Task DP handler created")
        this_task = self.task_state[task_id]
        dp_committed_event = this_task["dp_committed_event"]

        resource_demand = this_task["resource_request"]
        # peek remaining DP, reject if DP is already insufficient
        for i in resource_demand["block_idx"]:
            capacity = self.block_dp_storage.items[i]["dp_container"].capacity
            if capacity < resource_demand["epsilon"]:
                self.debug(task_id, "DP is insufficient before asking dp scheduler, Block ID: %d, remain epsilon: %.3f" % (i, capacity))
                dp_committed_event.fail(
                    InsufficientDpException(task_id,
                        "DP request is rejected by handler, Block ID: %d, remain epsilon: %.3f" % (i, capacity)))
                return

        # getevent -> blk_idx
        get_evt_block_mapping = OrderedDict()

        if self.is_policy_fcfs:
            for i in resource_demand["block_idx"]:
                get_evt_block_mapping[
                    self.block_dp_storage.items[i]["dp_container"].get(resource_demand["epsilon"])] = i
            try:
                yield self.env.all_of(get_evt_block_mapping)
            except Exception as err:
                self.debug(task_id, "fcfs policy error, should return immediately")
                raise err


        # appliable to other policies controlled by another (centralized) processes
        elif self.is_policy_dpf:
            t0 = self.env.now
            for i in resource_demand["block_idx"]:
                self.block_dp_storage.items[i]["arrived_task_num"] += 1
                if not self.block_dp_storage.items[i]["is_retired"]:
                    quota_increment = self.env.config["resource_master.block.init_epsilon"] / self.denom
                    yield from _should_return_immediately(self.env, self.block_dp_storage.items[i][
                        'dp_container'].get(quota_increment))

                    yield self.block_dp_storage.items[i]['dp_quota'].put(quota_increment)
                    if self.block_dp_storage.items[i]["arrived_task_num"] == self.denom:
                        self.block_dp_storage.items[i]["is_retired"] = True
                else:
                    self.task_state[task_id]['retired_blocks'].append(i)

            yield self.dp_waiting_tasks.put({'task_id': task_id,
                                             'dominant_resource_share': None})

            assert self.env.now - t0 < TIMEOUT_TOLERATE


            try:
                t0 = self.env.now
                self.debug(task_id, "wait for grant_dp_permitted" )
                yield self.task_state[task_id]["grant_dp_permitted_event"]
                self.debug(task_id, "grant_dp_permitted after %.1f sec" % (self.env.now - t0 ) )
                for i in resource_demand["block_idx"]:
                    get_evt_block_mapping[
                        self.block_dp_storage.items[i]["dp_quota"].get(resource_demand["epsilon"])] = i

                t0 = self.env.now
                yield self.env.all_of(get_evt_block_mapping)
                assert self.env.now - t0 < TIMEOUT_TOLERATE

            except simpy.Interrupt as err:
                self.debug(task_id, "policy=%s, fail to acquire dp due to [%s]" % (self.policy, err))
                # should not issue get to quota
                assert not self.task_state[task_id]["grant_dp_permitted_event"].triggered
                assert len(get_evt_block_mapping) == 0
                # interrupt dp_waiting_proc
                dp_committed_event.fail(RejectAllocException(task_id,"request DP interrupted"))
                return

        # attach containers to demanded blocks , handling rejection, return dp etc.
        elif self.policy == POLICY_RATE_LIMIT:

            for i in resource_demand["block_idx"]:
                cn = Pool(self.env, capacity=float('inf'), init=0.0)
                this_task["accum_containers"][i] = cn
                self.block_dp_storage.items[i]["accum_containers"][task_id] = cn

                get_evt_block_mapping[cn.get(resource_demand["epsilon"])] = i

            try:
                unget_blk_ids = set(get_evt_block_mapping.values())
                while (len(unget_blk_ids) != 0):
                    listened_gets = (get for get, blk in get_evt_block_mapping.items() if blk in unget_blk_ids)
                    succeed_gets = yield self.env.any_of(listened_gets)
                    for g in succeed_gets:
                        blk_idx = get_evt_block_mapping[g]
                        # stop wait once accum enough
                        accum_container = self.block_dp_storage.items[blk_idx]["accum_containers"].pop(task_id)
                        left_accum = accum_container.level
                        # return extra dp back
                        yield accum_container.get(left_accum)
                        self.block_dp_storage.items[blk_idx]["dp_container"].put(left_accum)
                        unget_blk_ids.remove(blk_idx)

            except simpy.Interrupt as err:
                self.debug(task_id, "request DP interrupted")
                # interrupt dp_waiting_proc
                dp_committed_event.fail(RejectAllocException(task_id,"request DP interrupted"))

                for blk_idx in resource_demand["block_idx"]:
                    # pop, stop task's all waiting container
                    waiting_tasks = self.block_dp_storage.items[blk_idx]["accum_containers"]
                    if task_id in waiting_tasks:
                        waiting_tasks.pop(task_id)

                for get_event, blk_idx in get_evt_block_mapping.items():

                    accum_container = this_task['accum_containers'][blk_idx]
                    dp_container = self.block_dp_storage.items[blk_idx]["dp_container"]
                    # return dp back to block
                    left_accum = accum_container.level
                    if left_accum:
                        yield accum_container.get(left_accum)
                        # if not self.block_dp_storage.items[blk_idx]["is_dead"]:
                        if dp_container.level + left_accum <= dp_container.capacity:

                            yield dp_container.put(left_accum)
                        else:

                            assert dp_container.level + left_accum < dp_container.capacity + NUMERICAL_DELTA
                            yield dp_container.put(dp_container.capacity - dp_container.level)

                    if get_event.triggered:
                        assert get_event.ok
                        if dp_container.level + get_event.amount <= dp_container.capacity:
                            return_dp_event = dp_container.put(get_event.amount)
                            yield return_dp_event
                        else:
                            # tolerate small numerical inaccuracy
                            assert dp_container.level + get_event.amount < dp_container.capacity + NUMERICAL_DELTA
                            yield dp_container.put(dp_container.capacity - dp_container.level)

                    else:
                        get_event.cancel()

                self.debug(task_id, "policy=%s, fail to acquire dp due to [%s]" % (self.policy, err))
                return

        yield from self._commit_dp_allocation(resource_demand["block_idx"], epsilon=resource_demand["epsilon"])
        self.task_state[task_id]["is_dp_granted"] = True
        assert not dp_committed_event.triggered
        dp_committed_event.succeed()

    def task_resources_handler(self, task_id):
        self.debug(task_id, "Task resource handler created")
        # add to wait queue
        self.resource_waiting_tasks.put(task_id)
        this_task = self.task_state[task_id]
        resource_allocated_event = this_task["resource_allocated_event"]
        resource_demand = this_task["resource_request"]
        task_completion_event = this_task["task_completion_event"]

        resource_permitted_event = this_task["resource_permitted_event"]

        get_resrc_events = {}
        try:

            yield resource_permitted_event
            get_cpu_event = self.cpu_pool.get(resource_demand["cpu"])
            get_resrc_events[get_cpu_event] = 'cpu'

            if not self.is_cpu_needed_only:
                get_memory_event = self.memory_pool.get(resource_demand["memory"])
                get_resrc_events[get_memory_event] = 'memory'

                get_gpu_event = self.gpu_pool.get(resource_demand["gpu"])
                get_resrc_events[get_gpu_event] = 'gpu'

            yield self.env.all_of(get_resrc_events)

            resource_allocated_event.succeed()
            self.task_state[task_id]["resource_allocate_timestamp"] = self.env.now
        except simpy.Interrupt as err:
            self.debug(task_id, 'get interrupted when waiting for resources, deallocate')
            put_resrc_event = []
            resource_pools = {'cpu': self.cpu_pool, 'memory':self.memory_pool, 'gpu': self.gpu_pool}
            for get_evt,resource_name in get_resrc_events.items():
                if get_evt.triggered:
                    assert get_evt.ok
                    put_event = resource_pools[resource_name].put(resource_demand[resource_name])
                    put_resrc_event.append(put_event)
                else:
                    get_evt.cancel()
            if len(put_resrc_event):
                yield self.env.all_of(put_resrc_event)
            return


        try:
            yield task_completion_event
        except TaskPreemptedError as err:
            self.debug(task_id, 'task fails while running due to [%s], going to release resource' % err)

        put_resrc_events = []
        put_cpu_event = self.cpu_pool.put(resource_demand["cpu"])
        put_resrc_events.append(put_cpu_event)
        if not self.is_cpu_needed_only:
            put_gpu_event = self.gpu_pool.put(resource_demand["gpu"])
            put_resrc_events.append(put_gpu_event)

            put_memory_event = self.memory_pool.put(resource_demand["memory"])
            put_resrc_events.append(put_memory_event)

        yield self.env.all_of(put_resrc_events)
        self.debug(task_id, "Resource released")
        this_task['resource_released_event'].succeed()

    ## for rate limit policy
    def dp_rate_release_loop(self, block_id):
        is_active = False
        rate = 0
        this_block = self.block_dp_storage.items[block_id]
        while True:
            yield self.clock_tick()
            rest_of_life = this_block["end_of_life"] - self.env.now + 1
            if rest_of_life <= 0:
                if waiting_task_nr != 0:
                    self.debug(block_id, "end of life, with %d waiting tasks' demand" % waiting_task_nr)
                    for task_id in this_block["accum_containers"]:
                        this_task = self.task_state[task_id]
                        assert not this_task["handler_proc_dp"].triggered
                        this_task["handler_proc_dp"].interrupt(
                            "dp allocation is rejected, because block %d reaches end of life" % block_id)
                this_block["is_dead"] = True
                return

            # copy, to avoid iterate over a changing dictionary
            task_ids = list(this_block["accum_containers"].keys())
            for task_id in task_ids:
                this_task = self.task_state[task_id]

                # compare capacity, conservative reject
                # fixme: following not executed??
                if this_task["resource_request"]["epsilon"] > this_block["dp_container"].capacity:
                    this_block["accum_containers"].pop(task_id)
                    this_task["handler_proc_dp"].interrupt(
                        "dp allocation is rejected, cause block %d, Insufficient DP left for task %d" % (
                            block_id, task_id))

            waiting_task_nr = len(this_block["accum_containers"])
            if waiting_task_nr > 0:
                # activate block
                if not is_active:
                    # some dp may left due to task's return operation
                    rate = this_block["dp_container"].level / rest_of_life
                    is_active = True
                try:
                    yield from _should_return_immediately(self.env, this_block["dp_container"].get(rate))

                except Exception as err:
                    self.debug(block_id, "rate %d waiter_nr %d" % (rate, waiting_task_nr))
                    raise err

                for task_id, cn in this_block["accum_containers"].items():
                    cn.put(rate / waiting_task_nr)
            else:
                is_active = False

    def generate_datablocks_loop(self):
        cur_block_nr = 0  # len(self.block_dp_storage.items)
        block_id = count()

        while True:
            if cur_block_nr > self.env.config["resource_master.block.init_amount"]:
                yield self.env.timeout(self.env.config["resource_master.block.arrival_interval"])

            elif cur_block_nr < self.env.config["resource_master.block.init_amount"]:
                cur_block_nr += 1

            elif cur_block_nr == self.env.config["resource_master.block.init_amount"]:
                cur_block_nr += 1
                self.init_blocks_ready.succeed()
                yield self.env.timeout(self.env.config["resource_master.block.arrival_interval"])

            # yield block_id
            cur_block_id = next(block_id)
            total_dp = self.env.config['resource_master.block.init_epsilon']
            new_block = Pool(self.env, capacity=total_dp, init=total_dp, name=cur_block_id, hard_cap=True)

            if self.policy == POLICY_DPF:
                new_quota = Pool(self.env, capacity=total_dp, init=0, name=cur_block_id, hard_cap=True)
                is_retired = False
                arrived_task_num = 0

            else:
                new_quota = None
                is_retired = None
                arrived_task_num = None

            if self.policy == POLICY_RATE_LIMIT:
                EOL = self.env.now + self.env.config['resource_master.dp_policy.rate_policy.lifetime'] - 1
                accum_cn = dict()
                is_dead = False
            else:
                EOL = None
                accum_cn = None
                is_dead = None

            block_item = {
                "dp_container": new_block,
                "dp_quota": new_quota,
                # lifetime :=  # of periods from born to end
                "end_of_life": EOL,
                "accum_containers": accum_cn,  # task_id: container
                'is_dead': is_dead,
                'is_retired': is_retired,
                'arrived_task_num': arrived_task_num,
            }

            yield self.block_dp_storage.put(block_item)
            yield self.unused_dp.put(total_dp)
            self.debug(block_id, "new data block created")

            if self.policy == POLICY_RATE_LIMIT:
                self.env.process(self.dp_rate_release_loop(cur_block_id))

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
            assert epsilon <= this_container.capacity
            # tolerate small numerical inaccuracy
            assert (this_container.level + epsilon) < this_container.capacity + NUMERICAL_DELTA

        for i in block_idx:
            this_container = self.block_dp_storage.items[i]["dp_container"]
            this_container.capacity = max(this_container.capacity - epsilon, this_container.level)
        yield self.unused_dp.get(epsilon * len(block_idx))
        self.committed_dp.put(epsilon * len(block_idx))

    def get_result_hook(self, result):
        if self.env.tracemgr.vcd_tracer.enabled:
            cpu_capacity = self.env.config['resource_master.cpu_capacity']
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
        self.add_connections('resource_master')
        self.add_process(self.generate_tasks)

        self.task_unpublished_count = Container(self.env)
        self.auto_probe('task_unpublished_count', vcd={})

        self.task_published_count = Container(self.env)
        self.auto_probe('task_published_count', vcd={})

        self.task_sleeping_count = Container(self.env)
        self.auto_probe('task_sleeping_count', vcd={})

        self.task_running_count = Container(self.env)
        self.auto_probe('task_running_count', vcd={})

        self.task_completed_count = Container(self.env)
        self.auto_probe('task_completed_count', vcd={})

        self.task_preempted_count = Container(self.env)
        self.auto_probe('task_preempted_count', vcd={})

        self.task_ungranted_count = Container(self.env)
        self.auto_probe('task_ungranted_count', vcd={})

        self.task_granted_count = Container(self.env)
        self.auto_probe('task_granted_count', vcd={})

        self.task_dp_rejected_count = Container(self.env)
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
            self.debug("task cpu demand is fix %d " % self.env.config['task.demand.num_cpu.constant'])
            assert isinstance(self.env.config['task.demand.num_cpu.constant'], int)
            self.cpu_dist = lambda: self.env.config['task.demand.num_cpu.constant']
        else:
            self.cpu_dist = partial(self.env.rand.randint, num_cpu_min, self.env.config['task.demand.num_cpu.max'])

        size_memory_min = 1 if "task.demand.size_memory.min" not in self.env.config else self.env.config[
            'task.demand.size_memory.min']
        self.memory_dist = partial(self.env.rand.randint, size_memory_min,
                                   self.env.config['task.demand.size_memory.max'])

        num_gpu_min = 1 if "task.demand.num_gpu.min" not in self.env.config else self.env.config[
            'task.demand.num_gpu.min']
        self.gpu_dist = partial(self.env.rand.randint, num_gpu_min, self.env.config['task.demand.num_gpu.max'])

        completion_time_min = 1 if "task.completion_time.min" not in self.env.config else self.env.config[
            'task.completion_time.min']
        if self.env.config.get('task.completion_time.constant') is not None:
            self.debug("task completion time is fixed %d" % self.env.config['task.completion_time.constant'])
            self.completion_time_dist = lambda: self.env.config['task.completion_time.constant']
        else:
            self.completion_time_dist = partial(self.env.rand.randint, completion_time_min,
                                                self.env.config['task.completion_time.max'])

        self.epsilon_dist = partial(
            self.env.rand.uniform, 0, self.env.config['resource_master.block.init_epsilon'] / self.env.config[
                'task.demand.epsilon.mean_tasks_per_block'] * 2
        )

        num_blocks_mu = self.env.config['task.demand.num_blocks.mu']
        num_blocks_sigma = self.env.config['task.demand.num_blocks.sigma']
        self.num_blocks_dist = partial(
            self.env.rand.normalvariate, num_blocks_mu, num_blocks_sigma
        )

    def generate_tasks(self):
        """Generate grocery store customers.

        Various configuration parameters determine the distribution of customer
        arrival times as well as the number of items each customer will shop
        for.

        """
        task_id = count()
        arrival_interval_dist = partial(
            self.env.rand.expovariate, 1 / self.env.config['task.arrival_interval']
        )
        ## wait for generating init blocks

        yield self.resource_master.init_blocks_ready
        while True:
            yield self.env.timeout(arrival_interval_dist())
            t_id = next(task_id)

            task_process = self.env.process(self.task(t_id))
            new_task_msg = {"message_type": NEW_TASK,
                            "task_id": t_id,
                            "task_process": task_process,
                            }

            self.resource_master.mail_box.put(new_task_msg)

    def task(self, task_id):
        yield self.task_unpublished_count.put(1)
        yield self.task_ungranted_count.put(1)
        yield self.task_sleeping_count.put(1)

        epsilon = self.epsilon_dist()

        t0 = self.env.now
        # query existing data blocks
        num_stored_blocks = len(self.resource_master.block_dp_storage.items)
        assert (num_stored_blocks > 0)
        num_blocks_demand = min(max(1, round(self.num_blocks_dist())), num_stored_blocks)

        self.debug(task_id, 'DP demand epsilon=%.3f for %d blocks' % (epsilon, num_blocks_demand))

        resource_allocated_event = self.env.event()
        dp_committed_event = self.env.event()
        task_completion_event = self.env.event()
        task_init_event = self.env.event()

        def run_task(task_id, resource_allocated_event, completion_event):

            assert not resource_allocated_event.triggered
            try:

                yield resource_allocated_event

                resource_alloc_time = self.resource_master.task_state[task_id]["resource_allocate_timestamp"]
                assert resource_alloc_time is not None
                resrc_allocation_wait_duration = resource_alloc_time - t0
                self.debug(task_id, 'Resources allocated after', timedelta(seconds=resrc_allocation_wait_duration))

                yield self.task_sleeping_count.get(1)
                yield self.task_running_count.put(1)

                # running task
                yield self.env.timeout(resource_request_msg["completion_time"])
                completion_event.succeed()

                # yield self.resource_master.mail_box.put(task_completion_msg)
                task_completion_timestamp = self.resource_master.task_state[task_id][
                    "task_completion_timestamp"] = self.env.now
                task_completion_duration = task_completion_timestamp - t0
                self.debug(task_id, 'Task completed after', timedelta(seconds=task_completion_duration))
                yield self.task_running_count.get(1)
                yield self.task_completed_count.put(1)
                return 0
            except simpy.Interrupt as err:
                task_preempted_timestamp = self.resource_master.task_state[task_id][
                    "task_completion_timestamp"] = -self.env.now
                # note negative sign here
                task_preempted_duration = - task_preempted_timestamp - t0
                self.debug(task_id, 'Task preempted after', timedelta(seconds=task_preempted_duration))

                if not resource_allocated_event.triggered:
                    # interrupt handler_proc_resource
                    self.debug(task_id, "Task gets preempted while waiting for resource alloc")
                    self.resource_master.task_state[task_id]["handler_proc_resource"].interrupt(
                        "Cancel resource allocation for me, get preempted")
                    yield self.task_sleeping_count.get(1)
                    yield self.task_preempted_count.put(1)

                else:
                    assert resource_allocated_event.ok
                    self.debug(task_id, "Task gets preempted while running")
                    # interrupt handler_proc_resource
                    completion_event.fail(TaskPreemptedError("Abort task, preempted while runninng"))
                    yield self.task_running_count.get(1)
                    yield self.task_preempted_count.put(1)

                return 1

        def wait_for_dp(task_id, dp_committed_event, execution_proc):
            # dp_committed_event = self.resource_master.task_state[task_id]["dp_committed_event"]

            assert not dp_committed_event.triggered
            try:
                yield dp_committed_event
                dp_committed_time = self.resource_master.task_state[task_id]["dp_commit_timestamp"] = self.env.now
                dp_committed_duration = dp_committed_time - t0
                self.debug(task_id, 'DP committed after', timedelta(seconds=dp_committed_duration))

                yield self.task_ungranted_count.get(1)
                yield self.task_granted_count.put(1)
                return 0
            except (InsufficientDpException, RejectAllocException) as err:
                assert not dp_committed_event.ok
                if execution_proc.is_alive:
                    execution_proc.interrupt('Stop! task_dp_handler fail to acquire DP')
                dp_rejected_timestamp = self.resource_master.task_state[task_id]["dp_commit_timestamp"] = -self.env.now
                allocation_rej_duration = - dp_rejected_timestamp - t0

                self.debug(task_id, 'fail to commit DP after', timedelta(seconds=allocation_rej_duration),
                           "due to [%s]" % err)
                self.task_dp_rejected_count.put(1)
                return 1

        # listen, wait for allocation
        running_task = self.env.process(run_task(task_id, resource_allocated_event, task_completion_event))
        waiting_for_dp = self.env.process(wait_for_dp(task_id, dp_committed_event, running_task))

        # prep allocation request,
        resource_request_msg = {
            "message_type": ALLOCATION_REQUEST,
            "task_id": task_id,
            "cpu": self.cpu_dist(),
            "memory": self.memory_dist(),
            "gpu": self.gpu_dist(),
            "epsilon": epsilon,
            # todo maybe exclude EOL blocks?
            "block_idx": list(range(num_stored_blocks))[-num_blocks_demand:],  # choose latest num_blocks_demand
            "completion_time": self.completion_time_dist(),
            "resource_allocated_event": resource_allocated_event,
            "dp_committed_event": dp_committed_event,
            'task_completion_event': task_completion_event,
            'task_init_event': task_init_event,
            "user_id": None,
            "model_id": None,
            'execution_proc': running_task,
            'waiting_for_dp_proc': waiting_for_dp,
        }
        # send allocation request, note do it after listen!
        yield self.resource_master.mail_box.put(resource_request_msg)

        dp_grant, task_exec = yield self.env.all_of([waiting_for_dp, running_task])

        if dp_grant.value == 0:
            # verify, if dp granted, then task must be completed.
            assert task_exec.value == 0
            self.resource_master.task_state[task_id]["task_publish_timestamp"] = self.env.now
            yield self.task_unpublished_count.get(1)
            yield self.task_published_count.put(1)
            self.debug(task_id, "task get published")
        else:
            assert dp_grant.value == 1
            self.debug(task_id, "task fail to publish")
            self.resource_master.task_state[task_id]["task_publish_timestamp"] = None

        if self.db:
            # verify iff cp commit fail <=> no publish
            if self.resource_master.task_state[task_id]["task_publish_timestamp"] is None:
                assert self.resource_master.task_state[task_id]["dp_commit_timestamp"] < 0

            if self.resource_master.task_state[task_id]["dp_commit_timestamp"] < 0:
                assert self.resource_master.task_state[task_id]["task_publish_timestamp"] is None

            NoneType = type(None)
            assert isinstance(task_id, int)
            assert isinstance(resource_request_msg["block_idx"][0], int)
            assert isinstance(num_blocks_demand, int)
            assert isinstance(resource_request_msg["epsilon"], float)
            assert isinstance(resource_request_msg["cpu"], int)
            assert isinstance(resource_request_msg["gpu"], (int, NoneType))
            assert isinstance(resource_request_msg["memory"], (float, int, NoneType))
            assert isinstance(t0, float)
            assert isinstance(self.resource_master.task_state[task_id]["dp_commit_timestamp"], (float, int))
            assert isinstance(self.resource_master.task_state[task_id]["resource_allocate_timestamp"],
                              (float, NoneType, int))
            assert isinstance(self.resource_master.task_state[task_id]["task_completion_timestamp"], (float, int))
            assert isinstance(self.resource_master.task_state[task_id]["task_publish_timestamp"],
                              (float, NoneType, int))
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


def _should_return_immediately(env, event, timeout_tolerate=0.01):
    aa = yield env.timeout(timeout_tolerate, value=TIMEOUT_VAL) | event
    if aa == TIMEOUT_VAL:
        raise ShouldTriggeredImmediatelyException("non-retired block cannot get")

if __name__ == '__main__':
    import copy

    run_test_parallel = True
    run_test_many = False
    run_test_single = False

    config = {

        'task.arrival_interval': 10,
        'task.demand.num_blocks.mu': 20,
        'task.demand.num_blocks.sigma': 10,
        # num_blocks * 1/mean_tasks_per_block = 1/10
        'task.demand.epsilon.mean_tasks_per_block': 200,

        'task.completion_time.constant': 1,
        # max = half of capacity
        'task.completion_time.max': 100,
        # 'task.completion_time.min': 10,
        'task.demand.num_cpu.constant': 1,  # [min, max]
        'task.demand.num_cpu.max': 80,
        'task.demand.num_cpu.min': 1,

        'task.demand.size_memory.max': 412,
        'task.demand.size_memory.min': 1,

        'task.demand.num_gpu.max': 3,
        'task.demand.num_gpu.min': 1,

        'resource_master.block.init_epsilon': 5.0,
        'resource_master.block.init_amount': 20,
        'resource_master.block.arrival_interval': 100,
        # 'resource_master.dp_policy': POLICY_RATE_LIMIT,
        # 'resource_master.dp_policy.rate_policy.lifetime': 300,

        # 'resource_master.dp_policy': FCFS_POLICY,
        'resource_master.dp_policy': POLICY_DPF,
        'resource_master.dp_policy.dpf.denominator': 20,
        # https://cloud.google.com/compute/docs/gpus
        # V100 VM instance
        'resource_master.is_cpu_needed_only': True,
        'resource_master.cpu_capacity': 96,  # number of cores
        'resource_master.memory_capacity': 624,  # in GB, assume granularity is 1GB
        'resource_master.gpu_capacity': 8,  # in cards

        'sim.db.enable': True,
        'sim.db.persist': True,
        'sim.dot.colorscheme': 'blues5',
        'sim.dot.enable': True,
        'sim.duration': '500000 s',
        # 'sim.duration': '50000 s',
        'sim.gtkw.file': 'sim.gtkw',
        'sim.gtkw.live': False,
        'sim.log.enable': True,
        "sim.log.level": "DEBUG",
        'sim.progress.enable': True,
        'sim.result.file': 'result.json',
        'sim.seed': 1234,
        'sim.timescale': 's',
        'sim.vcd.dump_file': 'sim_dp.vcd',
        'sim.vcd.enable': True,
        'sim.vcd.persist': True,
        'sim.workspace': 'workspace',
        'sim.workspace.overwrite': True,

    }

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
    factors = parse_user_factors(config, args.factors)
    if factors:
        simulate_factors(config, factors, Top)
    else:
        if run_test_single:
            pprint(config)
            simulate(copy.deepcopy(config), Top)

    task_configs = {}
    scheduler_configs = {}
    config1 = copy.deepcopy(config)
    # use rate limit by default
    config1["resource_master.dp_policy"] = POLICY_RATE_LIMIT
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

    scheduler_configs["fcfs_policy"] = {'resource_master.dp_policy': POLICY_FCFS}

    scheduler_configs["rate_policy_slow_release"] = {'resource_master.dp_policy': POLICY_RATE_LIMIT,
                                                     'resource_master.dp_policy.rate_policy.lifetime': config1[
                                                                                                           'task.arrival_interval'] * 10 * 5
                                                     # 500
                                                     }
    scheduler_configs["rate_policy_fast_release"] = {'resource_master.dp_policy': POLICY_RATE_LIMIT,
                                                     'resource_master.dp_policy.rate_policy.lifetime': config1[
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
    config2["sim.workspace"] = "workspace_rate_limiting"
    config2["resource_master.dp_policy"] = POLICY_RATE_LIMIT
    config2["resource_master.dp_policy.rate_policy.lifetime"] = 300

    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_fcfs"
    config2["resource_master.dp_policy"] = POLICY_FCFS

    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_dpf"
    config2["resource_master.dp_policy"] = POLICY_DPF

    configs.append(config2)
    if run_test_parallel:
        simulate_many(configs, Top)


    if run_test_many:
        for c in configs:
            pprint(c)
            simulate(c, Top)

    for sched_conf_k, sched_conf_v in scheduler_configs.items():
        for task_conf_k, task_conf_v in task_configs.items():
            new_config = copy.deepcopy(config1)
            new_config.update(sched_conf_v)
            new_config.update(task_conf_v)
            workspace_name = "workspace_%s-%s" % (sched_conf_k, task_conf_k)
            new_config["sim.workspace"] = workspace_name
            configs.append(new_config)
    # if run_test_many:
    #     simulate_many(configs, Top)

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
