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
from itertools import count
import simpy
from simpy import Container, Resource, Store, Event
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

from desmod.component import Component
from desmod.config import apply_user_overrides, parse_user_factors
from desmod.dot import generate_dot
from desmod.queue import Queue
from desmod.pool import Pool

from desmod.simulation import simulate, simulate_factors
ALLOCATION_SUCCESS="V"
ALLOCATION_FAIL="F"
ALLOCATION_REQUEST="allocation_request"
NEW_TASK="new_task_created"
# TASK_COMPLETION="task_completion"
TASK_RESOURCE_RELEASE="resource_release"
FCFS_POLICY="fcfs"
RATE_LIMIT_POLICY="rate"

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
            'datafmt': 'dec',
            'color': 'cycle',
            'extraflags': ['analog_step'],
        }
        with open(env.config['sim.gtkw.file'], 'w') as gtkw_file:
            gtkw = GTKWSave(gtkw_file)
            gtkw.dumpfile(env.config['sim.vcd.dump_file'], abspath=False)
            gtkw.treeopen('grocery')
            gtkw.signals_width(300)
            gtkw.trace('customers.active', **analog_kwargs)
            for i in range(env.config['grocery.num_lanes']):
                with gtkw.group(f'Lane{i}'):
                    scope = f'grocery.lane{i}'
                    gtkw.trace(f'{scope}.customer_queue', **analog_kwargs)
                    gtkw.trace(f'{scope}.feed_belt', **analog_kwargs)
                    gtkw.trace(f'{scope}.bag_area', **analog_kwargs)
                    gtkw.trace(f'{scope}.baggers', **analog_kwargs)

    def elab_hook(self):
        # We generate DOT representations of the component hierarchy. It is
        # only after elaboration that the component tree is fully populated and
        # connected, thus generate_dot() is called here in elab_hook().
        generate_dot(self)

class InsufficientDpException(Exception):
    pass

class RejectAllocException(Exception):
    pass

class ResourceMaster(Component):
    """Model a grocery store with checkout lanes, cashiers, and baggers."""

    base_name = 'resource_master'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_dp_storage = Queue(self.env, capacity=float("inf"))
        # self.block_waiting_containers = [] # [[],[],...,[] ], put each task's waiting DP container in sublist
        self.is_cpu_limited_only = self.env.config['resource_master.is_cpu_limited_only']
        self.cpu_pool = Pool(self.env, capacity=self.env.config["resource_master.cpu_capacity"],  init=self.env.config["resource_master.cpu_capacity"], hard_cap=True)

        if not self.is_cpu_limited_only:
            self.memory_pool = Pool(self.env, capacity=self.env.config["resource_master.memory_capacity"], init=self.env.config["resource_master.memory_capacity"],hard_cap=True)
            self.gpu_pool = Pool(self.env, capacity=self.env.config["resource_master.gpu_capacity"], init=self.env.config["resource_master.gpu_capacity"], hard_cap=True)

        self.mail_box = Queue(self.env)
        self.task_state = dict() # {task_id:{...},...,}
        self.add_processes(self.generate_datablocks)
        self.add_processes(self.allocator_loop)
        self.period = 1
        self.policy = self.env.config["resource_master.allocation_policy"]

    def clock_tick(self):
        return self.env.timeout(self.period)

    def allocator_loop(self):
        while True:
            msg = yield self.mail_box.get()
            if msg["message_type"] == NEW_TASK:
                assert msg["task_id"] not in self.task_state
                self.task_state[msg["task_id"]] = dict()
                self.task_state[msg["task_id"]]["task_proc"] = msg["task_process"]

            elif msg["message_type"] == ALLOCATION_REQUEST:
                assert msg["task_id"] in self.task_state
                self.task_state[msg["task_id"]]["resource_request"] = msg
                self.task_state[msg["task_id"]]["allocation_done_event"] = msg.pop("allocation_done_event")
                self.task_state[msg["task_id"]]["task_completion_event"] = self.env.event()
                # self.task_state[msg["task_id"]]["finished"] = False
                task_handler_gen = self.task_handler(msg["task_id"])
                ## trigger allocation
                handler_proc = self.env.process(task_handler_gen)
                self.task_state[msg["task_id"]]["handler_proc"] = handler_proc
                self.task_state[msg["task_id"]]["accum_containers"] = dict() # blk_idx: container
                for i in msg["block_idx"]:
                    cn = Pool(self.env, capacity=float('inf'), init=0.0)
                    self.task_state[msg["task_id"]]["accum_containers"][i] = cn
                self.block_dp_storage.items[i]["accum_containers"][msg['task_id']] = cn

            elif msg["message_type"] == TASK_RESOURCE_RELEASE:
                assert msg["task_id"] in self.task_state
                # self.task_state[msg["task_id"]]["finished"] = True


    def task_handler(self,task_id):
        self.debug(task_id, "Task handler created")
        this_task = self.task_state[task_id]
        allocation_done_event = this_task["allocation_done_event"]
        resource_demand = this_task["resource_request"]
        task_completion_event = this_task["task_completion_event"]
        # peek remaining DP
        for i in resource_demand["block_idx"]:
            capacity = self.block_dp_storage.items[i]["dp_container"].capacity
            if capacity < resource_demand["epsilon"]:
                self.debug(task_id, "InsufficientDpException, Block ID: %d, remain epsilon: %.3f" % (i, capacity))
                allocation_done_event.fail(
                    InsufficientDpException("Block ID: %d, remain epsilon: %.3f" % (i, capacity)))
                return
        # getevent: blk_idx
        get_dp_events = dict()
        for i in resource_demand["block_idx"]:
            if self.policy == FCFS_POLICY:
                get_dp_events[self.block_dp_storage.items[i]["dp_container"].get(resource_demand["epsilon"])] = i
            elif self.policy == RATE_LIMIT_POLICY:
                get_dp_events[this_task['accum_containers'][i].get(resource_demand["epsilon"])] = i

        if self.policy == FCFS_POLICY:
            try:
                yield self.env.all_of(get_dp_events)
            except simpy.Interrupt as e:
                self.debug(task_id, "request DP interrupted")
                allocation_done_event.fail(RejectAllocException("request DP interrupted"))
                # return DP to block
                for get_event,blk_idx in get_dp_events.items():
                    # level reduced
                    if get_event.triggered:
                        self.block_dp_storage.items[blk_idx]["dp_container"].put(get_event.amount)
                    else:
                        get_event.cancel()
                self.debug(task_id, e)
                return

        elif self.policy == RATE_LIMIT_POLICY:
            try:
                unget_blk_ids = set(get_dp_events.values())
                while (len(unget_blk_ids) != 0):
                    listened_gets = (get for get,blk in get_dp_events.items() if blk in unget_blk_ids)
                    succeed_gets = yield self.env.any_of(listened_gets)
                    for g in succeed_gets:
                        blk_idx = get_dp_events[g]
                        # stop wait once accum enough
                        accum_container = self.block_dp_storage.items[blk_idx]["accum_containers"].pop(task_id)
                        left_accum = accum_container.level
                        # return extra dp back
                        yield accum_container.get(left_accum)
                        self.block_dp_storage.items[blk_idx]["dp_container"].put(left_accum)
                        unget_blk_ids.remove(blk_idx)

            except simpy.Interrupt as e:
                self.debug(task_id, "request DP interrupted")
                allocation_done_event.fail(RejectAllocException("request DP interrupted"))

                for blk_idx in resource_demand["block_idx"]:
                    # pop, stop task's all waiting container
                    if task_id in self.block_dp_storage.items[blk_idx]:
                        self.block_dp_storage.items[blk_idx]["accum_containers"].pop(task_id)

                for blk_idx, get_event in zip(resource_demand["block_idx"], get_dp_events):

                    accum_container = this_task['accum_containers'][blk_idx]

                    # return dp back to block
                    left_accum = accum_container.level
                    if left_accum:
                        yield accum_container.get(left_accum)
                        self.block_dp_storage.items[blk_idx]["dp_container"].put(left_accum)
                    if get_event.triggered:
                        self.block_dp_storage.items[blk_idx]["dp_container"].put(get_event.amount)
                    else:
                        get_event.cancel()

                self.debug(task_id, e)
                return

        self._commit_dp_allocation(resource_demand["block_idx"], epsilon=resource_demand["epsilon"])

        get_cpu_event = self.cpu_pool.get(resource_demand["cpu"])
        if not self.is_cpu_limited_only:
            get_memory_event = self.memory_pool.get(resource_demand["memory"])
            get_gpu_event = self.gpu_pool.get(resource_demand["gpu"])

            yield self.env.all_of([get_cpu_event, get_memory_event, get_gpu_event])
        else:
            yield get_cpu_event

        allocation_done_event.succeed()

        yield task_completion_event

        put_cpu_event = self.cpu_pool.put(resource_demand["cpu"])
        if not self.is_cpu_limited_only:
            put_gpu_event = self.gpu_pool.put(resource_demand["gpu"])
            put_memory_event = self.memory_pool.put(resource_demand["memory"])

            yield self.env.all_of([put_cpu_event, put_memory_event, put_gpu_event])
        else:
            yield put_cpu_event
        self.debug(task_id, "Resource released")
        resource_release_msg = {
            "message_type": TASK_RESOURCE_RELEASE,
            "task_id": task_id }
        yield self.mail_box.put(resource_release_msg)

## for rate limit policy
    def rate_release_dp(self, block_id):
        is_active = False
        rate = 0
        this_block = self.block_dp_storage.items[block_id]
        while True:
            yield self.clock_tick()
            # self.debug(block_id,"clock tick")
            rest_of_life = this_block["end_of_life"] - self.env.now
            if rest_of_life < 0:
                if waiting_task_nr != 0:
                    self.debug(block_id, "end of life, with %d waiting tasks' demand" % waiting_task_nr)
                    for task_id in this_block["accum_containers"]:
                        this_task = self.task_state[task_id]
                        if this_task["handler_proc"]._value is PENDING:
                            this_task["handler_proc"].interrupt("block %d reaches end of life" % block_id)

                return

            # copy, to avoid iterate over a changing dictionary
            task_ids = list(this_block["accum_containers"].keys())
            for task_id in task_ids:
                this_task = self.task_state[task_id]
                # todo or > capacity or level??
                if this_task["resource_request"]["epsilon"] > this_block["dp_container"].capacity:
                    this_block["accum_containers"].pop(task_id)
                    this_task["handler_proc"].interrupt("block %d, Insufficient DP left for task %d" % (block_id,task_id))

            waiting_task_nr = len(this_block["accum_containers"])
            if waiting_task_nr != 0:
                # activate block
                if not is_active:
                    # some dp may left due to task's return operation
                    rate = this_block["dp_container"].level / rest_of_life
                    is_active = True
                try:
                    yield this_block["dp_container"].get(rate)
                except Exception as e:
                    self.debug(block_id, "rate %d waiter_nr %d" % (rate, waiting_task_nr) )
                for task_id, cn in this_block["accum_containers"].items():
                    cn.put(rate/waiting_task_nr)
            else:
                is_active = False


    def generate_datablocks(self):
        init_block_nr = 0
        block_id = count()

        while True:
            if init_block_nr >= self.env.config["resource_master.block.init_amount"]:
                yield self.env.timeout(self.env.config["resource_master.block.arrival_interval"])
            else:
                init_block_nr += 1
            # yield block_id
            cur_block_id = next(block_id)
            total_dp = self.env.config['resource_master.block.init_epsilon']
            new_block = Pool(self.env, capacity=total_dp,init=total_dp, name=cur_block_id)
            block_item = {
                "dp_container": new_block,
                # lifetime :=  # of periods from born to end
                "end_of_life": self.env.now + self.env.config['resource_master.block.lifetime'] - 1,
                "accum_containers": dict(), # task_id: container
            }
            yield self.block_dp_storage.put(block_item)
            # self.debug(block_id, "new data block created")
            # self.block_waiting_containers.append([])
            if self.policy == RATE_LIMIT_POLICY:
                self.env.process(self.rate_release_dp(cur_block_id))

    def _commit_dp_allocation(self, block_idx: List[int], epsilon: float):
        """
        each block's capacity is uncommitted DP, commit by deducting capacity by epsilon.
        Args:
            block_idx:
            epsilon:

        Returns:

        """
        for i in block_idx:
            try:
                assert self.block_dp_storage.items[i]["dp_container"].capacity >= epsilon
            except:
                raise Exception("Verification failed, insufficient epsilon to commit: data block idx %d, DP demand %.3f, uncommitted DP %.3f" % (i , epsilon, self.block_dp_storage.items[i].capacity))

        for i in block_idx:

            self.block_dp_storage.items[i]["dp_container"].capacity -= epsilon




class Tasks(Component):
    """Model customer arrival rate and in-store behavior.

    Each customer's arrival time, number of items, and shopping time is
    determined by configuration.

    A new process is spawned for each customer.

    A "customers" database table captures per-customer checkout times. A
    primary goal for this model is optimizing customer checkout time (latency)
    and throughput.

    """

    base_name = 'tasks'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_connections('resource_master')
        self.add_process(self.generate_tasks)
        self.active_count = Container(self.env)
        self.auto_probe('active_count', vcd={})
        self.db = None
        # if self.env.tracemgr.sqlite_tracer.enabled:
        #     self.db = self.env.tracemgr.sqlite_tracer.db
        #     self.db.execute(
        #         'CREATE TABLE customers '
        #         '(cust_id INTEGER PRIMARY KEY,'
        #         ' num_items INTEGER,'
        #         ' shop_time REAL,'
        #         ' checkout_time REAL)'
        #     )
        # else:
        #     self.db = None


        self.cpu_dist = partial(self.env.rand.randint, 1, self.env.config['task.demand.num_cpu.max'])
        self.memory_dist = partial(self.env.rand.randint, 1, self.env.config['task.demand.size_memory.max'])
        self.gpu_dist = partial(self.env.rand.randint, 1, self.env.config['task.demand.num_gpu.max'])
        self.completion_time_dist = partial(self.env.rand.randint, 1, self.env.config['task.completion_time.max'])

        self.epsilon_dist = partial(
            self.env.rand.uniform,0,self.env.config['resource_master.block.init_epsilon'] / self.env.config['task.demand.epsilon.mean_tasks_per_block'] * 2
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

        while True:
            ## wait for generating init blocks
            yield self.env.timeout(arrival_interval_dist())
            t_id = next(task_id)

            task_process = self.env.process(self.task(t_id))
            new_task_msg = {"message_type": NEW_TASK,
                           "task_id": t_id,
                            "task_process": task_process,
                            }

            self.resource_master.mail_box.put(new_task_msg)

    def task(self, task_id):
        """Grocery store customer behavior."""
        yield self.active_count.put(1)
        epsilon = self.epsilon_dist()
        # yield self.env.timeout(epsilon)
        # self.debug(task_id, 'ready to checkout after', timedelta(seconds=epsilon))

        t0 = self.env.now
        # query existing data blocks
        num_stored_blocks = len(self.resource_master.block_dp_storage.items)
        num_blocks_demand = min(max(1, round(self.num_blocks_dist())),  num_stored_blocks)

        self.debug(task_id, 'DP demand epsilon=%.3f for %d blocks' % (epsilon, num_blocks_demand))
        allocation_done_event = self.env.event()
        resource_request_msg = {
                        "message_type": ALLOCATION_REQUEST,
                        "task_id": task_id,
                        "cpu": self.cpu_dist(),
                       "memory": self.memory_dist(),
                       "gpu": self.gpu_dist(),
                       "epsilon" : epsilon,
                       "block_idx": list(range(num_stored_blocks))[-num_blocks_demand:], # choose latest num_blocks_demand
                       "completion_time":self.completion_time_dist(),
                        "allocation_done_event": allocation_done_event,
                       "user_id": None,
                       "model_id": None}

        yield self.resource_master.mail_box.put(resource_request_msg)

        # wait for allocation
        try:

            # allocation_done_event = self.resource_master.task_state[task_id]["allocation_done_event"]

            yield allocation_done_event
            allocation_done_time = self.env.now - t0

            
            self.debug(task_id, 'Allocation succeeded after', timedelta(seconds=allocation_done_time))
            # running task
            yield self.env.timeout(resource_request_msg["completion_time"])
            completion_event = self.resource_master.task_state[task_id]["task_completion_event"]
            completion_event.succeed()

            # yield self.resource_master.mail_box.put(task_completion_msg)

            jct = self.env.now - t0
            self.debug(task_id, 'Task completed after', timedelta(seconds=jct))

        except (InsufficientDpException, RejectAllocException) as e:
            allocation_done_time = self.env.now - t0
            self.debug(task_id, e)
            self.debug(task_id, 'Allocation failed after', timedelta(seconds=allocation_done_time))

        yield self.active_count.get(1)

        return

        # todo, async compute while wait for DP allocation??
        # yield self.env.all_of([self.env.timeout(self.env.config['task.completion_time.max']), allocation_done_event])




    def get_result_hook(self, result):
        if not self.db:
            return
        result['checkout_time_avg'] = self.db.execute(
            'SELECT AVG(checkout_time) FROM customers'
        ).fetchone()[0]
        result['checkout_time_min'] = self.db.execute(
            'SELECT MIN(checkout_time) FROM customers'
        ).fetchone()[0]
        result['checkout_time_max'] = self.db.execute(
            'SELECT MAX(checkout_time) FROM customers'
        ).fetchone()[0]
        result['customers_total'] = self.db.execute(
            'SELECT COUNT() FROM customers'
        ).fetchone()[0]
        result['customers_per_hour'] = result['customers_total'] / (
            self.env.time() / 3600
        )


if __name__ == '__main__':
    config = {
        'bagger.bag_time': 1.5,
        'bagger.policy': 'float-aggressive',
        'cashier.bag_time': 2.0,
        'cashier.scan_time': 2.0,
        'checkout.bag_area_capacity': 15,
        'checkout.feed_capacity': 20,
        
        'task.arrival_interval': 20,
        'task.demand.num_blocks.mu': 6,
        'task.demand.num_blocks.sigma': 5,
        'task.demand.epsilon.mean_tasks_per_block': 3.0,

        # max = half of capacity
        'task.completion_time.max': 10,
        'task.demand.num_cpu.max': 4,
        'task.demand.size_memory.max': 412,
        'task.demand.num_gpu.max': 4,

        'resource_master.block.init_epsilon': 5.0,
        'resource_master.block.init_amount': 50,
        'resource_master.block.arrival_interval': 120,
        'resource_master.block.lifetime': 300,
        'resource_master.allocation_policy': RATE_LIMIT_POLICY,
        # 'resource_master.allocation_policy': FCFS_POLICY,

        # https://cloud.google.com/compute/docs/gpus
        # V100 VM instance
        'resource_master.is_cpu_limited_only': True,
        'resource_master.cpu_capacity': 96, # number of cores
        'resource_master.memory_capacity': 624, # in GB, assume granularity is 1GB
        'resource_master.gpu_capacity': 8,  # in cards

        'grocery.num_baggers': 1,
        'grocery.num_lanes': 2,
        'sim.db.enable': True,
        'sim.db.persist': False,
        'sim.dot.colorscheme': 'blues5',
        'sim.dot.enable': True,
        'sim.duration': '100000 s',
        'sim.gtkw.file': 'sim.gtkw',
        'sim.gtkw.live': False,
        'sim.log.enable': True,
        "sim.log.level": "DEBUG",
        'sim.progress.enable': False,
        'sim.result.file': 'result.json',
        'sim.seed': 1234,
        'sim.timescale': 's',
        'sim.vcd.dump_file': 'sim.vcd',
        'sim.vcd.enable': True,
        'sim.vcd.persist': False,
        'sim.workspace': 'workspace',
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
        simulate(config, Top)
