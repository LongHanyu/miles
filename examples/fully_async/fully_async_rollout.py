import asyncio
import atexit
import logging
import os
import queue
import threading
import time

import aiohttp
import httpx

from miles.rollout.data_source import DataSource
from miles.rollout.filter_hub.base_types import call_dynamic_filter
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


METRICS_LOG_INTERVAL_SECONDS = float(os.environ.get("FULLY_ASYNC_METRICS_INTERVAL_SECONDS", "10"))
_SGLANG_METRIC_NAMES = ("sglang_num_running_reqs", "sglang_num_queue_reqs", "sglang_num_requests_total")

# Per-request timeout for generation POSTs. Non-streaming, so the server sends
# nothing back until the whole generation finishes -- ``read`` therefore bounds
# the total generation time (default 1h). ``connect``/``write`` stay finite so a
# stalled connect/upload still raises and feeds the retry loop in http_utils;
# ``pool`` is unbounded so non-generate calls sharing the client don't get a
# spurious PoolTimeout while long requests hold every connection slot.
_REQUEST_READ_TIMEOUT_SECONDS = float(os.environ.get("FULLY_ASYNC_REQUEST_READ_TIMEOUT_SECONDS", "3600"))
_REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, write=60.0, read=_REQUEST_READ_TIMEOUT_SECONDS, pool=None)

# Hard wall-clock deadline for a single group task. A task still running past
# this is treated as wedged (e.g. stuck in the post() retry / read-timeout loop,
# holding a concurrency slot without making progress) and is force-cancelled so
# its ``active_tasks`` slot is reclaimed and new groups can be submitted again.
# Cancelling propagates into the in-flight httpx request -> closes the connection
# -> sglang detects the disconnect, aborts the request and frees its KV slot too.
_GROUP_HARD_DEADLINE_SECONDS = float(os.environ.get("FULLY_ASYNC_GROUP_DEADLINE_SECONDS", "3600"))


def _parse_sglang_metrics(text: str) -> dict[str, float]:
    """Sum the ``_SGLANG_METRIC_NAMES`` lines from an OpenMetrics dump."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in _SGLANG_METRIC_NAMES:
            totals[name] = totals.get(name, 0.0) + float(line.rsplit(maxsplit=1)[-1])
    return totals


def group_oldest_weight_version(group: list[Sample]) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [s.oldest_weight_version for s in group if s.oldest_weight_version is not None]
    return min(versions) if versions else None


class _CachedWeightVersion:
    """Throttled query for the current engine weight version via /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query: float = 0.0

    async def get(self, args) -> int | None:
        now = time.monotonic()
        if self._value is not None and (now - self._last_query) < self._ttl:
            return self._value
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._value = int(data["weight_version"])
                        self._last_query = now
        except Exception as e:
            logger.debug(f"Failed to query engine weight version: {e}")
        return self._value


_cached_version = _CachedWeightVersion()


class _WeightVersionTracker:
    """Staleness tracking from observed request returns -- no HTTP /model_info query.

    cur_weight : highest weight version seen across ALL returned requests
                 (starts at 0 before any request comes back).
    _baselines : per-group snapshot of ``cur_weight`` taken at submission time,
                 keyed by the worker's monotonic ``group_id``. A dict (not a
                 list) because ``group_id`` grows across the worker's whole
                 lifetime; baselines are popped on consume to bound memory.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.cur_weight: int = 0
        self._baselines: dict[int, int] = {}

    @staticmethod
    def _max_version(group: list[Sample]) -> int | None:
        versions = [int(v) for s in group for v in s.weight_versions if str(v).isdigit()]
        return max(versions) if versions else None

    def on_submit(self, group_id: int) -> None:
        """Snapshot the current latest weight as this group's baseline (worker thread)."""
        with self._lock:
            self._baselines[group_id] = self.cur_weight

    def observe(self, group: list[Sample]) -> None:
        """Fold a returned group's weight versions into ``cur_weight`` (worker thread)."""
        newest = self._max_version(group)
        if newest is None:
            return
        with self._lock:
            if newest > self.cur_weight:
                self.cur_weight = newest

    def staleness(self, group_id: int) -> int | None:
        """``cur_weight - baseline`` for this group; pops the baseline (rollout thread).

        Returns ``None`` only when no baseline was recorded for this group
        (e.g. it was submitted while the staleness filter was disabled), in
        which case the group must not be treated as stale.
        """
        with self._lock:
            baseline = self._baselines.pop(group_id, None)
            if baseline is None:
                return None
            return self.cur_weight - baseline

    def exceeds(self, group_id: int, max_staleness: int) -> bool:
        with self._lock:
            baseline = self._baselines.get(group_id)
            return baseline is not None and self.cur_weight - baseline > max_staleness

    def discard(self, group_ids) -> None:
        """Drop baselines for groups that were pulled but never consumed.

        ``staleness`` is the only other place that pops a baseline, so groups
        that a rollout drained into its local buffer but dropped on early
        return (after reaching the target size) would otherwise leak their
        baseline forever. Cannot prune by ``group_id`` magnitude: under
        ``retract`` completions arrive out of order, so a low id may still be
        in flight.
        """
        with self._lock:
            for gid in group_ids:
                self._baselines.pop(gid, None)


_weight_tracker = _WeightVersionTracker()


# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer: DataSource):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer: DataSource, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.concurrency = concurrency
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)  # Continuous output queue
        self.worker_thread = None
        self.state = GenerateState(args)
        # Own submitted-task accounting (read by the periodic metrics log).
        self.submitted_groups = 0  # groups (tasks) handed to generate_and_rm_group
        self.completed_groups = 0  # groups whose task finished

    async def query_sglang_metrics(self) -> tuple[dict[str, float], str | None]:

        url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/engine_metrics"
        try:
            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        return {}, f"HTTP {resp.status}: {body.strip()[:200]}"
                    return _parse_sglang_metrics(body), None
        except Exception as e:
            return {}, f"{type(e).__name__}: {e}"

    def _recycle_group(self, group, group_id) -> None:
        """Reset a failed/cancelled group and return it to the buffer for a retry."""
        try:
            for sample in group:
                sample.reset_for_retry()
            self.data_buffer.add_samples([group])
            if getattr(self.args, "max_weight_staleness", None) is not None:
                _weight_tracker.discard([group_id])  # baseline re-taken on resubmit
        except Exception as e:
            logger.warning(f"Failed to recycle group {group_id}: {e}")

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        print("Continuous async rollout worker started")

        active_tasks = set()
        task_meta: dict[asyncio.Task, tuple] = {}  # task -> (start, group, group_id)
        max_concurrent_tasks = self.args.over_sampling_batch_size
        group_id_counter = 0
        last_metrics_log = time.monotonic()

        while self.running:
            try:
                # Clean up finished tasks; force-cancel any wedged past the deadline.
                if active_tasks:
                    now = time.monotonic()
                    max_staleness = getattr(self.args, "max_weight_staleness", None)
                    for task in active_tasks:
                        start, _, gid = task_meta[task]
                        if task.done() or task.cancelling():
                            continue
                        if now - start > _GROUP_HARD_DEADLINE_SECONDS:
                            logger.warning(
                                f"group {gid} exceeded {_GROUP_HARD_DEADLINE_SECONDS:.0f}s deadline "
                                f"(ran {now - start:.0f}s); force-cancelling to reclaim the slot"
                            )
                            task.cancel()  # closes the httpx conn -> sglang aborts -> frees the slot
                        elif max_staleness is not None and _weight_tracker.exceeds(gid, max_staleness):
                            logger.info(f"group {gid} already exceeds max_weight_staleness; cancelling")
                            task.cancel()

                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        # Success is forwarded by the done-callback; here we log every
                        # cancelled/failed group in full and return it to the buffer.
                        start, group, gid = task_meta.pop(task)
                        if task.cancelled():
                            logger.warning(f"group {gid} cancelled after {now - start:.0f}s; recycling to buffer")
                            self._recycle_group(group, gid)
                        elif task.exception() is not None:
                            exc = task.exception()
                            logger.error(
                                f"group {gid} failed after {now - start:.0f}s: "
                                f"{type(exc).__name__}: {exc}; recycling to buffer",
                                exc_info=exc,
                            )
                            self._recycle_group(group, gid)
                    active_tasks -= done_tasks
                    self.completed_groups += len(done_tasks)

                # If active task count hasn't reached limit, try to get new data and start tasks
                while len(active_tasks) < max_concurrent_tasks and self.running:
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        # Snapshot the latest observed weight as this group's staleness
                        # baseline (only when the staleness filter is active downstream).
                        if getattr(self.args, "max_weight_staleness", None) is not None:
                            _weight_tracker.on_submit(group_id)

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                                timeout=_REQUEST_TIMEOUT,
                            )
                        )

                        # Add completion callback
                        def make_callback(gid):
                            def task_done_callback(done_task):
                                # Failures/cancellations are logged and recycled by the
                                # cleanup loop; here we only forward successful results.
                                if done_task.cancelled() or done_task.exception() is not None:
                                    return
                                result = done_task.result()
                                _weight_tracker.observe(result)
                                self.output_queue.put((gid, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        task_meta[task] = (time.monotonic(), group, group_id)
                        self.submitted_groups += 1
                        break

                # Periodically log own submission counters + live SGLang depth.
                now = time.monotonic()
                if now - last_metrics_log >= METRICS_LOG_INTERVAL_SECONDS:
                    last_metrics_log = now
                    sg, sg_err = await self.query_sglang_metrics()
                    if sg_err is None:
                        sglang_str = (
                            f"running={sg.get('sglang_num_running_reqs')}, "
                            f"queued={sg.get('sglang_num_queue_reqs')}, "
                            f"received_total={sg.get('sglang_num_requests_total')}"
                        )
                    else:
                        sglang_str = f"unavailable ({sg_err})"
                    logger.info(
                        f"[fully_async worker] submitted_groups={self.submitted_groups}, "
                        f"completed_groups={self.completed_groups}, active_tasks={len(active_tasks)}, "
                        f"output_queue={self.get_queue_size()} | sglang: {sglang_str}"
                    )

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in continuous worker loop: {type(e).__name__}: {e}", exc_info=e)
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def get_completed_groups(self, max_groups: int | None = None) -> list[tuple]:
        """Get completed sample groups"""
        completed = []
        while max_groups is None or len(completed) < max_groups:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()


async def generate_rollout_async(args, rollout_id: int, data_buffer: DataSource) -> list[list[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    completed_groups = {}
    do_print = True
    stale_groups_recycled = 0
    staleness_values = []
    use_staleness_filter = getattr(args, "max_weight_staleness", None) is not None

    # Dynamic sampling filter: drop uninformative groups (e.g. zero reward std), mirroring sglang_rollout.
    dynamic_filter = load_function(args.dynamic_sampling_filter_path)
    dynamic_filter_drops = 0
    dynamic_filter_drop_reasons = {}

    # Hook to process all decisioned groups (kept + dynamic-filtered) at the end of the
    # rollout, mirroring sglang_rollout.generate_rollout. Loaded the same way as the
    # dynamic filter above; None when the arg is unset.
    all_samples_process = load_function(args.rollout_all_samples_process_path)

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")
    if use_staleness_filter:
        print(f"Staleness filter enabled: max_weight_staleness={args.max_weight_staleness}")
    if dynamic_filter is not None:
        print(f"Dynamic sampling filter enabled: {args.dynamic_sampling_filter_path}")

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    while len(data) < target_data_size:
        # Collect completed results
        remaining_slots = target_data_size - len(data) - len(completed_groups)
        completed = worker.get_completed_groups(max_groups=max(0, remaining_slots))

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        # Process completed groups in order (try to maintain order, but not strict requirement)
        processed_any = False

        # Process all available completed groups
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            # Pop this group's staleness (cur_weight - submission baseline) exactly once,
            # for every consumed group, so aborted / dropped groups don't leak baselines.
            group_staleness = _weight_tracker.staleness(group_id) if use_staleness_filter else None

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    for s in group:
                        s.reset_for_retry()
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer", flush=True)
                except Exception as e:
                    print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                # don't count as processed for training
                continue

            # Staleness filter: discard groups the latest observed weight has moved too far past.
            if group_staleness is not None:
                staleness_values.append(group_staleness)
                if group_staleness > args.max_weight_staleness:
                    try:
                        for s in group:
                            s.reset_for_retry()
                        data_buffer.add_samples([group])
                    except Exception as e:
                        logger.warning(f"Failed to recycle stale group {group_id}: {e}")
                    stale_groups_recycled += 1
                    logger.info(
                        f"Recycled stale group {group_id} "
                        f"(cur_weight={_weight_tracker.cur_weight}, "
                        f"staleness={group_staleness} > max={args.max_weight_staleness})"
                    )
                    continue

            # Record every group that reaches the filter decision (kept + dropped) so
            # rollout_all_samples_process_path can see all of them. Mirrors `all_data`
            # in sglang_rollout.generate_rollout. Aborted/stale groups above are recycled
            # to the buffer (not decisioned here), so they are intentionally excluded and
            # will be captured in the rollout where they finally resolve.
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                dynamic_filter_drops += 1
                # show current group num and dropped group num
                dynamic_filter_drop_reasons[dynamic_filter_output.reason] = (
                    dynamic_filter_drop_reasons.get(dynamic_filter_output.reason, 0) + 1
                )
                print(
                    f"Current group num: {len(data)}, Dropped group num: {dynamic_filter_drops}, Dropped reasons: {dynamic_filter_drop_reasons}",
                    flush=True,
                )

                continue

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            data.append(group)
            processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    # This should only happen if future changes leave a local backlog after
    # target_data_size is reached. Those groups are not returned for training.
    if completed_groups:
        logger.warning(
            f"Dropping {len(completed_groups)} completed groups that were drained but not consumed "
            f"after reaching target_data_size={target_data_size}"
        )
        if use_staleness_filter:
            _weight_tracker.discard(list(completed_groups.keys()))

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")
    if stale_groups_recycled > 0 or staleness_values:
        avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
        print(
            f"Staleness stats: recycled={stale_groups_recycled}, "
            f"avg_staleness={avg_staleness:.1f}, "
            f"max_staleness={max(staleness_values) if staleness_values else 0}"
        )
    if dynamic_filter is not None:
        print(f"Dynamic filter stats: dropped={dynamic_filter_drops}")

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)

    # Hand every decisioned group (kept + dynamic-filtered) to the user hook before
    # returning, mirroring sglang_rollout.generate_rollout. `data_buffer` plays the
    # `data_source` role in the standard fn(args, all_samples, data_source) signature.
    if all_samples_process is not None:
        all_data = sorted(all_data, key=lambda group: group[0].index)
        all_samples_process(args, all_data, data_buffer)

    return data


def generate_rollout_fully_async(args, rollout_id, data_buffer: DataSource, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


# Register exit cleanup function

atexit.register(stop_global_worker)
