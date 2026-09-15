"""AvaCore backend for Miles' public rollout/eval function hooks."""

import asyncio
import copy

from miles.rollout.avacore_rollout import generate_group
from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.utils.async_utils import run
from miles.utils.misc import load_function
from miles.utils.types import Sample


async def _finish(native, args, state, rollout_id, tasks):
    state.aborted = True
    for task in tasks:
        task.cancel()
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        # Native abort handles router versions and stops server-side requests.
        # No AvaCore task is passed to its partial-rollout collection logic.
        state.pendings.clear()
        state.aborted = False
        await native.abort(args, rollout_id)
    finally:
        state.reset()


async def train_rollout(native, args, state, rollout_id, data_source):
    if not args.rollout_global_dataset:
        raise ValueError("AvaCore requires the Miles global dataset")
    if args.rollout_batch_size < 1 or args.over_sampling_batch_size < 1:
        raise ValueError("Rollout batch sizes must be positive")
    await native.dumper_utils.configure_sglang(args)
    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None
    metrics = MetricGatherer()
    accepted, observed, pending, tasks = [], [], set(), set()
    target = args.rollout_batch_size
    try:
        while len(accepted) < target:
            while len(accepted) + len(pending) < target:
                groups = data_source.get_samples(args.over_sampling_batch_size)
                if not groups:
                    raise ValueError("AvaCore rollout data source returned no prompt groups")
                for group in groups:
                    if len(group) != args.n_samples_per_prompt:
                        raise ValueError("AvaCore requires fixed prompt group cardinality")
                    task = asyncio.create_task(generate_group(args, state, group, state.sampling_params.copy()))
                    pending.add(task)
                    tasks.add(task)
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                group = task.result()
                if len(group) != args.n_samples_per_prompt or any(not episode for episode in group):
                    raise ValueError("AvaCore returned an invalid episode group")
                observed.append(group)
                if any(sample.status == Sample.Status.ABORTED for episode in group for sample in episode):
                    metrics.on_dynamic_filter_drop("aborted")
                    continue
                decision = call_dynamic_filter(dynamic_filter, args, group)
                if not decision.keep:
                    metrics.on_dynamic_filter_drop(decision.reason)
                elif len(accepted) < target:
                    accepted.append(group)
    finally:
        await _finish(native, args, state, rollout_id, tasks)

    key = lambda group: group[0][0].index
    accepted.sort(key=key)
    observed.sort(key=key)
    # Record episode statistics before the batch hook adds zero-mask padding.
    rewards = [episode[0].reward for group in observed for episode in group]
    if args.rollout_sample_filter_path:
        load_function(args.rollout_sample_filter_path)(args, accepted)
    if args.rollout_all_samples_process_path:
        load_function(args.rollout_all_samples_process_path)(args, observed, data_source.get_samples)
    await native.recompute_samples_rollout_logprobs_via_prefill(
        args, [sample for group in accepted for episode in group for sample in episode],
        url=native.get_model_url(args, "default"), sampling_params=state.sampling_params,
    )
    output_metrics = metrics.collect()
    if rewards:
        output_metrics["rollout/prefilter_reward"] = sum(rewards) / len(rewards)
    return RolloutFnTrainOutput(samples=accepted, metrics=output_metrics)


async def eval_rollout(native, args, state, rollout_id):
    results, tasks = {}, set()
    try:
        for config in args.eval_datasets or []:
            if config.custom_generate_function_path:
                raise ValueError("Configure AvaCore evaluation generation through TOML")
            dataset = native.Dataset(
                path=config.path, tokenizer=state.tokenizer, processor=state.processor,
                max_length=args.eval_max_prompt_len, prompt_key=config.input_key,
                label_key=config.label_key, multimodal_keys=args.multimodal_keys,
                metadata_key=config.metadata_key, tool_key=config.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            )
            sampling = {
                **state.sampling_params, "temperature": config.temperature,
                "top_p": config.top_p, "top_k": config.top_k,
                "max_new_tokens": config.max_response_len,
            }
            dataset_tasks = []
            for prompt in dataset.samples:
                for repeat in range(config.n_samples_per_eval_prompt):
                    sample = copy.deepcopy(prompt)
                    sample.index = len(dataset_tasks)
                    sample.metadata = config.inject_metadata(sample.metadata)
                    params = sampling.copy()
                    if getattr(args, "sglang_enable_deterministic_inference", False):
                        params["sampling_seed"] = args.rollout_seed + repeat
                    task = asyncio.create_task(generate_group(args, state, [sample], params, evaluation=True))
                    dataset_tasks.append(task)
                    tasks.add(task)
            groups = await asyncio.gather(*dataset_tasks)
            # One final representative per episode, not one reward per call.
            samples = [group[0][-1] for group in groups]
            samples.sort(key=lambda sample: sample.index)
            results[config.name] = {
                "rewards": [sample.reward for sample in samples],
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in samples],
                "samples": samples,
            }
    finally:
        await _finish(native, args, state, rollout_id, tasks)
    return RolloutFnEvalOutput(data=results)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from miles.rollout import sglang_rollout as native

    state = native.GenerateState(args)
    if evaluation:
        return run(eval_rollout(native, args, state, rollout_id))
    return run(train_rollout(native, args, state, rollout_id, data_source))
