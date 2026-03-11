# this is the default way of interacting with the industrial automaton codebase

import inspect
import numpy as np
from typing import Any
from functools import partial

import jax

from industrial_automaton.utils import ANSI
from industrial_automaton.models_jax import (
    BabyNTM, BabyNTMModelConfig,
    SuzgunStackRNN, SuzgunStackRNNConfig,
    Transformer, TransformerConfig,
    LSTM, LSTMConfig,
    ModelPipeline,
)
# TapeRNN lives in models_torch — imported lazily in the tape_rnn branch below
from industrial_automaton.vocab import SIZE as VOCAB_SIZE
from industrial_automaton import tasks
from industrial_automaton.tasks import generate_variable_dataset, create_batch_iterator
from industrial_automaton.tasks.generators import create_online_batch_generator
from industrial_automaton.curriculum import (
    FixedCurriculum,
    LinearCurriculum,
    AdaptiveCurriculum,
    MultiTaskCurriculum,
    UniformCurriculum,
)


def _build_task_data(task_name_clean, settings, task_kwargs, eval_kwargs=None):
    """Build training + eval datasets for a single task. Returns (task_fn, train_data, eval_data, task_metadata)."""
    from industrial_automaton.tasks import MASTER_REGISTRY

    task_fn = partial(
        getattr(tasks, f"generate_{task_name_clean}"),
        **task_kwargs
    )
    sig = inspect.signature(task_fn.func)
    example_kwargs = {"batch_size": 1}
    task_entry = MASTER_REGISTRY.get(task_name_clean)
    if task_entry and task_entry.baseline:
        example_kwargs.update(task_entry.baseline)
        if "length" not in example_kwargs and "length" in sig.parameters:
            example_kwargs["length"] = settings.max_seqlen
    elif "length" in sig.parameters:
        example_kwargs["length"] = settings.max_seqlen

    train_kw = task_kwargs.copy()
    if settings.task_kwargs:
        train_kw.update(settings.task_kwargs)
    train_inputs, train_targets, train_loss_mask = generate_variable_dataset(
        task_fn, train_kw, settings.dataset_size, settings.max_seqlen,
        hard_array_limit=settings.hard_array_limit, task_name=task_name_clean,
    )

    eval_kw = (eval_kwargs or task_kwargs).copy()
    if settings.task_kwargs:
        eval_kw.update(settings.task_kwargs)
    if settings.eval_task_kwargs:
        eval_kw.update(settings.eval_task_kwargs)
    eval_inputs, eval_targets, eval_loss_mask = generate_variable_dataset(
        task_fn, eval_kw, settings.eval_dataset_size, settings.eval_max_seqlen,
        hard_array_limit=settings.hard_array_limit, task_name=task_name_clean,
    )

    task_metadata = None
    if task_entry is not None and task_entry.output_vocab is not None:
        task_metadata = {"output_vocab": task_entry.output_vocab}

    return task_fn, (train_inputs, train_targets, train_loss_mask), (eval_inputs, eval_targets, eval_loss_mask), task_metadata


def _round_robin_iterator(iterators):
    """Yield batches in round-robin order across multiple iterators."""
    while True:
        for it in iterators:
            yield next(it)


def cli():
    # Import here to not mess up other CLI commands
    from industrial_automaton.config import Settings
    from industrial_automaton.tasks import MASTER_REGISTRY
    from industrial_automaton.trainer_jx import Trainer, loss_fn, TrainingDivergedError, eval_fn

    # Create settings instance (will parse CLI args automatically)
    settings = Settings()

    # pretty print all the settings
    print("="*20 + f" {ANSI.bold('Settings')} " + "="*20)
    for k, v in sorted(settings.model_dump().items()):
        print(f"{ANSI.blue(k)}: {ANSI.italic(v)}")
    print("="*50)

    # --- Multi-task path ---
    if settings.tasks:
        task_names = [t.strip().replace("generate_", "") for t in settings.tasks.split(",")]
        print(f"     {ANSI.bold('Multi-task')}: {task_names}")

        all_train_iters = []
        all_eval_data = []
        all_task_metadata = []

        for tname in task_names:
            tk = {"rng": np.random.default_rng(settings.seed)}
            _, train_data, eval_data, tmeta = _build_task_data(tname, settings, tk)
            ti, tt, tm = train_data
            print(f"{ANSI.blue(f'  [{tname}]')}: train={ti.shape}, eval={eval_data[0].shape}")
            all_train_iters.append(create_batch_iterator(ti, tt, tm, batch_size=settings.batch_size, shuffle=True, seed=settings.seed))
            all_eval_data.append((tname, eval_data, tmeta))
            all_task_metadata.append(tmeta)

        combined_iter = _round_robin_iterator(all_train_iters)
        # Use first task's eval for Trainer's built-in early stopping
        first_eval_inputs, first_eval_targets, first_eval_mask = all_eval_data[0][1]
        first_task_meta = all_task_metadata[0]

        # Build model (shared across all tasks)
        if settings.model == "tape_rnn":
            raise ValueError("tape_rnn uses TorchTrainer and is not supported in multi-task mode yet. Use a single --task instead.")
        config_cls, model_cls = {
            "baby_ntm": (BabyNTMModelConfig, BabyNTM),
            "suzgun_stack_rnn": (SuzgunStackRNNConfig, SuzgunStackRNN),
            "transformer": (TransformerConfig, Transformer),
            "lstm": (LSTMConfig, LSTM),
        }.get(settings.model, (None, None))
        if config_cls is None:
            raise ValueError(f"Unknown model: {settings.model}")
        model_kwargs = settings.model_kwargs.copy()
        if settings.embedding_type == "one_hot":
            model_kwargs["embedding_dim"] = VOCAB_SIZE
        config = config_cls(**model_kwargs)
        model = ModelPipeline(config, model_cls, embedding_type=settings.embedding_type, key=jax.random.PRNGKey(settings.seed))
        num_params = sum(x.size for x in jax.tree.leaves(model) if isinstance(x, jax.numpy.ndarray))
        print(f"{ANSI.green('Model params')}: {num_params/1e3:.3f} K")

        trainer = Trainer(
            model, settings,
            task_metadata=first_task_meta,
            eval_inputs=first_eval_inputs,
            eval_labels=first_eval_targets,
            eval_loss_mask=first_eval_mask,
            eval_dataset_size=settings.eval_dataset_size,
        )

        if settings.load_ckpt:
            print(f"{ANSI.blue('Loading checkpoint from')} {settings.load_ckpt}...")
            trainer.load(settings.load_ckpt)

        print(f"\n" + "="*20 + f" {ANSI.bold('Multi-task Training')} " + "="*20)
        try:
            history = trainer.fit(data_generator=combined_iter)
        except TrainingDivergedError as e:
            print(f"{ANSI.red('Training diverged: ' + str(e))}")
            return

        print("\n" + "="*50)
        # Per-task evaluation
        print(f"{ANSI.bold('Per-task evaluation:')}")
        for tname, (ei, et, em), tmeta in all_eval_data:
            metrics = eval_fn(trainer.state.model, settings.eval_dataset_size, ei, et, em, task_metadata=tmeta)
            tok_acc = float(metrics.aux["token_accuracy"])
            seq_acc = float(metrics.aux["sequence_accuracy"])
            print(f"  {ANSI.blue(tname)}: tok_acc={tok_acc:.4f} | seq_acc={seq_acc:.4f}")
            print(f"Final token accuracy [{tname}]: {tok_acc:.4f}")
            print(f"Final sequence accuracy [{tname}]: {seq_acc:.4f}")
        return

    # --- Single-task path (original) ---
    # Generate a single example
    print(f"     {ANSI.bold('Task')}: {settings.task}")
    task_name_clean = settings.task.replace("generate_", "")  # Remove prefix if present
    task_kwargs: dict[str, Any] = {"rng": np.random.default_rng(settings.seed)}

    task_fn = partial(
        getattr(tasks, f"generate_{task_name_clean}"),
        **task_kwargs
    )

    # Check if the task function accepts a 'length' parameter and set example_kwargs
    sig = inspect.signature(task_fn.func) # Get signature from the wrapped function
    example_kwargs = {"batch_size": 1}

    task_entry = MASTER_REGISTRY.get(task_name_clean)
    if task_entry and task_entry.baseline:
        example_kwargs.update(task_entry.baseline)
        # Ensure 'length' is present, fallback to a reasonable default if not in baseline
        if 'length' not in example_kwargs and "length" in sig.parameters:
            example_kwargs['length'] = settings.max_seqlen # Use a default from settings if task has length param
    elif "length" in sig.parameters:
        example_kwargs["length"] = settings.max_seqlen # Fallback if no task_entry or baseline

    # Special handling for modular_arithmetic example length if it somehow ends up even
    if task_name_clean == 'modular_arithmetic' and 'length' in example_kwargs and example_kwargs['length'] % 2 == 0:
        example_kwargs['length'] += 1

    example = task_fn(**example_kwargs)

    # Print the example
    print(f"{ANSI.bold('Formatted')}: {example['input_formatted'][0]} -> {example['output_formatted'][0]}")
    print(f"{ANSI.bold('      Raw')}: {example['input'][0]} -> {example['output'][0]}")

    # Now we will train the model
    print("\n" + "="*20 + f" {ANSI.bold('Training')} " + "="*20)

    # Prepare model config
    print(f"{ANSI.blue('Vocab size')}: {VOCAB_SIZE}")

    # Initialize model based on settings
    # tape_rnn / tallerman → PyTorch + TorchTrainer (MPS); all others → JAX + Trainer
    use_torch = (settings.model in ("tape_rnn", "tallerman"))

    model_kwargs = settings.model_kwargs.copy()

    if use_torch:
        import torch
        from industrial_automaton.models_torch.tape import TapeRNN, TapeRNNConfig
        from industrial_automaton.models_torch.tallerman import Tallerman, TallermanConfig
        from industrial_automaton.models_torch.common import ModelPipeline as TorchModelPipeline
        from industrial_automaton.trainer_torch import TorchTrainer, make_generator

        if settings.embedding_type == "one_hot":
            print(f"{ANSI.blue('Using one_hot embedding. Embedding dimension set to vocab size: ' + str(VOCAB_SIZE))}")
            model_kwargs["embedding_dim"] = VOCAB_SIZE

        if settings.model == "tape_rnn":
            config = TapeRNNConfig(**model_kwargs)
            model_cls = TapeRNN
        elif settings.model == "tallerman":
            config = TallermanConfig(**model_kwargs)
            model_cls = Tallerman
        else:
            raise ValueError(f"Unknown torch model: {settings.model}")

        generator = make_generator(settings.seed)
        model = TorchModelPipeline(config, model_cls, embedding_type=settings.embedding_type, generator=generator)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{ANSI.green('Model')} (via TorchPipeline): {settings.model} [PyTorch/MPS]")
        print(f"{ANSI.green('Model params')}: {num_params/1e3:.3f} K")
    else:
        config_cls, model_cls = {
            "baby_ntm": (BabyNTMModelConfig, BabyNTM),
            "suzgun_stack_rnn": (SuzgunStackRNNConfig, SuzgunStackRNN),
            "transformer": (TransformerConfig, Transformer),
            "lstm": (LSTMConfig, LSTM),
        }.get(settings.model, (None, None))
        if config_cls is None:
            raise ValueError(f"Unknown model: {settings.model}")

        if settings.model == "transformer" and "max_seq_len" not in model_kwargs:
            model_kwargs["max_seq_len"] = max(settings.max_seqlen, settings.eval_max_seqlen) * 2

        if settings.embedding_type == "one_hot":
            print(f"{ANSI.blue('Using one_hot embedding. Embedding dimension set to vocab size: ' + str(VOCAB_SIZE))}")
            model_kwargs["embedding_dim"] = VOCAB_SIZE

        config = config_cls(**model_kwargs)
        model = ModelPipeline(config, model_cls, embedding_type=settings.embedding_type, key=jax.random.PRNGKey(settings.seed))
        num_params = sum(x.size for x in jax.tree.leaves(model) if isinstance(x, jax.numpy.ndarray))
        print(f"{ANSI.green('Model')} (via Pipeline): {settings.model}")
        print(f"{ANSI.green('Model params')}: {num_params/1e3:.3f} K")

    # --- Dataset Generation ---
    # Generate Training Data
    print(f"{ANSI.blue('Generating training dataset...')}")
    train_kwargs = task_kwargs.copy()
    if settings.task_kwargs:
        train_kwargs.update(settings.task_kwargs)

    train_inputs, train_targets, train_loss_mask = generate_variable_dataset(
        task_fn,
        train_kwargs,
        settings.dataset_size,
        settings.max_seqlen,
        hard_array_limit=settings.hard_array_limit,
        task_name=task_name_clean,
    )
    print(f"{ANSI.blue('Train Dataset')}: inputs={train_inputs.shape} targets={train_targets.shape} mask={train_loss_mask.shape}")

    # Generate Evaluation Data
    print(f"{ANSI.blue('Generating evaluation dataset...')}")
    eval_kwargs = task_kwargs.copy()
    if settings.task_kwargs:
        eval_kwargs.update(settings.task_kwargs)
    if settings.eval_task_kwargs:
        eval_kwargs.update(settings.eval_task_kwargs)

    eval_inputs, eval_targets, eval_loss_mask = generate_variable_dataset(
        task_fn,
        eval_kwargs,
        settings.eval_dataset_size,
        settings.eval_max_seqlen,
        hard_array_limit=settings.hard_array_limit,
        task_name=task_name_clean,
    )
    print(f"{ANSI.blue('Eval Dataset')}: inputs={eval_inputs.shape} targets={eval_targets.shape} mask={eval_loss_mask.shape}")

    # Create data iterator that yields batches from the fixed dataset
    data_iterator = create_batch_iterator(
        train_inputs,
        train_targets,
        train_loss_mask,
        batch_size=settings.batch_size,
        shuffle=True,
        seed=settings.seed,
    )

    # Initialize curriculum strategy if specified
    curriculum = None
    if settings.curriculum_type:
        curriculum_kwargs = settings.curriculum_kwargs or {}

        if settings.curriculum_type == "fixed":
            # Fixed curriculum - no adaptation
            fixed_params = curriculum_kwargs.get("fixed_params", {"length": settings.max_seqlen})
            curriculum = FixedCurriculum(fixed_params=fixed_params)
            print(f"{ANSI.blue('Curriculum')}: Fixed - {fixed_params}")

        elif settings.curriculum_type == "linear":
            # Linear curriculum - increase difficulty linearly
            initial_bound = curriculum_kwargs.get("initial_bound", 10)
            increase_freq = curriculum_kwargs.get("increase_freq", 500)
            increase_amount = curriculum_kwargs.get("increase_amount", 5)
            curriculum = LinearCurriculum(
                initial_bound=initial_bound,
                max_bound=settings.max_seqlen,
                increase_freq=increase_freq,
                increase_amount=increase_amount,
            )
            print(f"{ANSI.blue('Curriculum')}: Linear - start={initial_bound}, max={settings.max_seqlen}, freq={increase_freq}, amount={increase_amount}")

        elif settings.curriculum_type == "adaptive":
            # Adaptive curriculum - performance-based
            advance_threshold = curriculum_kwargs.get("advance_threshold", 0.9)
            advance_streak = curriculum_kwargs.get("advance_streak", 3)
            backoff_threshold = curriculum_kwargs.get("backoff_threshold", 2.0)
            ema_decay = curriculum_kwargs.get("ema_decay", 0.9)
            step_size = curriculum_kwargs.get("step_size", 5)
            curriculum = AdaptiveCurriculum(
                advance_threshold=advance_threshold,
                advance_streak=advance_streak,
                backoff_threshold=backoff_threshold,
                ema_decay=ema_decay,
                step_size=step_size,
            )
            print(f"{ANSI.blue('Curriculum')}: Adaptive - advance_threshold={advance_threshold}, streak={advance_streak}, backoff={backoff_threshold}, ema={ema_decay}")

        elif settings.curriculum_type == "multitask":
            # Multi-task curriculum
            task_names = curriculum_kwargs.get("task_names", [])
            if not task_names:
                raise ValueError("MultiTaskCurriculum requires 'task_names' in curriculum_kwargs")
            selection_mode = curriculum_kwargs.get("selection_mode", "first_unsolved")
            mastery_threshold = curriculum_kwargs.get("mastery_threshold", 0.95)
            mastered_revisit_ratio = curriculum_kwargs.get("mastered_revisit_ratio", 0.05)
            curriculum = MultiTaskCurriculum(
                task_names=task_names,
                selection_mode=selection_mode,
                mastery_threshold=mastery_threshold,
                mastered_revisit_ratio=mastered_revisit_ratio,
            )
            print(f"{ANSI.blue('Curriculum')}: MultiTask - tasks={task_names}, mode={selection_mode}")

        elif settings.curriculum_type == "uniform":
            # Uniform curriculum - sample length uniformly each step (Delétang et al. 2023)
            min_len = curriculum_kwargs.get("min_bound", 1)
            curriculum = UniformCurriculum()
            print(f"{ANSI.blue('Curriculum')}: Uniform - lengths {min_len}-{settings.max_seqlen} (online generation)")

        else:
            raise ValueError(f"Unknown curriculum_type: {settings.curriculum_type}. Choose from: fixed, linear, adaptive, multitask, uniform")

    # Build task_metadata with output_vocab mask for logit masking
    task_metadata = None
    if task_entry is not None and task_entry.output_vocab is not None:
        task_metadata = {"output_vocab": task_entry.output_vocab}

    # Length-varying curricula (uniform, linear, adaptive) all need online generation —
    # the dataset length changes every eval, so we can't pre-generate a fixed dataset.
    # Only fixed/multitask curricula (or no curriculum) use the pre-generated iterator.
    use_online = settings.curriculum_type in ("uniform", "linear", "adaptive")

    if use_online:
        # Online generator: callable (length) -> batch, used by trainer each step
        data_source = create_online_batch_generator(
            task_fn,
            train_kwargs,
            batch_size=settings.batch_size,
            hard_array_limit=settings.hard_array_limit,
            task_name=task_name_clean,
        )
        # Initialize curriculum state — max_seqlen is the starting (min) length for
        # adaptive/linear; the curriculum grows it up to the task's natural max.
        min_len = (settings.curriculum_kwargs or {}).get("min_bound", settings.max_seqlen)
        max_len = (settings.curriculum_kwargs or {}).get("max_bound", settings.max_seqlen)
        from industrial_automaton.curriculum import init_curriculum_state
        curriculum_state = init_curriculum_state(
            strategy=curriculum,
            min_bound=min_len,
            max_bound=max_len,
            initial_bound=min_len,
        )
        print(f"{ANSI.blue('Curriculum')}: online generation, "
              f"L={min_len}→{max_len}, type={settings.curriculum_type}")
    else:
        data_source = data_iterator
        curriculum_state = None

    # Initialize trainer — tape_rnn uses TorchTrainer, everything else uses JAX Trainer
    if use_torch:
        trainer = TorchTrainer(
            model,
            settings,
            task_metadata=task_metadata,
            eval_inputs=eval_inputs,
            eval_labels=eval_targets,
            eval_loss_mask=eval_loss_mask,
            eval_dataset_size=settings.eval_dataset_size,
        )
    else:
        trainer = Trainer(
            model,
            settings,
            task_metadata=task_metadata,
            curriculum=curriculum,
            eval_inputs=eval_inputs,
            eval_labels=eval_targets,
            eval_loss_mask=eval_loss_mask,
            eval_dataset_size=settings.eval_dataset_size,
        )

    if settings.load_ckpt:
        print(f"{ANSI.blue('Loading checkpoint from')} {settings.load_ckpt}...")
        trainer.load(settings.load_ckpt)

    # Inject pre-built curriculum state for curricula that need it (JAX only)
    if not use_torch and use_online and curriculum_state is not None:
        import equinox as eqx
        trainer.state = eqx.tree_at(
            lambda s: s.curriculum_state,
            trainer.state,
            curriculum_state,
        )

    # Train
    print(f"{ANSI.bold('Starting training...')}")
    try:
        history = trainer.fit(
            data_generator=data_source,
        )

        print("\n" + "="*50)
    
        # Add 5 examples of actual random I/O for the model. I want to see what
        # the model sees and predicts.

        print(f"{ANSI.green('Training complete!')}")
        print(f"Final loss: {float(history[-1].loss):.4f}")
        if history[-1].aux:
            tok_acc = history[-1].aux.get("token_accuracy")
            seq_acc = history[-1].aux.get("sequence_accuracy")
            if tok_acc is not None:
                print(f"Final token accuracy: {float(tok_acc):.4f}")
            if seq_acc is not None:
                print(f"Final sequence accuracy: {float(seq_acc):.4f}")

        # Add 5 examples of actual random I/O for the model. I want to see what
        # the model sees and predicts.
        from industrial_automaton.vocab import pretty, YIELD, PAD

        print("\n" + "="*15 + f" {ANSI.bold('Model Predictions (Eval Set)')} " + "="*15)
        indices = np.random.choice(len(eval_inputs), size=min(3, len(eval_inputs)), replace=False)
        for i, idx in enumerate(indices):
            x_raw = eval_inputs[idx]
            tgt_raw = eval_targets[idx]
            mask_raw = eval_loss_mask[idx]

            if use_torch:
                import torch
                inf_model = trainer.model
                inf_model.eval()
                with torch.no_grad():
                    x_t = torch.as_tensor(x_raw, dtype=torch.long, device=trainer.device)
                    state = inf_model.init_state(device=trainer.device)
                    logits, _ = inf_model(x_t, state)
                    preds = logits.argmax(dim=-1).cpu().numpy()
            else:
                import jax.numpy as jnp
                state = trainer.state.model.init_state()
                logits, _ = trainer.state.model(x_raw, state)
                preds = jnp.argmax(logits, axis=-1)

            # Find YIELD position to split input/output display
            x_list = [int(t) for t in x_raw if int(t) != PAD]
            yield_pos = next((j for j, t in enumerate(x_list) if t == YIELD), len(x_list))
            inp_str = pretty(x_list[:yield_pos])

            mask_positions = np.where(mask_raw == 1)[0]
            if len(mask_positions) > 0:
                tgt_tokens = [int(tgt_raw[p]) for p in mask_positions]
                pred_tokens = [int(preds[p]) for p in mask_positions]
                tgt_str = pretty(tgt_tokens)
                pred_str = pretty(pred_tokens)
            else:
                tgt_str = "(none)"
                pred_str = "(none)"

            print(f"\n{ANSI.bold(f'Example {i+1}:')}")
            print(f"  {ANSI.blue('Input')}:  {inp_str}")
            print(f"  {ANSI.green('Target')}: {tgt_str}")
            print(f"  {ANSI.red('Pred')}:   {pred_str}")
        print("\n" + "="*50)
    except TrainingDivergedError as e:
        print("\n" + "="*50)
        print(f"{ANSI.red('Training diverged!')}")
        print(f"{ANSI.red(f'Error: {str(e)}')}")
        print(f"Task: {settings.task}")
        print(f"Model: {settings.model}")
        print("Consider:")
        print("  - Reducing learning rate")
        print("  - Adjusting model capacity")
        print("  - Checking task constraints")
        return



def print_model_configs():
    print("="*20 + f" {ANSI.bold('Model Configurations')} " + "="*20)
    print(f"Legend: {ANSI.red('*')} = Required field")
    
    from industrial_automaton.models_torch.tape import TapeRNNConfig as TorchTapeRNNConfig
    from industrial_automaton.models_torch.tallerman import TallermanConfig
    configs = {
        "baby_ntm": BabyNTMModelConfig,
        "suzgun_stack_rnn": SuzgunStackRNNConfig,
        "tape_rnn [torch/mps]": TorchTapeRNNConfig,
        "tallerman [torch/mps]": TallermanConfig,
        "transformer": TransformerConfig,
        "lstm": LSTMConfig,
    }
    
    for name, config_cls in configs.items():
        print(f"\n{ANSI.bold(ANSI.blue(name))}")
        print("-" * len(name))
        
        schema = config_cls.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        for field_name, field_info in properties.items():
            type_str = field_info.get("type", "any")
            default = field_info.get("default", "REQUIRED" if field_name in required else "None")
            
            req_marker = ANSI.red("*") if field_name in required else " "
            print(f"{req_marker} {ANSI.green(field_name)} ({type_str}): default={default}")

    
    print("\n" + "="*60)
    print(f"Usage: uv run inmaton --model <name> --model_kwargs '{{\"key\": \"value\"}}'")


def print_task_configs():
    print("="*20 + f" {ANSI.bold('Task Configurations')} " + "="*20)
    print(f"Legend: {ANSI.red('*')} = Required field")

    task_funcs = {
        name.replace("generate_", ""): (obj, obj.__code__.co_filename.split("/")[-1])
        for name, obj in inspect.getmembers(tasks, inspect.isfunction)
        if name.startswith("generate_")
    }

    for task_name, (func, task) in sorted(task_funcs.items()):
        print(f"\n{ANSI.underline(ANSI.bold(ANSI.blue(task_name)))} (task: {ANSI.underline(task)})")

        if func.__doc__:
            desc = func.__doc__.strip().split("\n\n")[0]
            print(f"  {ANSI.italic(desc)}")
            
        sig = inspect.signature(func)
        has_params = False
        for param_name, param in sig.parameters.items():
            if param_name in ("batch_size", "length", "rng"):
                continue
            
            has_params = True
            type_str = "any"
            if param.annotation != inspect.Parameter.empty:
                type_str = getattr(param.annotation, "__name__", str(param.annotation))
                
            default = "REQUIRED" if param.default == inspect.Parameter.empty else param.default
            req_marker = ANSI.red("*") if param.default == inspect.Parameter.empty else " "
            
            print(f"{req_marker} {ANSI.green(param_name)} ({type_str}): default={default}")
            
        if not has_params:
            print(f"  {ANSI.italic('No extra task parameters.')}")

    print("\n" + "="*60)
    print(f"Usage: uv run inmaton --task <name> -n <n>")

def print_curriculum_configs():
    import dataclasses

    print("="*20 + f" {ANSI.bold('Curriculum Configurations')} " + "="*20)
    print(f"Legend: {ANSI.red('*')} = Required field")

    curriculum_configs = {
        "fixed": FixedCurriculum,
        "linear": LinearCurriculum,
        "adaptive": AdaptiveCurriculum,
        "multitask": MultiTaskCurriculum,
        "uniform": UniformCurriculum,
    }

    for name, config_cls in curriculum_configs.items():

        print(f"\n{ANSI.bold(ANSI.blue(name))}")
        print("-" * len(name))

        # Use dataclasses.fields() to inspect the class
        fields = dataclasses.fields(config_cls)

        for field in fields:
            field_name = field.name
            type_str = field.type.__name__ if hasattr(field.type, '__name__') else str(field.type)

            # Check if field has a default value
            if field.default != dataclasses.MISSING:
                default = field.default
                req_marker = " "
            elif field.default_factory != dataclasses.MISSING:
                default = f"<factory: {field.default_factory.__name__}>"
                req_marker = " "
            else:
                default = "REQUIRED"
                req_marker = ANSI.red("*")

            print(f"{req_marker} {ANSI.green(field_name)} ({type_str}): default={default}")

    print("\n" + "="*60)
    print(f"Usage: uv run inmaton --curriculum_type <name> --curriculum_kwargs '{{\"key\": \"value\"}}'")
