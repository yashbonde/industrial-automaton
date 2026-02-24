# this is the default way of interacting with the industrial automaton codebase

import time
import inspect
import numpy as np
from typing import Any
from functools import partial

import jax
import jax.numpy as jnp

from industrial_automaton.utils import ANSI
from industrial_automaton.models.baby_ntm import BabyNTM, BabyNTMModelConfig
from industrial_automaton.models.suzgun_stack_rnn import SuzgunStackRNN, SuzgunStackRNNConfig
from industrial_automaton.models.tape_rnn import TapeRNN, TapeRNNConfig
from industrial_automaton.models.transformer import Transformer, TransformerConfig
from industrial_automaton.models.lstm import LSTM, LSTMConfig
from industrial_automaton.vocab import SIZE as VOCAB_SIZE, ZERO, PAD
from industrial_automaton import tasks


def cli():
    # Import here to not mess up other CLI commands
    from industrial_automaton.config import settings
    from industrial_automaton.trainer import JAXTrainer

    # pretty print all the settings
    print("="*20 + f" {ANSI.bold('Settings')} " + "="*20)
    for k, v in settings.model_dump().items():
        print(f"{ANSI.blue(k)}: {ANSI.italic(v)}")
    print("="*50)

    # Generate a single example
    print(f"     {ANSI.bold('Task')}: {settings.task}")
    task_kwargs: dict[str, Any] = {"rng": np.random.default_rng(settings.seed)}
    if settings.n is not None:
        task_kwargs["n"] = settings.n

    task_fn = partial(
        getattr(tasks, f"generate_{settings.task}"),
        **task_kwargs
    )
    example = task_fn(batch_size=1, length=10)

    # Print the example
    print(f"{ANSI.bold('Formatted')}: {example['input_formatted'][0]} -> {example['output_formatted'][0]}")
    print(f"{ANSI.bold('      Raw')}: {example['input'][0]} -> {example['output'][0]}")

    # Now we will train the model
    print("\n" + "="*20 + f" {ANSI.bold('Training')} " + "="*20)

    # Prepare model config
    from industrial_automaton.vocab import SIZE as VOCAB_SIZE, PAD, ZERO
    print(f"{ANSI.blue('Vocab size')}: {VOCAB_SIZE}")

    # Initialize model based on settings
    config_cls, model_cls = {
        "baby_ntm": (BabyNTMModelConfig, BabyNTM),
        "suzgun_stack_rnn": (SuzgunStackRNNConfig, SuzgunStackRNN),
        "tape_rnn": (TapeRNNConfig, TapeRNN),
        "transformer": (TransformerConfig, Transformer),
        "lstm": (LSTMConfig, LSTM),
    }.get(settings.model, (None, None))
    if config_cls == None:
        raise ValueError(f"Unknown model: {settings.model}")
    
    # Handle Transformer max_seq_len specifically
    model_kwargs = settings.model_kwargs.copy()
    if settings.model == "transformer" and "max_seq_len" not in model_kwargs:
        model_kwargs["max_seq_len"] = max(settings.tr_max_seqlen, settings.tr_eval_max_seqlen) * 2

    config = config_cls(**model_kwargs)
    model = model_cls(config, key=jax.random.PRNGKey(settings.seed))
    print(f"{ANSI.green('Model')}: {settings.model} {config.model_dump()}")

    # Define loss function
    def loss_fn(model, batch, key):
        inputs_np, labels_np = batch
        inputs = jax.nn.one_hot(jnp.array(inputs_np), VOCAB_SIZE)
        labels = jnp.array(labels_np)

        # Detect if it's classification (scalar label) or sequence (vector label)
        is_sequence = labels.ndim == 2
        
        def single_example(inp, lab):
            state = model.init_state()
            outputs, _ = model(inp, state) # (T, vocab)
            
            if is_sequence:
                # Sequence Cross Entropy with Masking
                log_probs = jax.nn.log_softmax(outputs, axis=-1)
                # Select log_probs for the correct labels
                target_log_probs = jnp.take_along_axis(log_probs, lab[:, None], axis=-1).squeeze(-1)
                
                # Mask out PAD tokens
                mask = (lab != PAD)
                denom = jnp.sum(mask) + 1e-6
                
                loss = -jnp.sum(target_log_probs * mask) / denom
                
                # Accuracy only on non-padded tokens
                correct = (jnp.argmax(outputs, axis=-1) == lab)
                accuracy = jnp.sum(correct * mask) / denom
            else:
                # Scalar Classification (Binary)
                pred_logit = outputs[-1, 1] - outputs[-1, 0] if VOCAB_SIZE > 1 else outputs[-1, 0]
                pred = jax.nn.sigmoid(pred_logit)
                loss = -lab * jnp.log(pred + 1e-7) - (1 - lab) * jnp.log(1 - pred + 1e-7)
                accuracy = (pred > 0.5).astype(jnp.float32) == lab
            
            return loss, accuracy

        losses, accuracies = jax.vmap(single_example)(inputs, labels)
        loss = jnp.mean(losses)
        accuracy = jnp.mean(accuracies)

        return loss, {"accuracy": accuracy}

    # --- Dataset Generation Helper ---
    def generate_variable_dataset(
        base_task_fn, 
        base_kwargs, 
        num_examples, 
        max_seqlen_param, 
        min_seqlen_param=5
    ):
        """Generates a dataset with variable lengths up to max_seqlen_param."""
        chunk_size = 64
        num_chunks = int(np.ceil(num_examples / chunk_size))
        
        all_inputs = []
        all_labels = []
        
        max_in_width = 0
        max_out_width = 0
        
        print(f"Generating {num_examples} examples with lengths {min_seqlen_param}-{max_seqlen_param}...")
        
        for _ in range(num_chunks):
            # Pick a random length for this chunk
            l = np.random.randint(min_seqlen_param, max_seqlen_param + 1)
            
            # Generate
            data = base_task_fn(batch_size=chunk_size, length=l, **base_kwargs)
            inp, out = data["input"], data["output"]
            
            # Determine widths
            in_w = inp.shape[1]
            out_w = out.shape[1]
            
            # Structural Padding (ZERO) if needed for this chunk
            # e.g. duplicate_string: in=L, out=2L. We need in to be 2L with ZEROs.
            needed_in_width = max(in_w, out_w)
            if in_w < needed_in_width:
                 padding = np.full((chunk_size, needed_in_width - in_w), ZERO, dtype=inp.dtype)
                 inp = np.concatenate([inp, padding], axis=1)
                 in_w = needed_in_width
            
            # Update global max widths
            max_in_width = max(max_in_width, in_w)
            max_out_width = max(max_out_width, out_w)
            
            all_inputs.append(inp)
            all_labels.append(out)
            
        # Determine final unified width (usually input and output should match for seq-to-seq models to run)
        final_width = max(max_in_width, max_out_width)
        
        # Collate and Pad to final_width with PAD
        total_inputs = np.full((num_chunks * chunk_size, final_width), PAD, dtype=np.int64)
        total_labels = np.full((num_chunks * chunk_size, final_width), PAD, dtype=np.int64)
        
        cursor = 0
        for inp, out in zip(all_inputs, all_labels):
            n = inp.shape[0]
            w_in = inp.shape[1]
            w_out = out.shape[1]
            
            # Copy input (already structurally padded with ZERO if needed)
            total_inputs[cursor:cursor+n, :w_in] = inp
            
            # Copy output
            total_labels[cursor:cursor+n, :w_out] = out
            
            cursor += n
            
        return total_inputs[:num_examples], total_labels[:num_examples]

    # Generate Training Data
    print(f"{ANSI.blue('Generating training dataset...')}")
    train_dataset_size = 10000
    train_kwargs = task_kwargs.copy()
    if settings.tr_task_kwargs:
        train_kwargs.update(settings.tr_task_kwargs)
        
    train_inputs, train_labels = generate_variable_dataset(
        task_fn, 
        train_kwargs, 
        train_dataset_size, 
        settings.tr_max_seqlen
    )
    print(f"{ANSI.blue('Train Dataset')}: {train_inputs.shape} -> {train_labels.shape}")

    # Generate Evaluation Data
    print(f"{ANSI.blue('Generating evaluation dataset...')}")
    eval_dataset_size = 1000
    eval_kwargs = task_kwargs.copy()
    if settings.tr_eval_task_kwargs:
        eval_kwargs.update(settings.tr_eval_task_kwargs)
        
    eval_inputs, eval_labels = generate_variable_dataset(
        task_fn, 
        eval_kwargs, 
        eval_dataset_size, 
        settings.tr_eval_max_seqlen
    )
    print(f"{ANSI.blue('Eval Dataset')}: {eval_inputs.shape} -> {eval_labels.shape}")

    train_batch_size = 32
    # Create RNG for shuffling (using the same seed for full reproducibility)
    shuffle_rng = np.random.default_rng(settings.seed)

    # Create data iterator that yields batches from the fixed dataset
    def data_iterator():
        """Iterator that yields batches from the fixed dataset, shuffling after each epoch."""
        num_batches = train_dataset_size // train_batch_size
        indices = np.arange(train_dataset_size)

        while True:
            # Shuffle at the start of each epoch using the seeded RNG
            shuffle_rng.shuffle(indices)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * train_batch_size
                end_idx = start_idx + train_batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_inputs = train_inputs[batch_indices]
                batch_labels = train_labels[batch_indices]

                yield (batch_inputs, batch_labels)

    # Eval Function
    def eval_loop(model):
        """Runs evaluation on the fixed eval dataset."""
        # Run in larger batches for speed, but watch memory
        eval_batch_size = 128
        num_batches = int(np.ceil(eval_dataset_size / eval_batch_size))
        
        total_loss = 0.0
        total_acc = 0.0
        
        # JIT the single batch eval
        @jax.jit
        def eval_batch(m, inp, lab):
            return loss_fn(m, (inp, lab), jax.random.PRNGKey(0)) # Key doesn't matter for eval usually
            
        for i in range(num_batches):
            start = i * eval_batch_size
            end = min(start + eval_batch_size, eval_dataset_size)
            if start >= end: break
            
            b_in = eval_inputs[start:end]
            b_lbl = eval_labels[start:end]
            
            l, metrics = eval_batch(model, b_in, b_lbl)
            total_loss += float(l) * (end - start)
            total_acc += float(metrics["accuracy"]) * (end - start)
            
        avg_loss = total_loss / eval_dataset_size
        avg_acc = total_acc / eval_dataset_size
        
        # Return StepMetrics compatible with trainer
        from industrial_automaton.trainer import StepMetrics
        return StepMetrics(loss=jnp.array(avg_loss), accuracy=jnp.array(avg_acc))

    # Initialize trainer
    trainer = JAXTrainer(model, loss_fn, settings)

    # Train
    print(f"{ANSI.bold('Starting training...')}")
    history = trainer.fit(
        data_generator=data_iterator(),
        eval_fn=eval_loop
    )

    print("\n" + "="*50)
    print(f"{ANSI.green('Training complete!')}")
    print(f"Final loss: {float(history[-1].loss):.4f}")
    if history[-1].accuracy is not None:
        print(f"Final accuracy: {float(history[-1].accuracy):.4f}")



def print_model_configs():
    print("="*20 + f" {ANSI.bold('Model Configurations')} " + "="*20)
    print(f"Legend: {ANSI.red('*')} = Required field")
    
    configs = {
        "baby_ntm": BabyNTMModelConfig,
        "suzgun_stack_rnn": SuzgunStackRNNConfig,
        "tape_rnn": TapeRNNConfig,
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