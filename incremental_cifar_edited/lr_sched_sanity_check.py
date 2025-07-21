import optax 

def make_lr_scheduler(num_tasks: int,
                        base_lr: float,
                        base_steps_per_epoch: int,
                        epochs_per_task: int,
                        drop_factor: float = 0.2,
                        drop_epochs: tuple = (60, 120, 160)):

        # at what steps do we have a new task? 
        task_starts = [0]
        for t in range(1, num_tasks):
            S = base_steps_per_epoch * t
            task_starts.append(task_starts[-1] + S * epochs_per_task)
        
        def lr_fn(global_step : int) -> float: 
            # which task are we on?
            t = next(i for i, start in enumerate(task_starts)
                 if i == len(task_starts)-1 or global_step < task_starts[i+1])
            
            # which epoch are we on?
            # local step within the task
            S = base_steps_per_epoch * (t+1)
            local_step = global_step - task_starts[t]
            local_epoch = local_step // S

            # piecewise constant lr schedule within the task
            if local_epoch < drop_epochs[0]:
                return base_lr 
            elif local_epoch < drop_epochs[1]:
                return base_lr * drop_factor
            elif local_epoch < drop_epochs[2]:
                return base_lr * (drop_factor**2)
            else:
                return base_lr * (drop_factor**3)

        return lr_fn

lr_schedule = make_lr_scheduler(
    num_tasks = 20, #20 for incremental cifar-100
    base_lr = 0.1, #0.1 for incremental cifar-100
    base_steps_per_epoch = 25,
    epochs_per_task = 200,
    drop_factor = 0.2,
    drop_epochs = (60, 120, 160),
)

for step in range(0, 100000, 25):
    print(f"Epoch {step//25:4d} (global_step={step:5d}): lr = {lr_schedule(step):.5f}")