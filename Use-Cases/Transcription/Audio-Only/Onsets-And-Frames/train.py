# [Source] https://github.com/jongwook/onsets-and-frames/blob/master/train.py

from evaluate import evaluate
from onsets_and_frames import *

@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO': # train: 962 files, validation: 137 files
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    elif train_on == 'PIANOVAM_AUDIO':
        dataset = PIANOVAM_AUDIO(path=data_path, groups=['train'], sequence_length=sequence_length)
        validation_dataset = PIANOVAM_AUDIO(path=data_path, groups=['valid'], sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    best_model_path = None
    best_optimizer_path = None
    
    best_validation_loss = float("inf")  # Initialize best validation loss

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    optimizer.zero_grad()

    for i, batch in zip(loop, cycle(loader)):
        
        predictions, losses = model.run_on_batch(batch)
        train_loss = sum(losses.values()) / accumulation_steps # Scale loss

        train_loss.backward()  # Accumulate gradients

        if (i + 1) % accumulation_steps == 0:  # Perform update every 4 steps
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
            if clip_gradient_norm:
                clip_grad_norm_(model.parameters(), clip_gradient_norm)
            
            # **WandB Logging after every accumulation step (1 effective step)**
            wandb.log({"train_loss": train_loss.item() * accumulation_steps}, step= (i + 1) // accumulation_steps)

            for key, value in {'loss': train_loss * accumulation_steps, **losses}.items():
                writer.add_scalar(key, value.item(), global_step=i // accumulation_steps)

        if (i + 1) % (validation_interval) == 0:
            model.eval()
            with torch.no_grad():
                validation_results = evaluate(validation_dataset, model)
                validation_loss = validation_results.get('metric/eval-loss', None)

                if validation_loss is not None:
                    if isinstance(validation_loss, torch.Tensor):
                        validation_loss = validation_loss.detach().cpu().numpy()
                    elif isinstance(validation_loss, list):
                        validation_loss = np.mean(
                            [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in validation_loss])

                print(f"validation_loss: {validation_loss}")
                wandb_log_data = {"validation_loss": validation_loss}

                for key, value in validation_results.items():
                    metric_name = 'validation/' + key.replace(' ', '_')

                    avg_value = np.mean(
                        torch.tensor(value).detach().cpu().numpy() if isinstance(value, list) else value.detach().cpu().numpy()
                    )

                    writer.add_scalar(metric_name, avg_value, global_step=i // accumulation_steps)
                    wandb_log_data[metric_name] = avg_value

                wandb.log(wandb_log_data, step= (i+1) // accumulation_steps)

                # Save the best model checkpoint if validation loss improves
                if validation_loss < best_validation_loss:
                    if best_model_path is not None and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                        print(f"🗑 Deleted previous best model {best_model_path}")
                    if best_optimizer_path is not None and os.path.exists(best_optimizer_path):
                        os.remove(best_optimizer_path)
                        print(f"🗑 Deleted previous best optimizer {best_optimizer_path}")

                    best_validation_loss = validation_loss
                    best_model_path = os.path.join(logdir, f'{train_on}_{subtasks}',
                                                f'best_model-{i+1}step-valLoss{validation_loss:.6f}.pt')
                    best_optimizer_path = os.path.join(logdir, f'{train_on}_{subtasks}',
                            f'best_model-{i+1}step-optimizer-state.pt')
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

                    torch.save(model, best_model_path)
                    torch.save(optimizer.state_dict(), os.path.join(logdir, f'{train_on}_{subtasks}', f"best_model-{i+1}step-optimizer-state.pt"))
                    print(f"✅ New best validation loss: {best_validation_loss:.6f} | Saved model at {best_model_path}")
                    
            model.train()

        if (i + 1) % (checkpoint_interval) == 0:
            checkpoint_dir = os.path.join(logdir, f'{train_on}_{subtasks}')
            os.makedirs(checkpoint_dir, exist_ok=True)  # 🔹 Create directory if not exists

            torch.save(model, os.path.join(checkpoint_dir, f'model-{i+1}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'last-optimizer-state.pt'))

    wandb.finish()
