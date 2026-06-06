class Args:
    pass
args = Args()
args.total_env_steps = 10_000_000
args.num_epochs = 100
args.num_envs = 512
args.training_steps_multiplier = 1

print("training steps per env:", args.total_env_steps / args.num_envs)
# calculate epochs?
