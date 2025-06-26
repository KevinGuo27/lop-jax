from ppo import make_train
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import os
import argparse

def main(args):
    config = {
        "LR": 1e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 2048,  
        "TOTAL_TIMESTEPS": 2e4,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 16, # Results in a minibatch size of (2048*20)/32 = 1280
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0, # Can be tuned later if needed
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "ReLU",
        "ENV_NAME": "ant",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": False,
        "ENV_FRICTION": None,
        "WEIGHT_DECAY": 0.0,
    }

    config["TOTAL_TIMESTEPS"] = args.timesteps
    config["LR"] = args.lr
    config["NUM_ENVS"] = args.num_envs
    config["NUM_STEPS"] = args.num_steps
    config["NUM_MINIBATCHES"] = args.num_minibatches
    config["UPDATE_EPOCHS"] = args.num_updates
    config["GAMMA"] = args.gamma
    config["GAE_LAMBDA"] = args.gae_lambda
    config["CLIP_EPS"] = args.clip_eps
    config["ENT_COEF"] = args.entropy_coef
    config["VF_COEF"] = args.vf_coef
    config["MAX_GRAD_NORM"] = args.max_grad_norm
    config["ANNEAL_LR"] = args.anneal_lr
    config["NORMALIZE_ENV"] = args.normalize_env
    config["DEBUG"] = args.debug
    config["ACTIVATION"] = args.activation
    config["ENV_NAME"] = args.env_name
    config["WEIGHT_DECAY"] = args.weight_decay

    with open("sant/frictions", 'rb+') as f:
        frictions = pickle.load(f)
    network_params = None

    # Support multiple seeds for parallel training runs
    rng = jax.random.PRNGKey(args.seed)
    rngs = jax.random.split(rng, args.num_seeds)

    for i in range(args.task_number):
        friction = frictions[args.seed][i]
        print(f"Task {i}/{args.task_number}, Friction: {friction}")

        config["ENV_FRICTION"] = friction

        train_vjit = jax.jit(jax.vmap(make_train(config), in_axes=(0, 0)))
        outs = train_vjit(rngs, network_params)

        metrics = outs["metrics"]
        train_state = outs["runner_state"][0]
        network_params = train_state.params

        # Save serializable parts to file
        serializable_data = {
            "metrics": metrics,
            "params": network_params,
            "friction": friction,
            "task_number": i
        }
        with open(f"sant/data/friction_{i}_outs.pkl", "wb") as f:
            pickle.dump(serializable_data, f)

        plt.plot(metrics["returned_episode_returns"].mean(-1).reshape(-1))
        plt.xlabel("Updates")
        plt.ylabel("Return")
        # plt.show()
        plt.savefig(f"sant/data/friction_{i}_return_vs_updates.png")
        plt.close()



if __name__ == "__main__":
    os.makedirs("sant/data", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_number", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=2e6)
    parser.add_argument("--lr", type=float, default=1e-4)   
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--num_minibatches", type=int, default=16)
    parser.add_argument("--num_updates", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--anneal_lr", type=bool, default=False)
    parser.add_argument("--normalize_env", type=bool, default=True)
    parser.add_argument("--debug", type=bool, default=False)    
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--env_name", type=str, default="ant")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
    