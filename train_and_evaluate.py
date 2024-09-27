
from rlopt.config import PolicyHyperparams

from train_pi_params import run_train
from evaluate_policies import run_and_save_pe


if __name__ == "__main__":
    args = PolicyHyperparams().parse_args()

    train_path = run_train(args)

    print(f"Train results saved to {train_path}")

    eval_path = run_and_save_pe({
        'checkpoint_path': train_path
    })

    print(f"Eval results saved to {eval_path}")
