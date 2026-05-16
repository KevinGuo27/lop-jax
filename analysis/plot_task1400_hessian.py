import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_hessian_data(agent_name, task_num, seed='2025', at_init=True):
    """Load hessian data from the .npy file"""
    data_dir = Path("/users/kguo32/data/kguo32/lop/imagenet/hessian/data") / agent_name / seed
    phase = 'init' if at_init else 'end'
    fname = data_dir / f"hessian_task_{task_num}_{phase}.npy"
    
    if not fname.exists():
        raise FileNotFoundError(f"Hessian data not found: {fname}")
    
    data = np.load(fname, allow_pickle=True).item()
    return data['grids_train'], data['density_train'], data['grids_test'], data['density_test']

def plot_hessian_spectrum_large_font(grids_train, density_train, grids_test, density_test, 
                                   task_num, agent_name, at_init=True, seed='2025'):
    """Plot hessian spectrum with larger font sizes in PNG format - train only"""
    grids_np_train = np.array(grids_train)
    density_np_train = np.array(density_train)

    # Set larger font sizes
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 18

    plt.figure(figsize=(10, 8))
    plt.semilogy(grids_np_train, density_np_train, label=f'Task {task_num} train', color='blue', linewidth=2)
    plt.ylim(1e-10, 1e2)
    plt.xlim(-100, 1000)
    plt.ylabel("Density", fontsize=28)
    plt.xlabel("Eigenvalue", fontsize=28)
    # plt.title(f'Hessian Spectrum - {agent_name.upper()} (Task {task_num})', fontsize=22)
    plt.legend(fontsize=24)
    plt.grid(True, alpha=0.3)
    
    # Save as PNG
    out_dir = Path("/users/kguo32/rl-opt/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    phase = 'init' if at_init else 'end'
    fname = out_dir / f"hessian_spectrum_{agent_name}_task{task_num}_{phase}.pdf"
    
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved Hessian spectrum to {fname}")
    plt.close()

def main():
    task_num = 100
    seed = '2025'
    at_init = True
    
    agents = ['bp', 'l2_er']
    
    for agent in agents:
        try:
            print(f"Loading hessian data for {agent}...")
            grids_train, density_train, grids_test, density_test = load_hessian_data(
                agent, task_num, seed, at_init
            )
            
            print(f"Plotting hessian spectrum for {agent}...")
            plot_hessian_spectrum_large_font(
                grids_train, density_train, grids_test, density_test,
                task_num, agent, at_init, seed
            )
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error for {agent}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()
