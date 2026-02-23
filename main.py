def effectiveness():
    from experiments.effectiveness import run_eval_ov_credit, run_eval_ov_baseline

    run_eval_ov_credit()
    run_eval_ov_baseline()


def efficiency():
    from experiments.efficiency import run_verification_time_baseline, run_defense_time_baseline

    run_verification_time_baseline()
    run_defense_time_baseline()


def parameters():
    from experiments.parameters import param_beta, param_tau, param_sigma

    param_beta()
    param_tau()
    param_sigma()


def plots():
    from experiments.parameters import plot_sigma_gamma_range, plot_sigma_optimal, plot_sigma_band

    plot_sigma_gamma_range()
    plot_sigma_optimal()
    plot_sigma_band()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        func_name = sys.argv[1]
        if func_name in globals() and callable(globals()[func_name]):
            print(f"[INFO] Running function: {func_name}()")
            globals()[func_name]()
        else:
            print(f"[ERROR] Function '{func_name}' not found.")
            print("Available functions:",
                  ", ".join(fn for fn, val in globals().items() if callable(val) and not fn.startswith("__")))
    else:
        print("[USAGE] python main.py <function_name>")
        print("Available functions:",
              ", ".join(fn for fn, val in globals().items() if callable(val) and not fn.startswith("__")))
