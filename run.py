from pipeline import parse_args, run_experiment

def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
