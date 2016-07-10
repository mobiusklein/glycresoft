import csv


def convert_solution_to_dict(solution):
    return {
        "key": solution.key,
        "neutral_mass": str(solution.neutral_mass),
        "score": str(solution.score),
        "total_signal": str(solution.total_signal),
        "start_time": str(solution.start_time),
        "end_time": str(solution.end_time),
        "charge_states": ';'.join(map(str, solution.charge_states))
    }


def write_solutions_to_csv(solutions, handle):
    writer = csv.DictWriter(
        handle, ["key", "neutral_mass", "score", "total_signal",
                 "start_time", "end_time", "charge_states"])
    writer.writeheader()
    writer.writerows(map(convert_solution_to_dict, solutions))


def to_csv(solutions, path):
    with open(path, 'wb') as handle:
        write_solutions_to_csv(solutions, handle)
