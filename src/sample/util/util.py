import subprocess


def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#':
                continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process


def group_files(n, process):
    """
    groups pairs into sublists of size n
    :param n: (int) sublist size
    :param process: (list) list of pairs to process
    :return: grouped_files (list) list of sublists of pairs
    """
    grouped_files = []

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    return grouped_files


def run_cmd(cmd, error_msg=None, raise_except=False):
    try:
        return subprocess.check_output(
            cmd,
            universal_newlines=True,
            shell=True)
    except Exception as e:
        if error_msg is not None:
            print(error_msg)
        if raise_except:
            raise e


def group_grid(n, grid_size, step):
    grid = []
    vals = set()
    for i in range(0, grid_size + 1, step):
        vals.add(i)
        vals.add(-i)
    for dx in vals:
        for dy in vals:
            for dz in vals:
                grid.append((dx, dy, dz))

    print(grid_size, len(grid))

    grouped_files = []

    for i in range(0, len(grid), n):
        grouped_files += [grid[i: i + n]]

    return grouped_files


