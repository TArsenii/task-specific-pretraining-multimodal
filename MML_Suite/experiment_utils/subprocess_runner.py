from experiment_utils.logging import get_logger

logger = get_logger()


def run_process(process: str, args: str | list[str]):
    """
    Run a process with the given arguments.

    Args:
        process (str): The process to run.
        args (str | list[str]): The arguments to pass to the process.

    Returns:
        int: The return code of the process.
    """
    logger.info(f"Attempting to run process: {process} {args}")
    # if isinstance(args, str):
    #     args = shlex.split(args)

    # process = subprocess.Popen([process] + args)
    # process.communicate()

    return process.returncode
