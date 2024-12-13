CURRENT_RUN_ID: int = None
CURRENT_EXP_NAME: str = None


def set_current_run_id(run_id: int):
    global CURRENT_RUN_ID
    CURRENT_RUN_ID = run_id


def set_current_exp_name(exp_name: str):
    global CURRENT_EXP_NAME
    CURRENT_EXP_NAME = exp_name


def get_current_run_id() -> int:
    return CURRENT_RUN_ID


def get_current_exp_name() -> str:
    return CURRENT_EXP_NAME
