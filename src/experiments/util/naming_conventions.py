
def get_model_name_from_exp_name(exp_name: str) -> str:
    return exp_name[exp_name.index('model__')+7:exp_name.index('__unkn')]