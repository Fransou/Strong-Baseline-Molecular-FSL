from fs_mol.utils.simsearch_eval import SimSearchEval


class SimSearchEvaluator:
    def __init__(self, **kwargs):
        pass

    def __call__(self, task_sample, y_support, y_query, **kwargs):
        task_trainer = SimSearchEval(
            name=task_sample.name,
            inference_task_sample=task_sample,
            n_neighb=3,
        )
        y_pred_query = task_trainer()
        return y_pred_query
