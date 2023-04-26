
import os
from affordance.config import get_cfg
from affordance.engine import default_argument_parser, default_setup, launch, Trainer, comm
from affordance.evaluation import verify_results, DatasetEvaluators, VTSampler
from affordance.evaluation.bits_evaluation import BitsEvaluator
from affordance.evaluation.codes_extractor import CodesExtractor
from affordance.evaluation.mse_evaluation import MSEEvaluator


class MyTrainer(Trainer):


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        if "CodesExtractor" in cfg.TEST.EVALUATORS:
            evaluator_list.append(CodesExtractor(dataset_name, distributed=True, output_dir=output_folder))
        if "MSEEvaluator" in cfg.TEST.EVALUATORS:
            evaluator_list.append(MSEEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if "BitsEvaluator" in cfg.TEST.EVALUATORS:
            evaluator_list.append(BitsEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if "VTSampler" in cfg.TEST.EVALUATORS:
            evaluator_list.append(VTSampler(cfg, dataset_name, distributed=True, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        _, checkpointers = model.configure_optimizers_and_checkpointers()
        for item in checkpointers:
            item["checkpointer"].resume_or_load(item["pretrained"], resume=False)
        res = MyTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
