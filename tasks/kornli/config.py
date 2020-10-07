from typing import NamedTuple


class TrainConfig(NamedTuple):
    desc: str = ""

    """
    Model Hyperparameters
    """
    #: vocab, tokenizer, bert config and model 파일이 저장된 경로. join(resource_dir, tokenizer)로 합쳐서 사용
    resource_dir: str = "./resources"
    #: tokenizer_name
    tokenizer: str = ""
    #: bert 설정 파일 이름
    bert_config_file_name: str = "bert_config.json"
    #: pretrained bert 모델 파일 이름
    pretrained_bert_file_name: str = "bert_model.pth"

    """
    Training Hyperparameters
    """
    #: random seed
    seed: int = 42
    #: epoch 도는 횟수
    num_epochs: int = 3
    #: 훈련 시의 batch size
    batch_size: int = 64
    #: learning rate
    learning_rate: float = 1e-4
    #: bert fine tuning 레이어의 dropout 확률
    dropout_prob: float = 0.1
    #: warmup step의 비율 (warmup step = total step * warmup step ratio)
    warmup_step_ratio: float = 0.1
    #: max sequence length
    max_sequence_length: int = 128
    #: logging을 진행할 단위 step
    logging_interval: int = 100

    """
    Data Hyperparameters
    """
    #: training data 파일 경로
    train_path: str = "./dataset/nlu_tasks/kornli/sample_multinli.train.ko.tsv"
    #: dev data 파일 경로
    dev_path: str = "./dataset/nlu_tasks/kornli/sample_xnli.dev.ko.tsv"
    #: test data 파일 경로
    test_path: str = "./dataset/nlu_tasks/kornli/sample_xnli.test.ko.tsv"
    #: output dir
    log_dir: str = "./run_outputs/{}/kornli/logs"
    summary_dir: str = "./run_outputs/{}/kornli/summaries"
    # checkpoint_dir: str = "./run_outputs/{}/kornli/checkpoints"

    def __repr__(self):
        _repr_str = "Training Configuration:\n{\n"
        for k, v in self._asdict().items():
            _repr_str += f"\t{k}: {v}\n"
        _repr_str += "}\n"
        return _repr_str
