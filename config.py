import json


class Config:

    def __init__(self,
                 dataset_path="",
                 dataset_cache="",
                 model_checkpoint="",
                 num_candidates=2,
                 do_lower_case=True,
                 max_history=2,
                 train_batch_size=4,
                 valid_batch_size=4,
                 gradient_accumulation_steps=8,
                 lr=5e-5,
                 warmup_proportion=0.1,
                 lm_coef=1,
                 mc_coef=1,
                 max_norm=10,
                 n_epochs=2,
                 personality_permutations=1,
                 eval_before_start=False,
                 device="cpu",
                 fp16="",
                 local_rank=-1,
                 log_dir="",
                 ):
        self.dataset_path = dataset_path
        self.dataset_cache = dataset_cache
        self.model_checkpoint = model_checkpoint
        self.num_candidates = num_candidates
        self.do_lower_case = do_lower_case
        self.max_history = max_history
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.warmup_proportion = warmup_proportion
        self.lm_coef = lm_coef
        self.mc_coef = mc_coef
        self.max_norm = max_norm
        self.n_epochs = n_epochs
        self.personality_permutations = personality_permutations
        self.eval_before_start = eval_before_start
        self.device = device
        self.fp16 = fp16
        self.local_rank = local_rank
        self.log_dir = log_dir

    @classmethod
    def from_dict(cls, json_object):
        config = Config()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))


class InteractConfig:

    def __init__(self,
                 dataset_path="",
                 model="",
                 dataset_cache="",
                 model_checkpoint="",
                 max_history="",
                 device="",
                 no_sample="",
                 max_length="",
                 min_length="",
                 seed="",
                 temperature="",
                 top_k="",
                 top_p=""
                 ):
        self.dataset_path = dataset_path
        self.model = model
        self.dataset_cache = dataset_cache
        self.model_checkpoint = model_checkpoint
        self.max_history = max_history
        self.device = device
        self.no_sample = no_sample
        self.max_length = max_length
        self.min_length = min_length
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    @classmethod
    def from_dict(cls, json_object):
        config = InteractConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))
