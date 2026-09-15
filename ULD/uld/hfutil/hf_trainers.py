import torch
import datasets
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer
from transformers.trainer_utils import has_length
from typing import Callable, Optional

from ..data.datamodule import EqualForgetRetainSampler


class ForgetTrainer(Trainer):

    def __init__(self, model, train_loss_function: Callable, oracle_model=None, equal_sampler=False, seed=42, **kwargs):
        super(ForgetTrainer, self).__init__(model=model, **kwargs)
        self.train_loss_function = train_loss_function
        self.equal_sampler = equal_sampler
        self.oracle_model = oracle_model
        self.seed = seed
        if self.oracle_model is not None:
            self.oracle_model.requires_grad_(False)
            self._move_model_to_device(self.oracle_model, self.args.device)

    def _get_train_sampler(self, generator=None) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.equal_sampler:
            print("Using EqualForgetRetainSampler")
            return EqualForgetRetainSampler(self.train_dataset.forget_length, self.train_dataset.retain_length, generator=generator)
        else:
            # Build the sampler.
            return RandomSampler(self.train_dataset, generator=generator)

    def get_train_dataloader(self) -> DataLoader:
        """
        Override the original get_train_dataloader function simply for debugging.
        This is identical to the get_train_dataloader function in transformer.Trainer.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        print(f'Generator........Epoch-{self.state.global_step}')

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):

            # dataloader_params["generator"] = generator
            # dataloader_params["shuffle"] = True # set shuffle=True with specified generator.
            dataloader_params["sampler"] = self._get_train_sampler(generator=generator)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # transformers >= 4.46 passes num_items_in_batch; we ignore it.
        losses = self.train_loss_function(model, inputs, self.oracle_model)
        # print("loss", losses)
        loss = losses['loss']
        forgetloss = losses['forget_loss']
        retainloss = losses['retain_loss']

        #! Notice that these are evaluated on mini-batch instead of total effective batch
        logitems = {
            'trainloss/loss': loss.item(),
            'trainloss/forgetloss': forgetloss.item(),
            'trainloss/retainloss': retainloss.item()
        }
        self.log(logitems)

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        import inspect
        signature = inspect.signature(model.forward)
        _signature_columns = list(signature.parameters.keys())
        _signature_columns += list(set(["label", "label_ids"]))
        inputs = {k:v for k, v in inputs.items() if k in _signature_columns}
        labels = inputs['labels']

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            loss = outputs.loss

        if prediction_loss_only:
            return (loss, None, None)
        else:
            if len(logits) == 1:
                logits = logits[0]
            if len(labels) == 1:
                labels = labels[0]
            return (loss, logits, labels)
