import os
import math
import torch
import random
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .dataset import MultiTaskDataset
from ..vocalset_singer_id.dataset import SingerDataset
from ..fluent_commands.dataset import FluentCommandsDataset

class DownstreamExpert(nn.Module):
    """
    Multi-task model that handles both singer identification and fluent commands
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        # Initialize datasets
        self.get_datasets(kwargs)

        # Initialize model components
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        
        # Shared projector
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        
        # Model with two output heads
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim_1=len(self.singer_dataset.class2id.keys()),
            output_dim_2=sum(self.values_per_slot),
            **model_conf,
        )
        
        # Loss functions
        self.singer_objective = nn.CrossEntropyLoss()
        self.fluent_objective = nn.CrossEntropyLoss()
        
        self.register_buffer('best_singer_score', torch.zeros(1))
        self.register_buffer('best_fluent_score', torch.zeros(1))

    def get_datasets(self, kwargs):
        # Initialize singer dataset
        singer_root_dir = Path(self.datarc['singer_file_path'])
        self.singer_dataset = SingerDataset(
            singer_root_dir,
            self.datarc['singer_meta_data'],
            'train',
            upstream=kwargs['upstream'],
            features_path=kwargs['features_path'],
        )

        # Initialize fluent commands dataset
        self.base_path = self.datarc['fluent_file_path']
        train_df = pd.read_csv(os.path.join(self.base_path, "data", "train_data.csv"))
        valid_df = pd.read_csv(os.path.join(self.base_path, "data", "valid_data.csv"))
        test_df = pd.read_csv(os.path.join(self.base_path, "data", "test_data.csv"))

        Sy_intent = {"action": {}, "object": {}, "location": {}}
        values_per_slot = []
        for slot in ["action", "object", "location"]:
            slot_values = Counter(train_df[slot])
            for index, value in enumerate(slot_values):
                Sy_intent[slot][value] = index
                Sy_intent[slot][index] = value
            values_per_slot.append(len(slot_values))
        self.values_per_slot = values_per_slot
        self.Sy_intent = Sy_intent

        # Create combined datasets
        self.train_dataset = MultiTaskDataset(
            singer_root_dir,
            self.datarc['singer_meta_data'],
            train_df,
            self.base_path,
            self.Sy_intent,
            'train',
            upstream=kwargs['upstream'],
            features_path=kwargs['features_path'],
        )
        self.dev_dataset = MultiTaskDataset(
            singer_root_dir,
            self.datarc['singer_meta_data'],
            valid_df,
            self.base_path,
            self.Sy_intent,
            'dev',
            upstream=kwargs['upstream'],
            features_path=kwargs['features_path'],
        )
        self.test_dataset = MultiTaskDataset(
            singer_root_dir,
            self.datarc['singer_meta_data'],
            test_df,
            self.base_path,
            self.Sy_intent,
            'test',
            upstream=kwargs['upstream'],
            features_path=kwargs['features_path'],
        )

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)

        # Get predictions from both tasks
        singer_pred, fluent_pred, _ = self.model(features, features_len)

        # Handle singer identification task
        singer_labels = torch.LongTensor(labels['singer']).to(device)
        singer_loss = self.singer_objective(singer_pred, singer_labels)
        singer_predicted = singer_pred.max(dim=-1).indices
        records['singer_acc'] += (singer_predicted == singer_labels).view(-1).cpu().float().tolist()
        records['singer_loss'].append(singer_loss.item())

        # Handle fluent commands task
        fluent_labels = torch.stack([torch.LongTensor(label) for label in labels['fluent']]).to(device)
        fluent_loss = 0
        start_index = 0
        predicted_intent = []
        
        for slot in range(len(self.values_per_slot)):
            end_index = start_index + self.values_per_slot[slot]
            subset = fluent_pred[:, start_index:end_index]
            fluent_loss += self.fluent_objective(subset, fluent_labels[:, slot])
            predicted_intent.append(subset.max(1)[1])
            start_index = end_index

        predicted_intent = torch.stack(predicted_intent, dim=1)
        records['fluent_acc'] += (predicted_intent == fluent_labels).prod(1).view(-1).cpu().float().tolist()
        records['fluent_loss'].append(fluent_loss.item())

        # Store predictions and truths for logging
        records['filename'] += filenames
        records['predict_singer'] += self.singer_dataset.label2singer(singer_predicted.cpu().tolist())
        records['truth_singer'] += self.singer_dataset.label2singer(singer_labels.cpu().tolist())
        
        def idx2slots(indices: torch.Tensor):
            action_idx, object_idx, location_idx = indices.cpu().tolist()
            return (
                self.Sy_intent["action"][action_idx],
                self.Sy_intent["object"][object_idx],
                self.Sy_intent["location"][location_idx],
            )
        
        records["predict_fluent"] += list(map(idx2slots, predicted_intent))
        records["truth_fluent"] += list(map(idx2slots, fluent_labels))
        # print(singer_loss.item(), fluent_loss.item())
        # Combined loss
        total_loss = singer_loss + fluent_loss
        #total_loss = singer_loss
        return total_loss

    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        
        # Log singer identification metrics
        for key in ["singer_acc", "singer_loss"]:
            #average = torch.FloatTensor(records[key]).mean().item()
            average = torch.nanmean(torch.FloatTensor(records[key])).item()
            
            logger.add_scalar(
                f'multi_task/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'singer_acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} singer at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_singer_score:
                        self.best_singer_score = torch.ones(1) * average
                        f.write(f'New best singer on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-singer-best.ckpt')

        # Log fluent commands metrics
        for key in ["fluent_acc", "fluent_loss"]:
            #average = torch.FloatTensor(records[key]).mean().item()
            average = torch.nanmean(torch.FloatTensor(records[key])).item()
            logger.add_scalar(
                f'multi_task/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'fluent_acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} fluent at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_fluent_score:
                        self.best_fluent_score = torch.ones(1) * average
                        f.write(f'New best fluent on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-fluent-best.ckpt')

        # Save predictions
        if mode in ["dev", "test"]:
            # Save singer predictions
            with open(Path(self.expdir) / f"{mode}_singer_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_singer"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_singer_truth.txt", "w") as file:
                lines = [f"{f} {l}\n" for f, l in zip(records["filename"], records["truth_singer"])]
                file.writelines(lines)

            # Save fluent commands predictions
            with open(Path(self.expdir) / f"{mode}_fluent_predict.csv", "w") as file:
                lines = [f"{f},{a},{o},{l}\n" for f, (a, o, l) in zip(records["filename"], records["predict_fluent"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_fluent_truth.csv", "w") as file:
                lines = [f"{f},{a},{o},{l}\n" for f, (a, o, l) in zip(records["filename"], records["truth_fluent"])]
                file.writelines(lines)

        return save_names 
