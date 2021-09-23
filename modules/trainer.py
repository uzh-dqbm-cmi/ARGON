import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
import random
from nltk.translate.bleu_score import sentence_bleu
import json
from tqdm import tqdm
import torch.distributed as dist
import numpy as np
from modules.dataloaders import R2DataLoadert2tBart, R2DataLoader


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args , model_image2text=None):
        self.args = args
        self.args.record_dir = args.save_dir

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
            if model_image2text is not None:
                self.model_image2text = model_image2text.to(self.device)
                self.model_image2text = torch.nn.DataParallel(model_image2text, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            if not self.args.resume:
                self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if not self.args.resume and  self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):

        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        checkpoint = torch.load(resume_path,map_location=map_location)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _output_generation(self, predictions, gts, idxs, epoch, subset):
        # for evaluating and saving
        from nltk.translate.bleu_score import sentence_bleu
        import json
        # for saving json file
        output = list()

        for idx, pre, gt in zip(idxs, predictions, gts):
            score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt, 'bleu4': score})

        output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        output_filename = os.path.join(self.checkpoint_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_generated.json')
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def _train_epoch(self, epoch):

        train_loss = 0
        self.model.train()
        log={}
        if not self.args.resume:
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device)
                output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
            log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        if not self.args.resume:
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.val_dataloader)):
                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                        self.device), reports_masks.to(self.device)
                    output = self.model(images, mode='sample')
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            ids=[]
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                #ids.extend(images_id)
                ids.extend([int(image_id) for image_id in images_id.cpu().numpy()])
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        self._output_generation(test_res, test_gts, ids, epoch, 'test')

        self.lr_scheduler.step()

        return log

class TrainerProgressive(BaseTrainer):
    def __init__(self, model, model_image2text, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(TrainerProgressive, self).__init__(model, criterion, metric_ftns, optimizer, args)

        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.model_image2text =  self._load_checkpoint(model_image2text,os.path.join(self.args.best_checkpoint_0, 'model_best.pth'))
        self.model = self._load_checkpoint(model, os.path.join(self.args.best_checkpoint_1, 'model_best_2.pth'))

    def _load_checkpoint(self, model, checkpoint_path):
        checkpoint_path = str(checkpoint_path)

        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # setup GPU device if available, move model into configured device
       # self.start_epoch = checkpoint['epoch'] + 1
        self.device, device_ids = self._prepare_device(self.args.n_gpu)
        model = model.to(self.device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(checkpoint['state_dict'])
        #print(f"Checkpoint loaded form epoch {self.start_epoch}")
        return model

    def _output_generation(self, predictions, gts, idxs, epoch, subset):
        # for saving json file
        output = list()
        for idx, pre, gt in zip(idxs, predictions, gts):
            score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt, 'bleu4': score})

        output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        output_filename = os.path.join(self.checkpoint_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_generated.json')
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False,indent=4)

    def _train_epoch(self, epoch):
        log={}
        self.model.eval()
        self.model_image2text.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, _, _, labels) in enumerate(self.val_dataloader):
                labels = labels.to(self.device)
                images = images.to(self.device)
                # #step=0
                output_0 = self.model_image2text(images, mode='sample')

                ##### required for the next step
                output_from_image2text = self.model_image2text.tokenizer.decode_batch(output_0.cpu().numpy())
                input_encodings = self.model.tokenizer.batch_encode_plus(output_from_image2text, return_tensors="pt",
                                                                         pad_to_max_length=True,
                                                                         max_length=self.args.max_seq_length,
                                                                         truncation=True)
                input_ids = input_encodings['input_ids']
                attention_mask = input_encodings['attention_mask']


                outputs = self.model(input_ids.to(self.device), attention_mask.to(self.device), mode='sample')
                # tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                reports = []
                for output in outputs.cpu().numpy():
                    reports.append(self.model.tokenizer.decode(output, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False))

                ground_truths = []
                labels[labels == -100] = self.model.tokenizer.pad_token_id
                for label in labels.cpu().numpy():
                    ground_truths.append(self.model.tokenizer.decode(label, skip_special_tokens=True,
                                                                     clean_up_tokenization_spaces=False))

                val_res.extend(reports)
                val_gts.extend(ground_truths)
            #
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        self.model.eval()
        self.model_image2text.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            ids = []
            reports_mirqi = []
            for batch_idx, (images_id, images, _, _, labels) in enumerate(self.test_dataloader):
                labels = labels.to(self.device)
                images = images.to(self.device)
                # #step=0
                output_0 = self.model_image2text(images, mode='sample')

                ##### required for the next step
                output_from_image2text = self.model_image2text.tokenizer.decode_batch(output_0.cpu().numpy())


                input_encodings = self.model.tokenizer.batch_encode_plus(output_from_image2text, return_tensors="pt", pad_to_max_length=True, max_length=self.args.max_seq_length, truncation=True)
                input_ids=input_encodings['input_ids']
                attention_mask=input_encodings['attention_mask']

                outputs = self.model(input_ids.to(self.device), attention_mask.to(self.device), mode='sample')
                # tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                reports = []
                for output in outputs.cpu().numpy():
                    reports.append(self.model.tokenizer.decode(output, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False))

                ground_truths = []
                labels[labels == -100] = self.model.tokenizer.pad_token_id
                for label in labels.cpu().numpy():
                    ground_truths.append(self.model.tokenizer.decode(label, skip_special_tokens=True,
                                                                     clean_up_tokenization_spaces=False))

                test_res.extend(reports)
                test_gts.extend(ground_truths)
                ids.extend(images_id)
            #
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            #
            self._output_generation(test_res, test_gts, ids, epoch, 'test')

        self.lr_scheduler.step()

        return log


class BaseTrainer_(object):
    def __init__(self, model, model_image2text, criterion, metric_ftns, optimizer, optimizer_2, args):
        self.args = args
        self.model=model
        self.model_image2text = model_image2text

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.optimizer = optimizer
        self.optimizer_2 = optimizer_2

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir


        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        if args.resume is not None:
            self._resume_checkpoint(args.resume, args.resume_2)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        self.val_outputs = []
        self.val_labels = []
        self.test_outputs = []
        self.test_labels = []
    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def gather(self, tensor, tensor_list=None, root=0, group=None):
        """
            Sends tensor to root process, which store it in tensor_list.
        """
        rank = dist.get_rank()
        if group is None:
            group = dist.group.WORLD
        if rank == root:
            assert (tensor_list is not None)
            dist.gather(tensor, gather_list=tensor_list, group=group)
        else:
            dist.gather(tensor, dst=root, group=group)

    def train(self):
        not_improved_count = 0

        self.gpu = self.args.local_rank
        print(f"gpu: {self.args.local_rank}")
        torch.cuda.set_device(self.args.local_rank)
        self.device = torch.device('cuda', self.args.local_rank)
        torch.distributed.init_process_group(backend='nccl', rank= self.args.local_rank)

        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)

        self.model_image2text.to(self.device)
        self.model.to(self.device)

        self.model_image2text = torch.nn.parallel.DistributedDataParallel(self.model_image2text, device_ids=[ self.args.local_rank],
                                                                          find_unused_parameters=True,output_device=self.args.local_rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank], find_unused_parameters=True, output_device=self.args.local_rank)

        print(f'models loaded into device: {self.args.local_rank}')

        if self.args.folds:
            split_train = self.args.folds.split(",")
            split_val = [str(self.args.val_fold)]
            split_test = [str(self.args.test_fold)]

        else:
            split_train ='train'
            split_val = 'val'
            split_test = 'test'


        self.train_dataloader = R2DataLoader(self.args, self.image2text_tokenizer, split=split_train, shuffle=False)
        self.val_dataloader = R2DataLoader(self.args, self.image2text_tokenizer, split= split_val, shuffle=False)
        self.test_dataloader = R2DataLoader(self.args, self.image2text_tokenizer, split=split_test, shuffle=False)

        self.train_dataloader_2 = R2DataLoadert2tBart(self.args, self.tokenizer, split=split_train,
                                                      shuffle=False)
        self.val_dataloader_2 = R2DataLoadert2tBart(self.args, self.tokenizer, split= split_val, shuffle=False)
        self.test_dataloader_2 = R2DataLoadert2tBart(self.args, self.tokenizer, split=split_test, shuffle=False)

        for epoch in range(self.start_epoch, self.epochs + 1):

            result, (val_outputs, val_labels, val_ids), (test_outputs, test_labels, test_ids)  = self._train_epoch(epoch)

            torch.distributed.barrier()

            val_outputs_list = [torch.zeros_like(val_outputs) for _ in range(self.args.n_gpu)]
            val_labels_list = [torch.zeros_like(val_labels) for _ in range(self.args.n_gpu)]
            val_ids_list = [torch.zeros_like(val_ids) for _ in range(self.args.n_gpu)]


            dist.all_gather(val_outputs_list, val_outputs)
            dist.all_gather(val_labels_list, val_labels)
            dist.all_gather(val_ids_list, val_ids)


            if test_outputs is not None:
                test_outputs_list = [torch.zeros_like(test_outputs) for _ in range(self.args.n_gpu)]
                test_labels_list = [torch.zeros_like(test_labels) for _ in range(self.args.n_gpu)]
                test_ids_list = [torch.zeros_like(test_ids) for _ in range(self.args.n_gpu)]

                dist.all_gather(test_outputs_list, test_outputs)
                dist.all_gather(test_labels_list, test_labels)
                dist.all_gather(test_ids_list, test_ids)
            else:
                test_outputs_list=None

            if dist.get_rank() == 0:  # when root
                val_outputs = torch.cat(val_outputs_list, dim=0)
                #print(f"After: {val_outputs.size()}")
                val_labels = torch.cat(val_labels_list , dim=0)
                val_ids= torch.cat(val_ids_list, dim=0)

                log = {'epoch': epoch}
                log.update(result)
                val_res, val_gts, ids=[],[],[]
                # tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                val_labels[val_labels == -100] = self.tokenizer.pad_token_id
                for output, label, id in zip(val_outputs.cpu().numpy(), val_labels.cpu().numpy(), val_ids.cpu().numpy()):
                    val_res.append(self.tokenizer.decode(output, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False))

                    val_gts.append(self.tokenizer.decode(label, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False))

                    ids.append(int(id))
                #
                val_dict={}
                for val_gt, val_re, id in zip(val_gts, val_res, ids):
                    val_dict[id]={'gt':val_gt,'res':val_re}

                val_gts = [v['gt'] for k,v in val_dict.items()]
                val_res = [v['res'] for k,v in val_dict.items()]

                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()})
                test_res=[]
                test_gts=[]
                ids=[]

                if test_outputs_list is not None:
                    test_outputs = torch.cat(test_outputs_list, dim=0)
                    test_labels = torch.cat(test_labels_list, dim=0)
                    test_labels[test_labels == -100] = self.tokenizer.pad_token_id
                    test_ids = torch.cat(test_ids_list, dim=0)

                    for output, label, id in zip(test_outputs.cpu().numpy(),test_labels.cpu().numpy(), test_ids.cpu().numpy()):
                        test_res.append(self.tokenizer.decode(output, skip_special_tokens=True,
                                                                       clean_up_tokenization_spaces=False))


                        test_gts.append(self.tokenizer.decode(label, skip_special_tokens=True,
                                                                             clean_up_tokenization_spaces=False))
                        ids.append(int(id))

                    # clean the redundent outputs beacuse of distributed sampling
                    #print(f"before: {len(ids)}")
                    test_dict = {}
                    for test_gt, test_re, id in zip(test_gts, test_res, ids):
                        test_dict[id] = {'gt': test_gt, 'res': test_re}

                    test_gts = [v['gt'] for k, v in test_dict.items()]
                    test_res = [v['res'] for k, v in test_dict.items()]
                    ids = [k for k,v in test_dict.items()]
                    #print(f"after: {len(ids)}")
                    #
                    test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                                {i: [re] for i, re in enumerate(test_res)})
                    log.update(**{'test_' + k: v for k, v in test_met.items()})

                    self._output_generation(test_res, test_gts, ids, epoch, 'test')
                # save logged informations into log dict
                self._record_best(log)

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop or self.args.resume:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=best)

        self._print_best()
        self._print_best_to_file()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    def _print_best_to_file(self):

        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time

        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'

        if self.best_recorder['test']:
            self.best_recorder['test']['time'] = crt_time
            self.best_recorder['test']['seed'] = self.args.seed
            self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        #print(f"n_gpu_use:{n_gpu_use}, list_ids:{list_ids}")
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):

        state = {
            'epoch': epoch,
            'state_dict': self.model_image2text.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }

        state_2 = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer_2.state_dict(),
            'monitor_best': self.mnt_best
        }

        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)

        filename_2 = os.path.join(self.checkpoint_dir, 'current_checkpoint_2.pth')
        torch.save(state_2, filename_2)

        print("Saving checkpoints: {} and {} ...".format(filename, filename_2))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)

            best_path_2 = os.path.join(self.checkpoint_dir, 'model_best_2.pth')
            torch.save(state_2, best_path_2)
            print("Saving current bests: model_best.pth and model_best_2.pth ...")

    def _resume_checkpoint(self, resume_path, resume_path_2):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        self.start_epoch = checkpoint['epoch'] + 1
        self.model_image2text.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        resume_path_2 = str(resume_path_2)
        print("Loading checkpoint: {} ...".format(resume_path_2))
        checkpoint_2 = torch.load(resume_path_2, map_location=torch.device('cpu'))
        self.mnt_best = checkpoint_2['monitor_best']
        self.model.load_state_dict(checkpoint_2['state_dict'])
        self.optimizer_2.load_state_dict(checkpoint_2['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        if self.args.do_test:
            improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
                self.mnt_metric_test]) or \
                            (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                                self.mnt_metric_test])
            if improved_test:
                self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class TrainerProgressive_(BaseTrainer_):
    def __init__(self, model, bart_tokenizer,
                 model_image2text,
                 tokenizer,
                 criterion,
                 metric_ftns,
                 optimizer,
                 args,
                 lr_scheduler,
                 optimizer_2,
                 lr_scheduler_2,
                 ):
        super(TrainerProgressive_, self).__init__(model, model_image2text, criterion, metric_ftns, optimizer, optimizer_2, args)
        self.lr_scheduler = lr_scheduler

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.lr_scheduler_2 = lr_scheduler_2

        self.train_dataloader_2 = None
        self.val_dataloader_2 = None
        self.test_dataloader_2 = None

        self.tokenizer = bart_tokenizer
        self.image2text_tokenizer = tokenizer

        # if args.dataset_name=='mimic_cxr':
        # #        size mismatch for encoder_decoder.model.tgt_embed.0.lut.weight: copying a param with shape torch.Size([315, 512]) from checkpoint, the shape in current model is torch.Size([1873, 512]).
        # #       size mismatch for encoder_decoder.logit.weight: copying a param with shape torch.Size([315, 512]) from checkpoint, the shape in current model is torch.Size([1873, 512]).
        # #       size mismatch for encoder_decoder.logit.bias: copying a param with shape torch.Size([315]) from checkpoint, the shape in current model is torch.Size([1873]).
        #     self.model_image2text =  self._load_checkpoint(model_image2text,os.path.join(args.best_checkpoint, 'model_best.pth'))
        #     self.model = self._load_checkpoint(model, os.path.join(args.best_checkpoint, 'model_best_2.pth'))


    def _add_noise(self, text, noise_vocab):
        words = text.split()
        for i in range(len(words)):
            if random.random() < 0.1:
                words[i] = random.choice(noise_vocab)
                if i + 1 < len(words) and random.random() < 0.5:
                    words[i + 1] = random.choice(noise_vocab)
                    if i + 2 < len(words) and random.random() < 0.5:
                        words[i + 2] = random.choice(noise_vocab)
                        if i + 3 < len(words) and random.random() < 0.5:
                            words[i + 3] = random.choice(noise_vocab)

        result = ''.join(' '.join(words))
        return result

    def _load_checkpoint(self, model, checkpoint_path):
        checkpoint_path = str(checkpoint_path)
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location= torch.device('cpu'))
        # setup GPU device if available, move model into configured device
        # self.device, device_ids = self._prepare_device(self.args.n_gpu)
        # if len(device_ids) > 1:
        #     model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model = model.to(self.device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded.")
        return model

    def _output_generation(self, predictions, gts, idxs, epoch, subset):
        # for evaluating and saving
        from nltk.translate.bleu_score import sentence_bleu
        import json
        # for saving json file
        output = list()

        for idx, pre, gt in zip(idxs, predictions, gts):
            score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt, 'bleu4': score})

        output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        output_filename = os.path.join(self.checkpoint_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_generated.json')
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False,indent=4)



    def _train_epoch(self, epoch):
        train_loss = 0
        self.model_image2text.train()
        log={}
        if not self.args.resume:
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.train_dataloader),desc='Image2Text'):

                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)

                output = self.model_image2text(images, reports_ids, mode='train')

                loss = self.criterion(output, reports_ids, reports_masks)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model_image2text.parameters(), 0.1)
                self.optimizer.step()
            self.lr_scheduler.step()

            self.model.train()
            #inputs_ids, attention_masks
            for batch_idx, (_, _, input_batch, decoder_inputs_ids, labels) in tqdm(enumerate(
                    self.train_dataloader_2), desc='Text2Text'):
                #add noise
                #random.seed(batch_idx)

                noise_input_batch=[]
                for input in input_batch:
                    #noise_input=self._add_noise(input, noise_vocab = self.train_dataloader_2.vocab)
                    noise_input_batch.append(f'<s>{input}</s>')

                input_encodings = self.tokenizer.batch_encode_plus(noise_input_batch, return_tensors="pt", pad_to_max_length=True,
                                                                   max_length=self.args.src_max_seq_length, truncation=True,
                                                                   add_special_tokens=False)
                inputs_ids= input_encodings['input_ids']
                attention_masks= input_encodings['attention_mask']

                # inputs_ids, attention_masks, decoder_inputs_ids, labels = inputs_ids.to(self.device), attention_masks.to(
                #     self.device), decoder_inputs_ids.to(self.device), labels.to(
                #     self.device)

                inputs_ids, attention_masks, decoder_inputs_ids, labels = inputs_ids.to(self.device), attention_masks.to(self.device), decoder_inputs_ids.to(self.device), labels.to(self.device)

                output = self.model(input_ids=inputs_ids, attention_mask=attention_masks,
                                    decoder_input_ids=decoder_inputs_ids, labels=labels)


                # loss = self.criterion(output[0], labels)
                loss = output[0]
                train_loss += loss.item()
                self.optimizer_2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer_2.step()
            log = {'train_loss': train_loss / len(self.train_dataloader)}
            self.lr_scheduler_2.step()

        self.model.eval()
        self.model_image2text.eval()
        with torch.no_grad():
            val_outputs=[]
            val_labels=[]
            val_ids=[]

            for batch_idx, (images_id, images, _, _, labels) in tqdm(enumerate(self.val_dataloader_2), desc="validation"):
                labels = labels.to(self.device)
                images = images.to(self.device)
                images_id = images_id.to(self.device)

                # #step=0
                output_0 = self.model_image2text(images, mode='sample')
                ##### required for the next step
                output_from_image2text = self.image2text_tokenizer.decode_batch(output_0.cpu().numpy())
                input_encodings = self.tokenizer.batch_encode_plus(output_from_image2text, return_tensors="pt",
                                                                         pad_to_max_length=True,
                                                                         max_length=self.args.max_seq_length,
                                                                         truncation=True)
                input_ids = input_encodings['input_ids']
                attention_mask = input_encodings['attention_mask']

                outputs = self.model(input_ids.to(self.device), attention_mask.to(self.device), mode='sample')

                for output in outputs:
                    # pad to max_seq_length
                    difference = self.args.max_seq_length - len(output)
                    output = torch.nn.functional.pad(output,(0,difference), value= self.tokenizer.pad_token_id)
                    val_outputs.append(output)

                for label in labels:
                    val_labels.append(label)

                for image_id in images_id:
                    val_ids.append(image_id)


            val_outputs = torch.stack(val_outputs).to(self.device)
            val_labels = torch.stack(val_labels).to(self.device)
            val_ids = torch.stack(val_ids).to(self.device)



            if self.args.do_test:
                self.model.eval()
                self.model_image2text.eval()
                with torch.no_grad():
                    test_outputs = []
                    test_labels = []
                    test_ids=[]
                    for batch_idx, (images_id, images, _, _, labels) in tqdm(enumerate(self.test_dataloader_2), desc="test"):
                        labels = labels.to(self.device)
                        images = images.to(self.device)
                        images_id = images_id.to(self.device)
                        # #step=0
                        output_0 = self.model_image2text(images, mode='sample')
                        ##### required for the next step
                        output_from_image2text = self.image2text_tokenizer.decode_batch(output_0.cpu().numpy())
                        input_encodings = self.tokenizer.batch_encode_plus(output_from_image2text, return_tensors="pt", pad_to_max_length=True, max_length=self.args.max_seq_length, truncation=True)
                        input_ids = input_encodings['input_ids']
                        attention_mask = input_encodings['attention_mask']

                        outputs = self.model(input_ids.to(self.device), attention_mask.to(self.device), mode='sample')

                        for output in outputs:
                            difference = self.args.max_seq_length - len(output)
                            output = torch.nn.functional.pad(output, (0, difference), value=self.tokenizer.pad_token_id)
                            test_outputs.append(output)

                        for label in labels:
                            test_labels.append(label)

                        for image_id in images_id:
                            test_ids.append(image_id)

                    test_outputs = torch.stack(test_outputs).to(self.device)
                    test_labels = torch.stack(test_labels).to(self.device)
                    test_ids = torch.stack(test_ids).to(self.device)
            else:
                test_outputs = None
                test_labels = None
                test_ids = None

        return log, (val_outputs, val_labels, val_ids), (test_outputs, test_labels, test_ids)


class MyBaseTrainer(object):
    def __init__(self, model, model_image2text, criterion, metric_ftns, optimizer, optimizer_2, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.model_image2text = model_image2text.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
            self.model_image2text = torch.nn.DataParallel(model_image2text, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.optimizer = optimizer
        self.optimizer_2= optimizer_2

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        self._print_best()
        self._print_best_to_file()


    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model_image2text.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }

        state_2 = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer_2.state_dict(),
            'monitor_best': self.mnt_best
        }

        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)

        filename_2 = os.path.join(self.checkpoint_dir, 'current_checkpoint_2.pth')
        torch.save(state_2, filename_2)

        print("Saving checkpoints: {} and {} ...".format(filename, filename_2))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)

            best_path = os.path.join(self.checkpoint_dir, 'model_best_2.pth')
            torch.save(state_2, best_path)
            print("Saving current bests: model_best.pth and model_best_2.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))

class My_TrainerProgressive(MyBaseTrainer):
    def __init__(self, model, model_image2text,
                 criterion,
                 metric_ftns,
                 optimizer,
                 args,
                 lr_scheduler,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 optimizer_2,
                 lr_scheduler_2,
                 train_dataloader_2,
                 val_dataloader_2,
                 test_dataloader_2
                 ):
        super(My_TrainerProgressive, self).__init__(model,model_image2text, criterion, metric_ftns, optimizer, optimizer_2, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.lr_scheduler_2 = lr_scheduler_2
        self.train_dataloader_2 = train_dataloader_2
        self.val_dataloader_2 = val_dataloader_2
        self.test_dataloader_2 = test_dataloader_2
        self.model_image2text=model_image2text
        self.model=model

    def _add_noise(self, text, noise_vocab):
        words = text.split()
        for i in range(len(words)):
            if random.random() < 0.1:
                words[i] = random.choice(noise_vocab)
                if i + 1 < len(words) and random.random() < 0.5:
                    words[i + 1] = random.choice(noise_vocab)
                    if i + 2 < len(words) and random.random() < 0.5:
                        words[i + 2] = random.choice(noise_vocab)
                        if i + 3 < len(words) and random.random() < 0.5:
                            words[i + 3] = random.choice(noise_vocab)

        result = ''.join(' '.join(words))
        return result

    def _load_checkpoint(self, model, checkpoint_path):
        checkpoint_path = str(checkpoint_path)
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(self.args.n_gpu)
        model = model.to(self.device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded.")
        return model

    def _output_generation(self, predictions, gts, idxs, epoch, subset):
        # for evaluating and saving
        from nltk.translate.bleu_score import sentence_bleu
        import json
        # for saving json file
        output = list()

        for idx, pre, gt in zip(idxs, predictions, gts):
            score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt, 'bleu4': score})

        output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        output_filename = os.path.join(self.checkpoint_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_generated.json')
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False,indent=4)

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model_image2text.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)
            output = self.model_image2text(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model_image2text.parameters(), 0.1)
            self.optimizer.step()

        self.lr_scheduler.step()

        self.model.train()
        #inputs_ids, attention_masks
        for batch_idx, (_, _, input_batch, decoder_inputs_ids, labels) in enumerate(
                self.train_dataloader_2):
            #add noise
            #random.seed(batch_idx)

            noise_input_batch=[]
            for input in input_batch:
                noise_input_batch.append(f'<s>{input}</s>')
            input_encodings = self.model.tokenizer.batch_encode_plus(noise_input_batch, return_tensors="pt", pad_to_max_length=True,
                                                               max_length=self.args.src_max_seq_length, truncation=True,
                                                               add_special_tokens=False)
            inputs_ids = input_encodings['input_ids']
            attention_masks= input_encodings['attention_mask']

            inputs_ids, attention_masks, decoder_inputs_ids, labels = inputs_ids.to(self.device), attention_masks.to(
                self.device), decoder_inputs_ids.to(self.device), labels.to(
                self.device)


            output = self.model(input_ids=inputs_ids, attention_mask=attention_masks,
                                decoder_input_ids=decoder_inputs_ids, labels=labels)


            loss = output[0]
            train_loss += loss.item()
            self.optimizer_2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer_2.step()


        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        self.model_image2text.eval()

        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, _, _, labels) in enumerate(self.val_dataloader_2):
                labels = labels.to(self.device)
                images = images.to(self.device)
                # #step=0
                output_0 = self.model_image2text(images, mode='sample')

                ##### required for the next step
                output_from_image2text = self.model_image2text.tokenizer.decode_batch(output_0.cpu().numpy())
                input_encodings = self.model.tokenizer.batch_encode_plus(output_from_image2text, return_tensors="pt",
                                                                         pad_to_max_length=True,
                                                                         max_length=self.args.max_seq_length,
                                                                         truncation=True)
                input_ids = input_encodings['input_ids']
                attention_mask = input_encodings['attention_mask']

                outputs = self.model(input_ids.to(self.device), attention_mask.to(self.device), mode='sample')
                # tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                reports = []
                for output in outputs.cpu().numpy():
                    reports.append(self.model.tokenizer.decode(output, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False))

                ground_truths = []
                labels[labels == -100] = self.model.tokenizer.pad_token_id
                for label in labels.cpu().numpy():
                    ground_truths.append(self.model.tokenizer.decode(label, skip_special_tokens=True,
                                                                     clean_up_tokenization_spaces=False))
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            #
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})


        self.model.eval()
        self.model_image2text.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            ids = []
            for batch_idx, (images_id, images, _, _, labels) in enumerate(self.test_dataloader_2):

                labels = labels.to(self.device)
                images = images.to(self.device)
                # #step=0
                output_0 = self.model_image2text(images, mode='sample')

                ##### required for the next step
                output_from_image2text = self.model_image2text.tokenizer.decode_batch(output_0.cpu().numpy())
                input_encodings = self.model.tokenizer.batch_encode_plus(output_from_image2text, return_tensors="pt", pad_to_max_length=True, max_length=self.args.max_seq_length, truncation=True)
                input_ids=input_encodings['input_ids']
                attention_mask=input_encodings['attention_mask']



                outputs = self.model(input_ids.to(self.device),attention_mask.to(self.device), mode='sample')
                # tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                reports = []
                for output in outputs.cpu().numpy():
                    reports.append(self.model.tokenizer.decode(output, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False))

                ground_truths = []
                labels[labels == -100] = self.model.tokenizer.pad_token_id
                for label in labels.cpu().numpy():
                    ground_truths.append(self.model.tokenizer.decode(label, skip_special_tokens=True,
                                                                     clean_up_tokenization_spaces=False))

                test_res.extend(reports)
                test_gts.extend(ground_truths)

                ids.extend([int(img_id) for img_id in images_id.cpu().numpy()])
            #
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            #
            self._output_generation(test_res, test_gts , ids, epoch, 'test')

        self.lr_scheduler_2.step()

        return log
class Trainert2tBart(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainert2tBart, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _output_generation(self, predictions, gts, idxs, epoch, subset):
        # for evaluating and saving

        # for saving json file
        output = list()

        for idx, pre, gt in zip(idxs, predictions, gts):
            score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt, 'bleu4': score})

        output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        output_filename = os.path.join(self.checkpoint_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_generated.json')

        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()

        for batch_idx, (_, _, input_batch, decoder_inputs_ids, labels) in enumerate(self.train_dataloader):
            decoder_inputs_ids, labels = decoder_inputs_ids.to(self.device), labels.to(self.device)

            noise_input_batch = []
            for input in input_batch:
                 noise_input_batch.append(f'<s>{input}</s>')

            input_encodings = self.model.tokenizer.batch_encode_plus(noise_input_batch, return_tensors="pt",
                                                                      pad_to_max_length=True,
                                                                      max_length=self.args.src_max_seq_length,
                                                                      truncation=True,
                                                                      add_special_tokens=False)
            inputs_ids = input_encodings['input_ids']
            attention_masks = input_encodings['attention_mask']

            output = self.model(input_ids=inputs_ids.to(self.device), attention_mask=attention_masks.to(self.device), decoder_input_ids=decoder_inputs_ids, labels=labels)
            #loss = self.criterion(output[0], labels)
            loss = output[0]
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        self.lr_scheduler.step()
        self.model.eval()

        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id,_, inputs_batch, decoder_inputs_ids, labels) in enumerate(self.val_dataloader):
                decoder_inputs_ids, labels =decoder_inputs_ids.to(self.device),labels.to(self.device)

                noise_input_batch = []
                for input in inputs_batch:
                    noise_input_batch.append(f'<s>{input}</s>')

                input_encodings = self.model.tokenizer.batch_encode_plus(noise_input_batch, return_tensors="pt",
                                                                         pad_to_max_length=True,
                                                                         max_length=self.args.src_max_seq_length,
                                                                         truncation=True,
                                                                         add_special_tokens=False)
                inputs_ids = input_encodings['input_ids']
                attention_masks = input_encodings['attention_mask']

                outputs = self.model(input_ids=inputs_ids.to(self.device), attention_mask=attention_masks.to(self.device), mode='sample')

                reports=[]
                for output in outputs.cpu().numpy():
                    reports.append(self.model.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False))

                ground_truths=[]
                labels[labels == -100] = self.model.tokenizer.pad_token_id
                for label in labels.cpu().numpy():
                    ground_truths.append(self.model.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=False))

                val_res.extend(reports)
                val_gts.extend(ground_truths)
            #
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            ids = []
            for batch_idx, (images_id,_, inputs_batch, decoder_inputs_ids, labels) in enumerate(
                    self.test_dataloader):
                decoder_inputs_ids, labels =  decoder_inputs_ids.to(self.device), labels.to(
                    self.device)

                noise_input_batch = []
                for input in inputs_batch:
                    noise_input_batch.append(f'<s>{input}</s>')

                input_encodings = self.model.tokenizer.batch_encode_plus(noise_input_batch, return_tensors="pt",
                                                                         pad_to_max_length=True,
                                                                         max_length=self.args.src_max_seq_length,
                                                                         truncation=True,
                                                                         add_special_tokens=False)
                inputs_ids = input_encodings['input_ids']
                attention_masks = input_encodings['attention_mask']

                outputs = self.model(input_ids=inputs_ids.to(self.device), attention_mask=attention_masks.to(self.device), mode='sample')

                reports=[]
                for output in outputs.cpu().numpy():
                    reports.append(self.model.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False))

                labels[labels == -100] = self.model.tokenizer.pad_token_id
                ground_truths=[]
                for label in labels.cpu().numpy():
                    ground_truths.append(self.model.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=False))

                test_res.extend(reports)
                test_gts.extend(ground_truths)
                ids.extend(images_id)
            #
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                       {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in val_met.items()})

            self._output_generation(test_res, test_gts, ids, epoch, 'test')



        return log

class TrainerGPT(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(TrainerGPT, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _output_generation(self, predictions, gts, idxs, epoch, subset):
        # for evaluating and saving
        from nltk.translate.bleu_score import sentence_bleu
        import json
        # for saving json file
        output = list()

        for idx, pre, gt in zip(idxs, predictions, gts):
            score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt, 'bleu4': score})

        output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        output_filename = os.path.join(self.checkpoint_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_generated.json')

        with open(output_filename, 'w') as writer:
            writer.write(json.dumps(output, ensure_ascii=False, indent=4))


    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        tr_loss = torch.tensor(0.0).to(self.device)
        for batch_idx, (images_id, images, inputs_ids, labels) in enumerate(self.train_dataloader):
            images,  inputs_ids, labels =  images.to(self.device),inputs_ids.to(self.device),  labels.to(
                self.device)
            output = self.model (images=images, input_ids=inputs_ids, labels=labels)

            loss = output[0]
            # loss = loss / self.args.gradient_accumulation_steps
            # tr_loss += loss
            train_loss += loss.item()
            # if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0 or (
            #         # last step in epoch but step is always smaller than gradient_accumulation_steps
            #         len(self.train_dataloader) <= self.args.gradient_accumulation_steps
            #         and (batch_idx + 1) == len(self.train_dataloader)
            #     ):
            #     self.optimizer.zero_grad()
            #     tr_loss.backward()
            #     torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            #     self.optimizer.step()
            #     tr_loss = torch.tensor(0.0).to(self.device)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()

        with torch.no_grad():
            val_gts, val_res = [], []

            for batch_idx, (images_id, images, inputs_ids, labels) in enumerate(self.val_dataloader):
                images, inputs_ids, labels = images.to(self.device), inputs_ids.to(self.device), labels.to(
                    self.device)

                outputs = self.model(images, mode='sample')

                reports=[]
                for output in outputs.cpu().numpy():
                    reports.append(self.model.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False))

                ground_truths=[]
                labels[labels == -100] = self.model.tokenizer.pad_token_id
                for label in labels.cpu().numpy():
                    ground_truths.append(self.model.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=False))

                val_res.extend(reports)
                val_gts.extend(ground_truths)
            #
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()

        with torch.no_grad():
            test_gts, test_res = [], []
            ids = []
            for batch_idx, (images_id, images, inputs_ids, labels) in enumerate(self.test_dataloader):
                images, inputs_ids, labels = images.to(self.device), inputs_ids.to(self.device), labels.to(
                    self.device)
                outputs = self.model(images, mode='sample')
                reports = []

                for output in outputs.cpu().numpy():
                    reports.append(self.model.tokenizer.decode(output, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False))

                ground_truths = []
                labels[labels == -100] = self.model.tokenizer.pad_token_id
                for label in labels.cpu().numpy():
                    ground_truths.append(self.model.tokenizer.decode(label, skip_special_tokens=True,
                                                                     clean_up_tokenization_spaces=False))

                test_res.extend(reports)
                test_gts.extend(ground_truths)

                ids.extend(images_id)
            #
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                       {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in val_met.items()})
            #
            self._output_generation(test_res, test_gts, ids, epoch, 'test')

        self.lr_scheduler.step()

        return log

