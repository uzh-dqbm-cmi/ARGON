import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from nltk.translate.bleu_score import sentence_bleu
import json

from modules.visual_extractor import VisualExtractorDenseNet121 as VisualExtractor
from modules.meshed_memory_encoder_decoder import EncoderDecoder
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.metrics import compute_scores


class M2TrModel(pl.LightningModule):
    def __init__(self, args, tokenizer, criterion):
        super(M2TrModel, self).__init__()
        self.save_hyperparameters()

        self.args = args
        self.checkpoint_dir = args.save_dir
        self.args.max_seq_length = args.src_max_seq_length
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.criterion=criterion


        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        self.metric_ftns= compute_scores

        # # in case we use lr_scheduler.step()
        # self.automatic_optimization = False

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, mages_id, images, reports_ids, reports_masks ):
        # in lightning, forward defines the prediction/inference actions
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')

        return output

    def configure_optimizers(self):
        # build optimizer, learning rate scheduler
        optimizer=build_optimizer(self.args,self.encoder_decoder,self.visual_extractor)
        sch = build_lr_scheduler(self.args, optimizer)

        return {"optimizer": optimizer,
                "lr_scheduler" : {
                "scheduler" : sch,
                'interval': 'epoch'
                }
            }

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        mages_id, images, reports_ids, reports_masks=batch
        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
            self.device)

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        output = self.encoder_decoder(fc_feats, att_feats, reports_ids, mode='forward')
        loss = self.criterion(output, reports_ids, reports_masks)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        # # step every `n` epochs
        # if self.trainer.is_last_batch:
        #     sch.step()

        return loss

    def validation_step(self, batch, batch_idx):
        images_id, images, reports_ids, reports_masks=batch
        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
            self.device)

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        reports = self.tokenizer.decode_batch(output.cpu().numpy())
        ground_truths = self.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        return {"reports": reports, "gths": ground_truths}

    def validation_epoch_end(self, outputs):
        gths=[]
        reports = []
        for x in outputs:
            gths.extend(x["gths"])
            reports.extend(x["reports"])

        val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(gths)},
                                   {i: [re]for i, re in enumerate(reports)})


        monitor_metrics=val_met[self.args.monitor_metric]
        self.log('monitor_metrics', monitor_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images_id, images, reports_ids, reports_masks=batch
        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
            self.device)

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        output,_ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        reports = self.tokenizer.decode_batch(output.cpu().numpy())
        ground_truths = self.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        return {"reports": reports, "gths": ground_truths,"images_id":images_id.cpu().numpy()}

    def test_epoch_end(self, outputs):
        gths = []
        reports = []
        image_ids=[]

        for x in outputs:
            gths.extend(x["gths"])
            reports.extend(x["reports"])
            image_ids.extend(x["image_ids"])
        test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(gths)},
                                   {i: [re] for i, re in enumerate(reports)})

        self._output_generation(reports, gths, image_ids, self.current_epoch(), 'test', self.checkpoint_dir)


def _output_generation(predictions, gts, idxs, epoch, subset, checkpoint_dir):
    # for evaluating and saving

    # for saving json file
    output = list()

    for idx, pre, gt in zip(idxs, predictions, gts):
        score = sentence_bleu([gt.split()], pre.split())
        output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt, 'bleu4': score})

    output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
    output_filename = os.path.join(checkpoint_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_generated.json')
    with open(output_filename, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)