import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from nltk.translate.bleu_score import sentence_bleu
import json

from modules.visual_extractor import VisualExtractorDenseNet121 as VisualExtractor
from modules.meshed_memory_encoder_decoder import EncoderDecoder
from transformers import  BartForConditionalGeneration
from modules.optimizers import build_optimizer, build_lr_scheduler,build_optimizer_t2t
from modules.metrics import compute_scores


class M2TrGenModelProgressive(pl.LightningModule):
    def __init__(self, args, tokenizer, bart_tokenizer, criterion):
        super(M2TrGenModelProgressive, self).__init__()
        self.save_hyperparameters()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.args = args
        self.record_dir = args.record_dir
        self.args.max_seq_length = args.src_max_seq_length
        self.image2text_tokenizer = tokenizer
        self.t2t_tokenizer = bart_tokenizer


        self.visual_extractor = VisualExtractor(args)
        self.criterion=criterion

        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.text2text_bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)

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
        im_txt_optimizer = build_optimizer(self.args,self.encoder_decoder,self.visual_extractor)
        im_txt_sch = build_lr_scheduler(self.args, im_txt_optimizer)
        txt_txt_optimizer = build_optimizer_t2t(self.args, self.text2text_bart)
        txt_txt_sch = build_lr_scheduler(self.args, txt_txt_optimizer)
        return [im_txt_optimizer, txt_txt_optimizer],[im_txt_sch, txt_txt_sch]

    def training_step(self, batch, batch_idx, optimizer_idx):
        img_txt_opt, txt_txt_opt = self.optimizers()
        img_txt_sch,txt_txt_sch = self.lr_schedulers()
        # training_step defined the train loop.
        # It is independent of forward
        mages_id, images, reports_ids, reports_masks, input_batch_t2t, decoder_inputs_ids_t2t, labels_t2t  = batch[0]
        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
            self.device)

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        output = self.encoder_decoder(fc_feats, att_feats, reports_ids, mode='forward')
        loss = self.criterion(output, reports_ids, reports_masks)
        # Logging to TensorBoard by default
        self.log("train_loss_img_txt", loss)
        ##########################
        # Optimize img_2_txt #
        ##########################
        img_txt_opt.zero_grad()
        self.manual_backward(loss)
        img_txt_opt.step()

        # step every `n` epochs
        if self.trainer.is_last_batch:
            img_txt_sch.step()


        bart_input_batch = []
        for input in input_batch_t2t:
            # noise_input=self._add_noise(input, noise_vocab = self.train_dataloader_2.vocab)
            bart_input_batch.append(f'<s>{input}</s>')

        input_encodings_t2t = self.t2t_tokenizer.batch_encode_plus(bart_input_batch, return_tensors="pt",
                                                                   pad_to_max_length=True,
                                                                   max_length=self.args.src_max_seq_length, truncation=True,
                                                                   add_special_tokens=False)
        inputs_ids_t2t = input_encodings_t2t['input_ids']
        attention_masks_t2t = input_encodings_t2t['attention_mask']

        inputs_ids_t2t, attention_masks_t2t, decoder_inputs_ids_t2t, labels_t2t = inputs_ids_t2t.to(self.device), attention_masks_t2t.to(
            self.device), decoder_inputs_ids_t2t.to(self.device), labels_t2t.to(self.device)

        output = self.text2text_bart(input_ids=inputs_ids_t2t, attention_mask=attention_masks_t2t,
                                     decoder_input_ids=decoder_inputs_ids_t2t, labels=labels_t2t)
        # loss = self.criterion(output[0], labels)
        loss = output[0]
        ######################
        # Optimize txt_2_txt #
        ######################
        txt_txt_opt.zero_grad()
        self.manual_backward(loss)
        txt_txt_opt.step()

        # step every `n` epochs
        if self.trainer.is_last_batch:
            txt_txt_sch.step()
        return loss

    def validation_step(self, batch, batch_idx):

        images_id, images, reports_ids, reports_masks,_, _, labels_t2t= batch
        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
            self.device)

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        output_0, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        output_from_image2text = self.image2text_tokenizer.decode_batch(output_0.cpu().numpy())
        input_encodings_t2t = self.t2t_tokenizer.batch_encode_plus(output_from_image2text, return_tensors="pt",
                                                                   pad_to_max_length=True,
                                                                   max_length=self.args.max_seq_length,
                                                                   truncation=True)
        input_ids_t2t = input_encodings_t2t['input_ids']
        attention_mask_t2t = input_encodings_t2t['attention_mask']


        outputs = self.text2text_bart.generate(input_ids=input_ids_t2t.to(self.device), attention_mask=attention_mask_t2t.to(self.device),
                                               num_beams=self.args.beam_size, max_length=self.args.tgt_max_seq_length,
                                               early_stopping=True)

        t2t_output=[]
        for output in outputs:
            difference = self.args.max_seq_length - len(output)
            output = torch.nn.functional.pad(output, (0, difference), value=self.t2t_tokenizer.pad_token_id)
            t2t_output.append(output)

        outputs = torch.stack(t2t_output).to(self.device)
        reports=[]
        for output in outputs.cpu().numpy():
            reports.append(self.t2t_tokenizer.decode(output, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False))

        ground_truths=[]
        labels_t2t[labels_t2t[:, :] == -100] = self.t2t_tokenizer.pad_token_id
        for report in labels_t2t.cpu().numpy():
            ground_truths.append(self.t2t_tokenizer.decode(report, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False))
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
        # print logged informations to the screen
        print("")
        print('\tval_{:15s}: {}'.format(str("epoch"), self.current_epoch))
        for key, value in val_met.items():
            print('\tval_{:15s}: {}'.format(str(key), value))

        self.log('monitor_metrics', monitor_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images_id, images, reports_ids, reports_masks, _, _ , labels_t2t = batch

        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
            self.device)

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        output_0, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        output_from_image2text = self.image2text_tokenizer.decode_batch(output_0.cpu().numpy())
        input_encodings = self.t2t_tokenizer.batch_encode_plus(output_from_image2text, return_tensors="pt",
                                                               pad_to_max_length=True,
                                                               max_length=self.args.max_seq_length,
                                                               truncation=True)
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']

        outputs = self.text2text_bart.generate(input_ids=input_ids.to(self.device),
                                              attention_mask=attention_mask.to(self.device),
                                              num_beams=self.args.beam_size, max_length=self.args.tgt_max_seq_length,
                                              early_stopping=True)
        t2t_output = []
        for output in outputs:
            difference = self.args.max_seq_length - len(output)
            output = torch.nn.functional.pad(output, (0, difference), value=self.t2t_tokenizer.pad_token_id)
            t2t_output.append(output)

        outputs = torch.stack(t2t_output).to(self.device)

        reports = []
        for output in outputs.cpu().numpy():
            reports.append(self.t2t_tokenizer.decode(output, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False))
        ground_truths = []
        labels_t2t[labels_t2t[:, :] == -100] = self.t2t_tokenizer.pad_token_id
        for report in labels_t2t[:, 1:].cpu().numpy():
            ground_truths.append(self.t2t_tokenizer.decode(report, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False))
        return {"reports": reports, "gths": ground_truths,"image_ids":images_id.cpu().numpy()}

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
        # print logged informations to the screen
        print("")
        print('\ttest_{:15s}: {}'.format(str("epoch"), self.current_epoch))
        for key, value in test_met.items():
            print('\ttest_{:15s}: {}'.format(str(key), value))
        self.output_generation(reports, gths, image_ids, self.current_epoch, 'test', self.record_dir,test_met)

    @staticmethod
    def output_generation(predictions, gts, idxs, epoch, subset, record_dir,test_metrics):
        # for evaluating and saving

        # for saving json file
        output = list()

        for idx, pre, gt in zip(idxs, predictions, gts):
            score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx.item(), 'prediction': pre, 'ground_truth': gt, 'bleu4': score})

        output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        output_filename = os.path.join(record_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_generated.json')
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        output_metrics_filename = os.path.join(record_dir, 'Enc2Dec-' + str(epoch) + '_' + subset + '_metrics.json')
        with open(output_metrics_filename, 'w') as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=4)