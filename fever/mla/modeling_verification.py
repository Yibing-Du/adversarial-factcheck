# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning.utilities import rank_zero_info
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from attentions import (
    MultiHeadedAttention,
    SelfAttention,
)
from modeling_utils import (
    Classifier,
    PositionalEncoding,
)


class VerificationModel(PreTrainedModel):
    def __init__(self, hparams, num_labels):
        config = AutoConfig.from_pretrained(
            hparams.pretrained_model_name, num_labels=num_labels
        )
        super().__init__(config)
        self.num_labels = num_labels
        self.num_evidence = hparams.num_evidence
        self.max_seq_length = hparams.max_seq_length
        assert hparams.aggregate_mode in {"concat", "attn", "mean", "sum"}
        self.aggregate_mode = hparams.aggregate_mode
        rank_zero_info(f"aggregate mode: {hparams.aggregate_mode}")
        self.attn_bias_type = hparams.attn_bias_type
        rank_zero_info(f"attention bias type: {hparams.attn_bias_type}")

        setattr(
            self,
            self.config.model_type,
            AutoModel.from_pretrained(
                hparams.pretrained_model_name, config=self.config
            ),
        )

        if self.aggregate_mode == "concat":
            hidden_size = self.config.hidden_size * (hparams.num_evidence + 1)
        else:
            hidden_size = self.config.hidden_size

        self.classifier = Classifier(
            hidden_size, num_labels, dropout=hparams.classifier_dropout_prob
        )

        self.aggregate_attn = None
        if hparams.aggregate_mode == "attn":
            self.aggregate_attn = MultiHeadedAttention(self.config, self.attn_bias_type)

        self.sent_attn = None
        if hparams.sent_attn:
            self.sent_attn = SelfAttention(self.config)
            self.sent_position = PositionalEncoding(
                self.num_evidence, self.config.hidden_size
            )

        self.word_attn = None
        if hparams.word_attn:
            self.word_attn = SelfAttention(self.config)
            self.word_position = PositionalEncoding(
                self.num_evidence * self.max_seq_length, self.config.hidden_size
            )

    def get_logits(self, encoder_outputs, attention_mask=None, sent_scores=None):
        # hidden_states: batch*(evidence+1) x len x hidden
        # attention_mask: batch*(evidence+1) x len
        # sent_scores: batch x evidence
        num_evidence_plus = self.num_evidence + 1
        hidden_states = encoder_outputs.last_hidden_state
        hidden_size = self.config.hidden_size

        sents = None
        if self.word_attn:
            seq_length = self.num_evidence * self.max_seq_length
            sent_hidden_states = hidden_states.view(
                -1, num_evidence_plus, self.max_seq_length, self.config.hidden_size
            )
            sent_hidden_states = sent_hidden_states[:, 1:]  # skip claim
            sent_hidden_states = sent_hidden_states.view(
                -1, seq_length, self.config.hidden_size
            )
            sent_mask = attention_mask.view(-1, num_evidence_plus, self.max_seq_length)
            sent_mask = sent_mask[:, 1:]  # skip claim
            sent_mask = sent_mask.view(-1, seq_length)

            sent_hidden_states = self.word_position(sent_hidden_states)
            sent_hidden_states = self.word_attn(
                sent_hidden_states, sent_mask.unsqueeze(1)
            )

            # batch x evidence x len x hidden -> batch x evidence x hidden
            sent_hidden_states = sent_hidden_states.view(
                -1, self.num_evidence, self.max_seq_length, self.config.hidden_size
            )
            sents = sent_hidden_states[:, :, 0]  # equiv. to [CLS]

        # features: batch*(evidence+1) x hidden
        features = hidden_states[:, 0]  # equiv. to [CLS]

        # claims: batch x hidden
        # sents: batch x evidence x hidden
        claims = features[0::num_evidence_plus]
        if sents is None:
            sents = features.view(-1, num_evidence_plus, hidden_size)[:, 1:]

        if self.sent_attn:
            sents = self.sent_position(sents)
            sents = self.sent_attn(sents)

        if self.aggregate_mode == "sum":
            aggregate_output = sents.sum(dim=1)
        elif self.aggregate_mode == "mean":
            aggregate_output = sents.mean(dim=1)
        elif self.aggregate_mode == "attn":
            aggregate_output = self.aggregate_attn(
                claims,
                sents,
                sents,
                bias=sent_scores,
            ).squeeze(1)
        elif self.aggregate_mode == "concat":
            x = torch.cat([claims.unsqueeze(1), sents], dim=1)
            aggregate_output = x.view(x.size(0), -1)

        return self.classifier(aggregate_output)

    def create_outputs(self, loss, logits, encoder_outputs, return_dict):
        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def encoder(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert input_ids.dim() == 3  # batch x evidence x len
        input_ids = input_ids.view(-1, self.max_seq_length)
        attention_mask = attention_mask.view(-1, self.max_seq_length)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, self.max_seq_length)

        return getattr(self, self.config.model_type)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        selection_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_weights=None,
    ):
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits = self.get_logits(encoder_outputs, attention_mask)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return self.create_outputs(loss, logits, encoder_outputs, return_dict)


class VerificationJointModel(VerificationModel):
    def __init__(
        self,
        hparams,
        num_labels,
    ):
        super().__init__(hparams, num_labels=num_labels)
        assert 0.0 < hparams.lambda_joint <= 1.0
        self.lambda_joint = hparams.lambda_joint
        self.sent_num_labels = 2
        self.sent_classifier = Classifier(
            self.config.hidden_size,
            self.sent_num_labels,
            dropout=hparams.classifier_dropout_prob,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        selection_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_weights=None,
    ):
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # batch*(evidence+1) x hidden
        features = encoder_outputs.last_hidden_state[:, 0]  # equiv. to [CLS]

        logits_s = self.sent_classifier(features)
        logits_s = logits_s.view(-1, self.num_evidence + 1, self.sent_num_labels)[
            :, 1:
        ].contiguous()  # exclude claim

        selection_loss = None
        if selection_labels is not None:
            selection_labels = selection_labels[:, 1:].contiguous()  # exclude claim
            loss_fct = CrossEntropyLoss()
            selection_loss = loss_fct(
                logits_s.view(-1, self.sent_num_labels), selection_labels.view(-1)
            )

        sent_scores = None
        if self.attn_bias_type != "none":
            # sent_scores:  batch x evidence
            sent_scores = torch.softmax(logits_s, dim=-1)[:, :, 1]

        logits = self.get_logits(encoder_outputs, attention_mask, sent_scores)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if selection_loss is not None:
                loss = loss + (self.lambda_joint * selection_loss)

        return self.create_outputs(loss, logits, encoder_outputs, return_dict)
