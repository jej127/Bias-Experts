from transformers import BertModel, BertPreTrainedModel
import torch
import torch.nn as nn
import numpy as np
import math

class BertForClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, confidence=None, teacher_probs=None, bias=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None : return logits
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits
    
class BertForPOE(BertPreTrainedModel):
    def __init__(self, config, lamda=0.3):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.lamda = lamda
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, confidence=None, teacher_probs=None, bias=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None : return logits

        logits_ = nn.functional.log_softmax(logits,-1)
        teacher_logits = torch.log(teacher_probs)
        loss_ce = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        loss_poe = self.loss_fn((logits_ + teacher_logits).view(-1, self.num_labels), labels.view(-1))
        return self.lamda * loss_ce + loss_poe, logits
    
class BertForReweighting(BertPreTrainedModel):
    def __init__(self, config, lamda=0.3):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.lamda = lamda
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.loss_fn2 = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, teacher_probs=None, bias=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None : return logits

        loss_ce = self.loss_fn2(logits.view(-1, self.num_labels), labels.view(-1))
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        one_hot_labels = nn.functional.one_hot(labels, num_classes=self.num_labels).float()
        weights = 1 - (one_hot_labels * teacher_probs).sum(1)
        return self.lamda * loss_ce + (weights * loss).sum() / weights.sum(), logits
    
class BertForConfidenceRegulation(BertPreTrainedModel):
    def __init__(self, config, lamda=0.3):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.lamda = lamda
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, teacher_probs=None, bias=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None : return logits

        loss_ce = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        p = nn.functional.softmax(logits,-1)
        one_hot_labels = nn.functional.one_hot(labels, num_classes=self.num_labels).float()
        weights = 1 - torch.sum(one_hot_labels * bias, dim=1, keepdims=True)
        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / torch.sum(exp_teacher_probs, dim=1, keepdims=True)
        loss = self.lamda * loss_ce - (norm_teacher_probs * p.log()).sum(1).mean()
        return loss, logits

class BertForPOE_annealed(BertPreTrainedModel):
    """Pre-trained BERT model that uses CF framework"""
    def __init__(self, config, lamda=0.3, max_a=1.0, min_a=0.8, total_steps=36816):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.lamda = lamda
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.linspace_a = np.linspace(max_a, min_a, total_steps+3)
        self.current_step = 0
        self.current_a = max_a
        self.post_init()

    def get_current_a(self):
        current_a = self.linspace_a[self.current_step]
        self.current_step += 1
        return current_a

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, confidence=None, teacher_probs=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None : return logits

        self.current_a = self.get_current_a()
        teacher_probs_ = teacher_probs ** self.current_a
        teacher_probs_ /= torch.sum(teacher_probs_, dim=1, keepdims=True)
        teacher_probs = teacher_probs_

        logits_ = nn.functional.log_softmax(logits,-1)
        teacher_logits = torch.log(teacher_probs)
        loss_ce = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        loss_poe = self.loss_fn((logits_ + teacher_logits).view(-1, self.num_labels), labels.view(-1))
        return self.lamda * loss_ce + loss_poe, logits
    
class BertForOnevsRest(BertPreTrainedModel):
    def __init__(self, config, label_weight=None):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.classifier = nn.ModuleList([self.get_classifier(classifier_dropout, config.hidden_size) for _ in range(self.num_labels)])
        if label_weight is not None:
            assert len(label_weight.size()) == 1 and label_weight.size(0) == self.num_labels
            self.label_weight = label_weight
        self.loss_fn = nn.BCELoss(reduction='none')  
        self.post_init()

    def get_classifier(self, dropout, hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, teacher_probs=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logits = torch.stack([self.classifier[i](outputs[0][:,0]).view(-1) for i in range(self.num_labels)], dim=1)
        if labels is None : return logits
        t = nn.functional.one_hot(labels, num_classes=self.num_labels).float()
        loss = self.loss_fn(torch.sigmoid(logits), t)
        
        # computing weights
        weight = t * (1-self.label_weight) + (1-t) * self.label_weight
        loss = (loss*weight).sum(1).mean()
        return loss, logits
    
class BertForOnevsRest_td(BertPreTrainedModel):
    def __init__(self, config, q=0.0, label_weight=None):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.classifier = nn.ModuleList([self.get_classifier(classifier_dropout, config.hidden_size) for _ in range(self.num_labels)])
        if label_weight is not None:
            assert len(label_weight.size()) == 1 and label_weight.size(0) == self.num_labels
            self.label_weight = label_weight
        self.loss_fn = nn.BCELoss(reduction='none')
        self.q = q
        self.post_init()

    def get_classifier(self, dropout, hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, confidence=None, teacher_probs=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logits = torch.stack([self.classifier[i](outputs[0][:,0]).view(-1) for i in range(self.num_labels)], dim=1)
        if labels is None : return logits
        t = nn.functional.one_hot(labels, num_classes=self.num_labels).float()
        loss = self.loss_fn(torch.sigmoid(logits), t)
        
        # computing weights
        confidence = confidence.view(-1,1)
        weight = t * (1-self.label_weight) * confidence**self.q + (1-t) * self.label_weight * (1-confidence)**self.q
        loss = (loss*weight).sum(1).mean()
        return loss, logits