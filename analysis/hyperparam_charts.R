library(tidyverse)
library(gridExtra)

param.summ <- read_csv('results/hyperparameter/summary.txt')
param.summ$emsize <- as.factor(param.summ$emsize)
param.summ$nhid <- as.factor(param.summ$nhid)
param.summ$nlayers <- as.factor(param.summ$nlayers)
summary(param.summ)

ggplot(param.summ) +
  geom_bar(aes(epoch, val_acc, fill="val_acc"), stat='summary', fun.y='mean') +
  geom_bar(aes(epoch, val_prec, fill="val_prec"), stat='summary', fun.y='mean')

## accuracy  
labels_layers <- c('1'="1 Layer", '2'="2 Layers")
labels_hid <- c('32'="32 Units", '64'="64 Units")

ggplot(param.summ, aes(x=epoch, y=val_acc, color=emsize, group=interaction(condition,run))) +
  geom_line() +
  facet_grid(nlayers ~ nhid, labeller=labeller(nlayers=labels_layers,
                                               nhid=labels_hid)) +
  labs(title='Accuracy on validation set by epochs
       Grouped by embedding size, LSTM layer size,
       and number of LSTM layers',
       x='Number of epochs',
       y='Accuracy on validation set') +
  theme(plot.title = element_text(hjust = 0.5)) +
  # ggsave('analysis/charts/epoch-val_acc_facets.pdf', device='pdf')
  ggsave('analysis/charts/epoch-val_acc_facets.jpeg', device='jpeg')
  
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_acc, color=nhid), fun.y='mean', geom='line')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_acc, color=nlayers), fun.y='mean', geom='line')+
  labs(title='Mean accuracy on validation set by epochs
       Ignoring Speaker Info',
       x='Number of epochs',
       y='Accuracy on validation set') +
  theme(plot.title = element_text(hjust = 0.5)) +
  # ggsave('analysis/charts/epoch-val_acc_nlayers.pdf', device='pdf')
  ggsave('analysis/charts/epoch-val_acc_nlayers.jpeg', device='jpeg')

## loss
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=emsize), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-val_loss-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=nhid), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-val_loss-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=nlayers), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-val_loss-nlayers.pdf', device='pdf')

## f-score
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=emsize), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-val_f-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=nhid), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-val_f-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=nlayers), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-val_f-nlayers.pdf', device='pdf')



## training loss
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=emsize), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-train_loss-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=nhid), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-train_loss-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=nlayers), fun.y='mean', geom='line') +
  ggsave('analysis/charts/epoch-train_loss-nlayers.pdf', device='pdf')



ggplot(param.summ) +
  geom_bar(aes(emsize, epoch_time), stat='summary', fun.y='mean')

ggplot(param.summ) +
  geom_bar(aes(nlayers, epoch_time), stat='summary', fun.y='mean')

################### ALSO INCLUDE SPEAKER ############################

param.summ.optimal <- subset(param.summ, nhid=='64' & emsize=='500' & nlayers=='2')

param.summ.speaker <- read_csv('results/include_speaker/summary.txt')
param.summ.speaker <- merge(param.summ.optimal, param.summ.speaker, all=TRUE)
param.summ.speaker$emsize <- as.factor(param.summ.speaker$emsize)
param.summ.speaker$nhid <- as.factor(param.summ.speaker$nhid)
param.summ.speaker$nlayers <- as.factor(param.summ.speaker$nlayers)
summary(param.summ.speaker)

## accuracy  
spkr.acc = ggplot(param.summ.speaker, aes(x=epoch)) +
   stat_summary(aes(y=val_acc, color=ignore_speaker), fun.y='mean', geom='line') +
  labs(title='Accuracy',
       x='Number of epochs',
       y='Mean proportion correct\non validation set',
       color='Use speaker info') +
  scale_color_manual(labels = c("Yes", "No"), values = c("#F8766D", "#00BFC4")) +
  theme(plot.title = element_text(hjust = 0.5))

## loss
spkr.loss = ggplot(param.summ.speaker, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=ignore_speaker), fun.y='mean', geom='line') +
  labs(title='Loss',
     x='Number of epochs',
     y='Mean loss\non validation set',
     color='Use speaker info') +
  scale_color_manual(labels = c("Yes", "No"), values = c("#F8766D", "#00BFC4")) +
  theme(plot.title = element_text(hjust = 0.5))

## f-score
spkr.f = ggplot(param.summ.speaker, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=ignore_speaker), fun.y='mean', geom='line') +
  labs(title='F-Score',
       x='Number of epochs',
       y='Mean F-score\non validation set',
       color='Use speaker info') +
  scale_color_manual(labels = c("Yes", "No"), values = c("#F8766D", "#00BFC4")) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("analysis/charts/spkr/epoch-val-acc.jpeg", spkr.acc + guides(color=FALSE), width=2.25, height=2)
ggsave("analysis/charts/spkr/epoch-val-f.jpeg", spkr.f + guides(color=FALSE), width=2.25, height=2)
ggsave("analysis/charts/spkr/epoch-val-loss.jpeg", spkr.loss, width=3.375, height=2)


################### PARTIAL CONTEXT ############################

# speaker info does better, so let's update param.sum.optimal to reflect that
param.summ.optimal <- subset(param.summ.speaker, ignore_speaker=='False')
param.summ.optimal$full_context = rep('True',nrow(param.summ.optimal))

param.summ.context <- read_csv('results/partial_context/summary.txt')
param.summ.context <- merge(param.summ.context, param.summ.optimal, all=TRUE)

## accuracy  
cont.acc = ggplot(param.summ.context, aes(x=epoch)) +
  aes(y=val_acc, color=full_context, group=save) + geom_line() +
  labs(title='Accuracy',
       x='Number of epochs',
       y='Proportion correct\non validation set') +
  theme(plot.title = element_text(hjust = 0.5))

## loss
cont.loss = ggplot(param.summ.context, aes(x=epoch)) +
  aes(y=val_loss, color=full_context, group=save) + geom_line() +
  labs(title='Loss',
       x='Number of epochs',
       y='Loss\non validation set') +
  theme(plot.title = element_text(hjust = 0.5))

## f-score
cont.f = ggplot(param.summ.context, aes(x=epoch)) +
  aes(y=val_f, color=full_context, group=save) + geom_line() +
  labs(title='F-score',
       x='Number of epochs',
       y='F-score\non validation set',
       color='Amount of context') +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("analysis/charts/context/epoch-val-acc.jpeg", cont.acc + guides(color=FALSE), width=2.25, height=2)
ggsave("analysis/charts/context/epoch-val-f.jpeg", cont.f + guides(color=FALSE), width=2.25, height=2)
ggsave("analysis/charts/context/epoch-val-loss.jpeg", cont.loss + labs(color='Amount of context') + scale_color_manual(labels = c("Current utterance", "Full discourse"), values = c("#F8766D", "#00BFC4")), width=3.375, height=2)
