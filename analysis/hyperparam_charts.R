library(tidyverse)

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
       By embedding size, LSTM layer size and number of LSTM layers',
       x='Number of epochs',
       y='Accuracy on validation set') +
  theme(plot.title = element_text(hjust = 0.5)) +
  # ggsave('analysis/epoch-val_acc_facets.pdf', device='pdf')
  ggsave('analysis/epoch-val_acc_facets.jpeg', device='jpeg')
  
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_acc, color=nhid), fun.y='mean', geom='line')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_acc, color=nlayers), fun.y='mean', geom='line')+
  labs(title='Accuracy on validation set by epochs
       Ignoring Speaker Info',
       x='Number of epochs',
       y='Accuracy on validation set') +
  theme(plot.title = element_text(hjust = 0.5)) +
  # ggsave('analysis/epoch-val_acc_nlayers.pdf', device='pdf')
  ggsave('analysis/epoch-val_acc_nlayers.jpeg', device='jpeg')

## loss
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=emsize), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-val_loss-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=nhid), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-val_loss-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=nlayers), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-val_loss-nlayers.pdf', device='pdf')

## f-score
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=emsize), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-val_f-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=nhid), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-val_f-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=nlayers), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-val_f-nlayers.pdf', device='pdf')



## training loss
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=emsize), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-train_loss-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=nhid), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-train_loss-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=nlayers), fun.y='mean', geom='line') +
  ggsave('analysis/epoch-train_loss-nlayers.pdf', device='pdf')



ggplot(param.summ) +
  geom_bar(aes(emsize, epoch_time), stat='summary', fun.y='mean')

ggplot(param.summ) +
  geom_bar(aes(nlayers, epoch_time), stat='summary', fun.y='mean')

################### ALSO INCLUDE SPEAKER ############################

param.summ.speaker <- read_csv('results/include_speaker/summary.txt')
param.summ.speaker <- merge(param.summ.speaker, param.summ, all=TRUE)
param.summ.speaker$emsize <- as.factor(param.summ.speaker$emsize)
param.summ.speaker$nhid <- as.factor(param.summ.speaker$nhid)
param.summ.speaker$nlayers <- as.factor(param.summ.speaker$nlayers)
summary(param.summ.speaker)

param.summ.speaker <- subset(param.summ.speaker, nhid=='64' && emsize=='500' && nlayers=='1000')

## accuracy  
ggplot(param.summ.speaker, aes(x=epoch)) +
   stat_summary(aes(y=val_acc, color=ignore_speaker), fun.y='mean', geom='line') +
  labs(title='Accuracy on validation set by epochs',
       x='Number of epochs',
       y='Accuracy on validation set (%)') +
  theme(plot.title = element_text(hjust = 0.5)) +
  # ggsave('analysis/epoch-val_acc_all.pdf', device='pdf')
  ggsave('analysis/epoch-val_acc_all.jpeg', device='jpeg')

## loss
ggplot(param.summ.speaker, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=ignore_speaker), fun.y='mean', geom='line') +
  labs(title='Loss on validation set by epochs',
     x='Number of epochs',
     y='Loss on validation set') +
  theme(plot.title = element_text(hjust = 0.5)) +
  # ggsave('analysis/epoch-val_loss_all.pdf', device='pdf')
  ggsave('analysis/epoch-val_loss_all.jpeg', device='jpeg')

## f-score
ggplot(param.summ.speaker, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=ignore_speaker), fun.y='mean', geom='line') +
  labs(title='F-score on validation set by epochs',
       x='Number of epochs',
       y='F-score on validation set') +
  theme(plot.title = element_text(hjust = 0.5)) +
  # ggsave('analysis/epoch-val_f_all.pdf', device='pdf')
  ggsave('analysis/epoch-val_f_all.jpeg', device='jpeg')
