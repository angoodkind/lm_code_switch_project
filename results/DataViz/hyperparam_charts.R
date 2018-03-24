library(tidyverse)

param.summ <- read_csv('hyperparameter/summary_compact.csv')
param.summ$emsize <- as.factor(param.summ$emsize)
param.summ$nhid <- as.factor(param.summ$nhid)
param.summ$nlayers <- as.factor(param.summ$nlayers)
summary(param.summ)



ggplot(param.summ) +
  geom_bar(aes(epoch, val_acc, fill="val_acc"), stat='summary', fun.y='mean') +
  geom_bar(aes(epoch, val_prec, fill="val_prec"), stat='summary', fun.y='mean')

## accuracy  
ggplot(param.summ, aes(x=epoch, y=val_acc, color=emsize, group=interaction(condition,run))) +
  geom_line() +
  facet_grid(nlayers ~ nhid)
  
  # geom_point(aes(y=val_acc, color=emsize, group=interaction(condition,run))) +
  stat_summary(fun.y='median', geom='line') +
  geom_smooth(method='loess', se=FALSE)

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_acc, color=nhid), fun.y='mean', geom='line')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_acc, color=nlayers), fun.y='mean', geom='line')

## loss
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=emsize), fun.y='mean', geom='line') +
  ggsave('epoch-val_loss-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=nhid), fun.y='mean', geom='line') +
  ggsave('epoch-val_loss-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_loss, color=nlayers), fun.y='mean', geom='line') +
  ggsave('epoch-val_loss-nlayers.pdf', device='pdf')

## f-score
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=emsize), fun.y='mean', geom='line') +
  ggsave('epoch-val_f-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=nhid), fun.y='mean', geom='line') +
  ggsave('epoch-val_f-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=val_f, color=nlayers), fun.y='mean', geom='line') +
  ggsave('epoch-val_f-nlayers.pdf', device='pdf')



## training loss
ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=emsize), fun.y='mean', geom='line') +
  ggsave('epoch-train_loss-emsize.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=nhid), fun.y='mean', geom='line') +
  ggsave('epoch-train_loss-nhid.pdf', device='pdf')

ggplot(param.summ, aes(x=epoch)) +
  stat_summary(aes(y=train_loss, color=nlayers), fun.y='mean', geom='line') +
  ggsave('epoch-train_loss-nlayers.pdf', device='pdf')



ggplot(param.summ) +
  geom_bar(aes(emsize, epoch_time), stat='summary', fun.y='mean')

ggplot(param.summ) +
  geom_bar(aes(nlayers, epoch_time), stat='summary', fun.y='mean')


  # stat_summary(aes(y=val_prec), fun.y='mean', geom='line') +
  # stat_summary(aes(y=val_recall), fun.y='mean', geom='line') +
  # labs(y='')
