import os, sys, time
import numpy as np
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
      self.reset()
  
    def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0
  
    def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
      self.reset(total_epoch)
  
    def reset(self, total_epoch):
      assert total_epoch > 0
      self.total_epoch   = total_epoch
      self.current_epoch = 0
      self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
      self.epoch_losses  = self.epoch_losses - 1
  
      self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
      self.epoch_accuracy= self.epoch_accuracy
  
    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
      assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
      self.epoch_losses[idx, 0] = train_loss
      self.epoch_losses[idx, 1] = val_loss
      self.epoch_accuracy[idx, 0] = train_acc
      self.epoch_accuracy[idx, 1] = val_acc
      self.current_epoch = idx + 1
      return self.max_accuracy(False) == val_acc
  
    def max_accuracy(self, istrain):
      if self.current_epoch <= 0: return 0
      if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
      else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
  

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT='%Y-%m-%d'
    string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string + '-{}'.format(random.randint(1, 10000))
