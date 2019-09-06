import os
import sys
import time
import glob
import tqdm
import torch
import utils
import pickle
import logging
import argparse
import genotypes
import torch.utils
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from torch import optim
from typing import Dict
from datasets import Dataset
from models import CP, ComplEx
from torch.autograd import Variable
from model import NetworkKBC as Network
from regularizers import N2, N3, Regularizer

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--channels', type=int, default=36, help='num of channels')
parser.add_argument('--layers', type=int, default=1, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='KBCNet', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--interleaved', action='store_true', default=False, help='interleave subject and relation embeddings rather than stacking')
parser.add_argument('--label_smooth', type=float, default = 0.1, help='label smoothing parameter')
datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
parser.add_argument('--dataset', choices=datasets, help="Dataset in {}".format(datasets))
regularizers = ['N3', 'N2']
parser.add_argument('--regularizer', choices=regularizers, default='N3', help="Regularizer in {}".format(regularizers))
parser.add_argument('--emb_dim', default=200, type=int, help="Embedding dimension")
parser.add_argument('--init', default=1e-3, type=float, help="Initial scale")
parser.add_argument('--reg', default=0, type=float, help="Regularization weight")
optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer', choices=optimizers, default='Adagrad', help="Optimizer in {}".format(optimizers))
parser.add_argument('--decay1', default=0.9, type=float, help="decay rate for the first moment estimate in Adam")
parser.add_argument('--decay2', default=0.999, type=float, help="decay rate for second moment estimate in Adam")
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  dataset = Dataset(args.dataset)
  train_examples = torch.from_numpy(dataset.get_train().astype('int64'))

  #TODO: does below need reintroducing somewhere?
  # device = 'cuda'
  # model.to(device)

  CLASSES = dataset.get_shape()[0]

  #criterion = nn.CrossEntropyLoss(reduction='mean')
  criterion = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion = criterion.cuda()

  regularizer = {
    'N2': N2(args.reg),
    'N3': N3(args.reg),
  }[args.regularizer]

  genotype = eval("genotypes.%s" % args.arch)
  logging.info('genotype = %s', genotype)
  model = Network(args.channels,
    CLASSES, args.layers, criterion, regularizer, genotype, args.interleaved,
    dataset.get_shape(), args.emb_dim, args.init)
  model = model.cuda()

  optimizer = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
  }[args.optimizer]()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  #optimizer = torch.optim.SGD(
  #    model.parameters(),
  #    args.learning_rate,
      #momentum=args.momentum,
      #weight_decay=args.weight_decay
  #    )

  train_queue = torch.utils.data.DataLoader(
      train_examples, batch_size=args.batch_size,
      shuffle = True,
      #sampler=torch.utils.data.sampler.RandomSampler(),
      pin_memory=True, num_workers=2)

  #TODO do we want the learning rate min here?
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  best_acc = 0
  patience = 0
  #curve = {'train': [], 'valid': [], 'test': []}
  curve = {'valid':[], 'test':[]}

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    train_epoch(train_examples, train_queue, model, optimizer, 
      regularizer, args.batch_size)
    if (epoch + 1) % args.report_freq == 0:
      valid, test = [
              avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
              for split in ['valid', 'test']#, 'train']
          ]
      curve['valid'].append(valid)
      curve['test'].append(test)
      #curve['train'].append(train)

      #print("\t TRAIN: ", train)
      print("\t VALID: ", valid)
      print("\t TEST: ", test)

      is_best = False
      if valid['MRR'] > best_acc:
        best_acc = valid['MRR']
        is_best = True
        patience = 0
      else:
        patience +=1

      utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc_top1': best_acc,
        'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)

      if patience >= 10:
        print('no improvement for 10 evaluations, early stopping...')
        break

    #utils.save(model, os.path.join(args.save, 'weights.pt'))
    #torch.save(model.embeddings, os.path.join(args.save, 'embeddings.pt'))

  results = dataset.eval(model, 'test', -1)
  print("\n\nTEST : ", results)
  with open(os.path.join(args.save, 'curve.pkl'), 'wb') as f:
    pickle.dump(curve, f, pickle.HIGHEST_PROTOCOL)

def train_epoch(train_examples, train_queue, model, optimizer: optim.Optimizer, 
  regularizer: Regularizer, batch_size: int, verbose: bool = True):
  loss = nn.CrossEntropyLoss(reduction='mean')
  with tqdm.tqdm(total=train_examples.shape[0], unit='ex', disable=not verbose) as bar:
      bar.set_description(f'train loss')
      for step, input in enumerate(train_queue):
          model.train()

          input_var = Variable(input, requires_grad=False).cuda()
          target_var = Variable(input[:,2], requires_grad=False).cuda()#async=True)

          predictions, factors = model.forward(input_var)
          truth = input_var[:, 2]

          l_fit = loss(predictions, truth)
          l_reg = regularizer.forward(factors)
          l = l_fit + l_reg

          optimizer.zero_grad()
          l.backward()
          nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
          optimizer.step()

          bar.update(input_var.shape[0])
          bar.set_postfix(loss=f'{l.item():.0f}')

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h.numpy()}

if __name__ == '__main__':
  main() 
