import torch
import torch.nn as nn
import torch.nn.functional as F

from operations import *
from genotypes import Genotype
from genotypes import PRIMITIVES
from abc import ABC, abstractmethod
from torch.autograd import Variable
from typing import Tuple, List, Dict

class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores > targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

class MixedOp(nn.Module):

  def __init__(self, C, stride, emb_dim, dropout):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, emb_dim, False, dropout)
      #TODO reintroduce this?
      #if 'pool' in primitive:
      #  op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, emb_dim, C, dropout=0.2):
    super(Cell, self).__init__()
    self._steps = steps
    self._emb_dim = emb_dim
    self._dropout = dropout
    self._ops = nn.ModuleList()
    for i in range(self._steps):
      stride = 2
      op = MixedOp(C*(4**i), stride, emb_dim, self._dropout)
      self._ops.append(op)

  def forward(self, s0, weights):
    #states = [s0]
    #offset = 0
    for i in range(self._steps):
      s0 = self._ops[i](s0, weights[i])
      #s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      #offset += len(states)
      #states.append(s)

    #return torch.cat(states[-self._multiplier:], dim=1)
    return s0

class Network(KBCModel):

  def __init__(self, C, num_classes, layers, criterion, regularizer, 
    interleaved, sizes: Tuple[int, int, int], emb_dim: int, 
    init_size: float = 1e-3, steps=4):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._regularizer = regularizer
    self._steps = steps
    self.emb_dim = emb_dim
    if self.emb_dim % 32 != 0:
      raise ValueError('embedding size must be divisble by 32')
    self.emb_height = self.emb_dim//32
    self.sizes = sizes
    self._init_size = init_size
    self._interleaved = interleaved
    self.embeddings = nn.ModuleList([
      #TODO restore sparse here?
            nn.Embedding(s, emb_dim, max_norm=1)#, sparse=True)
            for s in sizes[:2]
        ])
    self.embeddings[0].weight.data *= init_size
    self.embeddings[1].weight.data *= init_size
    #self.embeddings = torch.load('embeddings_conve.pt')

    self.cells = nn.ModuleList()
    for i in range(layers):
      cell = Cell(steps, self.emb_dim, self._C)
      self.cells += [cell]

    self.input_drop = torch.nn.Dropout(p=0.2)
    self.input_bn = torch.nn.BatchNorm2d(1, affine=False)
    self.projection = nn.Linear(2*self.emb_dim*self._C, self.emb_dim)#, bias=False)
    #self.projection = nn.Linear(C_prev, self.emb_dim, bias=False)
    #self.classifier = nn.Linear(C_prev, num_classes)

    self.output_bn = nn.BatchNorm1d(self.emb_dim, affine=False)
    self.output_drop = torch.nn.Dropout(p=0.3)
    self._initialize_alphas()
    self.epoch=None

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, 
      self._regularizer, self._interleaved,
      self.sizes, self.emb_dim, self._init_size, self._steps).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def lhs_rel_forward(self,lhs,rel):
    if self._interleaved:
      lhs = lhs.view([lhs.size(0),1,self.emb_height,32])
      rel = rel.view([rel.size(0),1,self.emb_height,32])
      s0 = torch.cat([lhs,rel],3)
      s0 = s0.view([lhs.size(0),1,2*self.emb_height,32])
    else:
      lhs = lhs.view([lhs.size(0),1,self.emb_height,32])
      rel = rel.view([rel.size(0),1,self.emb_height,32])
      s0 = torch.cat([lhs,rel], 2)
    s0 = self.input_bn(s0)
    s0 = self.input_drop(s0)
    s0 = s0.expand(-1,self._C, -1, -1)

    for i, cell in enumerate(self.cells):
      weights = F.softmax((1.05**self.epoch)* self.alphas_normal, dim=-1)
      s0 = cell(s0, weights)
    out = s0.view(s0.size(0),1, -1)
    out = self.projection(out)
    out = out.squeeze()
    out = self.output_drop(out)
    out = self.output_bn(out)
    out = F.relu(out)
    return out


  def score(self, x):
    lhs = self.embeddings[0](x[:, 0])
    rel = self.embeddings[1](x[:, 1])
    rhs = self.embeddings[0](x[:, 2])
    to_score = self.embeddings[0].weight
    out = self.lhs_rel_forward(lhs,rel)
    out = torch.sum(
        out * rhs, 1, keepdim=True
    )
    return out

  def forward(self, x):
    lhs = self.embeddings[0](x[:, 0])
    rel = self.embeddings[1](x[:, 1])
    rhs = self.embeddings[0](x[:, 2])
    to_score = self.embeddings[0].weight

    out = self.lhs_rel_forward(lhs,rel)
    out = out @ to_score.transpose(0,1)
    return out, (lhs,rel,rhs)

  def get_rhs(self, chunk_begin: int, chunk_size: int):
    return self.embeddings[0].weight.data[
        chunk_begin:chunk_begin + chunk_size
    ].transpose(0, 1)

  def get_queries(self, queries: torch.Tensor):
    lhs = self.embeddings[0](queries[:, 0])
    rel = self.embeddings[1](queries[:, 1])

    out = self.lhs_rel_forward(lhs,rel)

    return out

  def _loss(self, input, target):  
    #TODO: definitely just alias for forward method?
    logits, factors = self(input)
    l_fit = self._criterion(logits, target)

    #l_reg = self._regularizer.forward(factors)
    
    #return self._criterion(logits, target) 
    return l_fit #+ l_reg

  def _initialize_alphas(self):
    #k = sum(1 for i in range(self._steps) for n in range(1+i))
    k = self._steps
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    #set middle identity strength to zero to force learning convolution
    #self.alphas_normal.data[2,0] = -1e8
    #self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [self.alphas_normal]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      #n = 2
      n = 1
      start = 0
      for i in range(self._steps):
        W = weights.copy()
        k_best = None
        for k in range(len(W[i])):
          if k_best is None or W[i][k] > W[i][k_best]:
            k_best = k
        gene.append((PRIMITIVES[k_best], i))
        #start = end
        #n += 1
      return gene

    gene_normal = _parse(F.softmax((1.05**self.epoch)*self.alphas_normal, dim=-1).data.cpu().numpy())

    #concat = range(2+self._steps-self._multiplier, self._steps+2)
    concat = [self._steps]
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat
    )
    return genotype

