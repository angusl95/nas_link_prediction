import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("e_s", fillcolor='darkseagreen2')
  g.node("r_r", fillcolor='darkseagreen2')
  g.node("e_o", fillcolor='darkseagreen2')
  #assert len(genotype) % 2 == 0
  #steps = len(genotype) // 2
  steps = len(genotype)

  for i in range(steps+1):
    g.node(str(i), fillcolor='lightblue')

  g.edge("e_s", "0", label="concat", fillcolor="gray")
  g.edge("r_r", "0", label="concat", fillcolor="gray")

  for i in range(steps):
      op, _ = genotype[i]
      g.edge(str(i), str(i+1), label=op, fillcolor="gray")
      g.edge(str(i), str(i+1), fillcolor="gray")
      g.edge(str(i), str(i+1), fillcolor="gray")
      g.edge(str(i), str(i+1), fillcolor="gray")

  g.node("6", fillcolor='lightblue')
  g.node("f", fillcolor='palegoldenrod')

  g.edge(str(steps), "6", label="fully\nconnected", fillcolor="gray")
  
  #for i in range(steps):
  #  g.edge(str(i), "c_{i}", fillcolor="gray")
  g.edge("6", "f", label="dot product", fillcolor="gray")
  g.edge("e_o", "f", label="dot product", fillcolor="gray" )
  g.render(filename, view=True)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot(genotype.normal, "normal")
  #plot(genotype.reduce, "reduction")

