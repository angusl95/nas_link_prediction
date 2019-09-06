from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')# reduce reduce_concat')

# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5',
#     'conv_3x3',
#     'conv_5x5',
#     'conv_7x7'
# ]

PRIMITIVES = [
    #'identity',
    'max_pool_3x3',
    'avg_pool_3x3',
    'conv_3x3',
    'conv_5x5',
    #'conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_1x5',
    'conv_5x1'
    #'conv_7x1_1x7'
    #'linear'
]

ConvE = Genotype(
    normal = [
    ('conv_3x3', 0)
    ],
  normal_concat = [0]
)

WNNet_1 = ConvE
FBNet_1 = ConvE
WNNet_2 = Genotype(normal=[('conv_3x3', 0), ('conv_5x5', 1)], normal_concat=[2])
FBNet_2 = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1)], normal_concat=[2])
WNNet_3 = Genotype(normal=[('conv_3x3', 0), ('conv_5x5', 1), ('conv_5x5', 2)], normal_concat=[3])
FBNet_3 = Genotype(normal=[('conv_3x3', 0), ('conv_5x5', 1), ('conv_3x3', 2)], normal_concat=[3])

WNNet_3_Pool = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('conv_5x5', 2)], normal_concat=[3])
FBNet_3_Pool = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('conv_3x3', 2)], normal_concat=[3])
