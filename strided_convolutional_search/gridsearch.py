import os

for dataset in ['WN18RR', 'FB237']:
	for lr in [5e-5, 1e-4, 5e-4]:
		for wd in [5e-5, 1e-4, 5e-4]:
			print('submitting {}-LR{}-WD{}'.format(dataset,lr,wd))
			contents = """#$ -l gpu=1
#$ -l tmem=8G
#$ -l h_rt=6:00:00
#$ -S /bin/bash
#$ -N train
#$ -cwd

hostname

python /home/angulamb/darts-kbc/kbc/interleaved/train.py --dataset {} --learning_rate {} \
--learning_rate_min {} --emb_dim 1000 --channels 64 --arch ConvE --epochs 500 --batch_size 256 \
--reg 0 --report_freq 3 --optimizer Adam --layers 1 --weight_decay {} --seed 123 \
--interleaved
			""".format(dataset,lr, lr, wd)
			with open("autosub.sub", "w") as file:
				file.write(contents)
			os.system("qsub autosub.sub")