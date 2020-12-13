# EDITOR: an Edit-Based Transformer with Repositioning
Source code for the TACL paper:

[**EDITOR: an Edit-Based Transformer with Repositioning for Neural Machine Translation with Soft Lexical Constraints**](https://arxiv.org/abs/2011.06868)

Weijia Xu, Marine Carpuat

## Installation
Dependencies:
* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options

To install fairseq-editor:
```
git clone https://github.com/Izecson/fairseq-editor.git
cd fairseq
pip install --editable .

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

## Training
The following command will train an EDITOR model on the binarized dataset in ``$bin_data_dir``:
```
fairseq-train \
	$bin_data_dir \
	--save-dir $model_dir \
	--ddp-backend=no_c10d \
	--task translation_lev \
	--criterion nat_loss \
	--arch editor_transformer \
	--noise random_delete_shuffle \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--lr 0.0005 --lr-scheduler inverse_sqrt \
	--min-lr '1e-09' --warmup-updates 10000 \
	--warmup-init-lr '1e-07' --label-smoothing 0.1 \
	--share-all-embeddings --no-share-discriminator \
	--dropout 0.3 --weight-decay 0.01 \
	--decoder-learned-pos --encoder-learned-pos \
	--apply-bert-init \
	--log-format 'simple' \
	--log-interval 100 \
	--fixed-validation-seed 7 \
	--max-tokens 8100 \
	--distributed-world-size 8 \
	--save-interval-updates 10000 \
	--max-update 300000
```

## Inference
To test on the test set:
```
fairseq-generate \
	$bin_data_dir \
	--gen-subset test \
	--task translation_lev \
	--path $model_dir/checkpoint_best.pt \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 1 --remove-bpe \
	--batch-size 16
```

To translate the input file with soft lexical constraints:
```
fairseq-interactive \
	-s $lang_src -t $lang_tgt \
	$bin_data_dir \
	--input $input \
	--task translation_lev \
	--path $model_dir/checkpoint_best.pt \
	--iter-decode-max-iter 10 \
	--iter-decode-eos-penalty 0 \
	--beam 1 --remove-bpe \
	--print-step --retain-iter-history \
	--constrained-decoding
```

Each line of the input file contains a source sentence and a constraint sequence separated by `` <sep> ``. An example of input for English-German is:
```
Gu@@ ta@@ ch : Incre@@ ased safety for pedestri@@ ans <sep> Sicherheit,Fußgän@@ ger
Two sets of lights so close to one another : inten@@ tional or just a sil@@ ly error ? <sep> Anlagen,Schil@@ d@@ bürger@@ stre@@ ich,Absicht
```

To translate with hard lexical constraints, add the argument ``--hard-constrained-decoding``.

## Acknowledgments
Our code was modified from [fairseq](https://github.com/pytorch/fairseq) codebase. We use the same license as fairseq(-py).

## Citation
```
@article{xu2020editor,
	title={EDITOR: an Edit-Based Transformer with Repositioning for Neural Machine Translation with Soft Lexical Constraints},
	author={Xu, Weijia and Carpuat, Marine},
	journal={arXiv preprint arXiv:2011.06868},
	year={2020}
}
```