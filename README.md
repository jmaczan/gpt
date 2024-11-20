# gpt

> 💜 See also the article how to build the minimalistic GPT version in Paged Out! #5 Issue, page 6 ["GPT in PyTorch"](https://pagedout.institute/download/PagedOut_005.pdf#page=6)

Generative Pre-trained Transformer in PyTorch from scratch

## Train

### CLI
```sh
python src/train.py
```

Options:
```sh
--batch_size 64
--num-epochs 100
--lr 0.0001
--from-checkpoint checkpoint_path.pth
```

Model is checkpointed after each epoch and stored in `checkpoints/` directory


### Code
```py
from train import train

train()
```

## Run

### CLI

```sh
python src/run.py --from-checkpoint checkpoint_path.pth
```

### Code
```py
from run import run

run(model_path="checkpoint_path.pth", prompt="Rick:\nMorty, where are you?)
```

## Cite
If you use this software in your research, please use the following citation:

```bibtex
@misc{Maczan_GPT_2024,
  title = "Generative Pre-trained Transformer in PyTorch",
  author = "{Maczan, Jędrzej Paweł}",
  howpublished = "\url{https://github.com/jmaczan/gpt}",
  year = 2024,
  publisher = {GitHub}
}
```

## License

GPL v3

## Author

Jędrzej Maczan, 2024
