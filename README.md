# gpt

Generative Pre-trained Transformer in PyTorch from scratch

<figure>
<div align="center">
<a href="https://anitamaczan.pl/#problem_n_cial" target="_blank">
<img src="https://anitamaczan.pl/problem_n_cial.jpg" width="200" alt="'N-body problem' by Anita Maczan, Acrylic on canvas, 80x100, 2024">
</a>
</div>
</p>
</figure>

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

## License

GPL v3

Jędrzej Maczan, 2024
