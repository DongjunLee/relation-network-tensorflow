# Relation Network

TensorFlow implementation of [A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427) for bAbi task.

## Requirements

- Python 3.6
- TensorFlow >= 1.4
- [hb-config](https://github.com/hb-research/hb-config) (Singleton Config)
- requests
- tqdm (progress bar)
- [Slack Incoming Webhook URL](https://my.slack.com/services/new/incoming-webhook/)


## Project Structure

init Project by [hb-base](https://github.com/hb-research/hb-base)

    .
    ├── config                  # Config files (.yml, .json) using with hb-config
    ├── data                    # dataset path
    ├── notebooks               # Prototyping with numpy or tf.interactivesession
    ├── relation_network        # transformer architecture graphs (from input to logits)
        ├── __init__.py             # Graph logic
        ├── encoder.py              # Encoder
        └── relation.py             # Relation Network Module
    ├── data_loader.py          # raw_date -> precossed_data -> generate_batch (using Dataset)
    ├── hook.py                 # training or test hook feature (eg. print_variables)
    ├── main.py                 # define experiment_fn
    └── model.py                # define EstimatorSpec

Reference : [hb-config](https://github.com/hb-research/hb-config), [Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator), [experiments_fn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment), [EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec)

## Todo

- model was trained on the joint version of bAbI (all 20 tasks simultaneously), using the full dataset of 10K examples per task. (paper experiments)

## Config

Can control all **Experimental environment**.

example: bAbi_task1.yml

```yml
data:
  base_path: 'data/'
  task_path: 'en-10k/'
  task_id: 1
  PAD_ID: 0

model:
  batch_size: 64
  use_pretrained: false             # (true or false)
  embed_dim: 32                     # if use_pretrained: only available 50, 100, 200, 300
  encoder_type: uni                 # uni, bi
  cell_type: lstm                    # lstm, gru, layer_norm_lstm, nas
  num_layers: 1
  num_units: 32
  dropout: 0.5

  g_units:
    - 64
    - 64
    - 64
    - 64
  f_units:
    - 64
    - 128


train:
  learning_rate: 0.00003
  optimizer: 'Adam'                # Adagrad, Adam, Ftrl, Momentum, RMSProp, SGD

  train_steps: 200000
  model_dir: 'logs/bAbi_task1'

  save_checkpoints_steps: 1000
  check_hook_n_iter: 1000
  min_eval_frequency: 1

  print_verbose: False
  debug: False

slack:
  webhook_url: ""                  # after training notify you using slack-webhook
```

* debug mode : using [tfdbg](https://www.tensorflow.org/programmers_guide/debugger)


## Usage

Install requirements.

```pip install -r requirements.txt```

Then, prepare dataset.

```
sh scripts/fetch_babi_data.sh
```

Finally, start trand and evalueate model
```
python main.py --config bAbi_task1 --mode train_and_evaluate
```

### Experiments modes

:white_check_mark: : Working  
:white_medium_small_square: : Not tested yet.


- :white_check_mark: `evaluate` : Evaluate on the evaluation data.
- :white_medium_small_square: `extend_train_hooks` :  Extends the hooks for training.
- :white_medium_small_square: `reset_export_strategies` : Resets the export strategies with the new_export_strategies.
- :white_medium_small_square: `run_std_server` : Starts a TensorFlow server and joins the serving thread.
- :white_medium_small_square: `test` : Tests training, evaluating and exporting the estimator for a single step.
- :white_check_mark: `train` : Fit the estimator using the training data.
- :white_check_mark: `train_and_evaluate` : Interleaves training and evaluation.

---


### Tensorboar

```tensorboard --logdir logs```


## Reference

- [hb-research/notes - A simple neural network module for relational reasoning](https://github.com/hb-research/notes/blob/master/notes/relational_network.md)
- [arXiv - A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427)


## Author

[Dongjun Lee](https://github.com/DongjunLee) (humanbrain.djlee@gmail.com)
