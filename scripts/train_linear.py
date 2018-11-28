import json, os

import torch
from torch.utils.data import DataLoader, TensorDataset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from models.holefilling.linear_model import HoleFillingBiLinear as Model

def check_dataset(transcripts_path) :
    ctxs, toks = list(), list()
    with open(transcripts_path) as f :
        for line in f :
            seq = [int(x) for x in line.split()]
            for a, b, c in zip(seq, seq[1:], seq[2:]) :
                ctxs.append((a, c))
                toks.append(b)
    t_ctx, t_tok = torch.LongTensor(ctxs), torch.LongTensor(toks)
    dataset = TensorDataset(t_ctx, t_tok)
    metadata = dict(
        num_unique_ctxs = t_ctx.max().item() + 1,
        num_unique_toks = t_tok.max().item() + 1,
    )
    return dataset, metadata


def train(args) :
    train_dataset, train_dataset_metadata = check_dataset(args.input_train)
    train_loader = DataLoader(train_dataset,
        batch_size = args.batch_size, shuffle = True,
    )

    model = Model(
        num_embeddings = train_dataset_metadata['num_unique_toks'],
        embedding_dim = args.emb_dim,
        share_weights = True,
    )
    optimizer = torch.optim.SGD(model.parameters(),
        lr = args.lr, momentum = args.momentum,
    )
    loss = torch.nn.NLLLoss()

    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            'accuracy': Accuracy(),
            'nll': Loss(loss),
        },
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("\rEpoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output), end='')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("\nTraining Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    model.reset_parameters()
    trainer.run(train_loader, max_epochs = args.n_epoch)

    return model


def main() :
    import argparse, itertools, os

    parser = argparse.ArgumentParser(description =
        '''Train hole-filling model.''')
    parser.add_argument('--input-train', required = True,
        help='directory containing processed phoneme transcripts.')
    parser.add_argument('--output-dir', required = True,
        help='output directory used to save model.metadata.json and model.weights.bin.')
    ## model hyperparameters
    parser.add_argument('--emb-dim', type=int, default=10)
    ## training hyperparameters
    parser.add_argument('--n-epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = train(args)

    model_metadata_path = os.path.join(args.output_dir, "model.metadata.json")
    with open(model_metadata_path, 'w') as f :
        json.dump(model.metadata, f)

    model_weights_path = os.path.join(args.output_dir, "model.weights.bin")
    torch.save(model.state_dict(), model_weights_path)


if __name__ == '__main__' :
    main()
