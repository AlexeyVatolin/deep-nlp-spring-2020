import torch
from torch import nn
from torch.functional import F

import numpy as np
from batcher import Batcher
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.embedding_v = nn.Embedding(vocab_size, embedding_size)
        self.embedding_u = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x_positive: torch.Tensor, y: torch.Tensor):
        # [batch_size, embedding_dim, 1]
        h = self.embedding_v(x_positive).mean(dim=1).unsqueeze(-1)
        # [batch_size, 1, 1]
        positive_score = torch.bmm(self.embedding_u(y).reshape(y.shape[0], 1, 100), h)
        # [batch_size]
        positive_score = - F.logsigmoid(positive_score.squeeze())

        # [batch_size, vocab_size, 1]
        negative_score = torch.bmm(self.embedding_u.weight.repeat(y.shape[0], 1, 1).neg(), h)
        # [batch_size]
        negative_score = negative_score.sigmoid().squeeze().sum(dim=1).log()
        return torch.mean(positive_score + negative_score)


class CBOW_Sampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.embedding_v = nn.Embedding(vocab_size, embedding_size)
        self.embedding_u = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x_positive: torch.Tensor, x_negative: torch.Tensor, y: torch.Tensor):
        # [batch_size, embedding_dim, 1]
        h = self.embedding_v(x_positive).mean(dim=1).unsqueeze(-1)
        # [batch_size, 1, embedding_dim]
        target_vector = self.embedding_u(y).unsqueeze(1)
        # [batch_size, 1, 1]
        positive_score = torch.bmm(target_vector, h)
        # [batch_size]
        positive_score = F.logsigmoid(positive_score).squeeze()

        # [batch_size, negative_samples, embedding_dim]
        negative_vectors = self.embedding_u(x_negative)
        # [batch_size, negative_samples, 1]
        negative_score = torch.bmm(-negative_vectors, h)
        # [batch_size]
        negative_score = F.logsigmoid(negative_score).squeeze(dim=-1).sum(dim=1)
        return -torch.mean(positive_score + negative_score)


class Word2Vec:
    def __init__(self, batcher: Batcher, embedding_size: int, model_type: str = 'cbow', device='cpu'):
        self.batcher = batcher
        if model_type == 'cbow':
            self.model: nn.Module = CBOW(batcher.vocab_size, embedding_size).to(device)
        elif model_type == 'cbow_neg_sampling':
            self.model: nn.Module = CBOW_Sampling(batcher.vocab_size, embedding_size).to(device)
        else:
            raise ValueError("model_type value must be in {'cbow', 'cbow_neg_sampling'}")
        self.device = device

    def fit(self, epochs: int, batch_size: int = 1, window_size: int = 5, lr: int = 0.1, log_every: int = 50,
            num_negative_samples: int = 0):
        optimizer = torch.optim.SGD(self.model.parameters(), lr)
        writer = SummaryWriter()
        global_iter = 0

        self.model.train()
        for epoch in range(epochs):
            t = tqdm(self.batcher.generate_batches(batch_size, window_size,
                                                   num_negative_samples=num_negative_samples), position=0)
            for i, batch in enumerate(t):

                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)

                self.model.zero_grad()
                loss = self.model(**batch)
                loss.backward()
                optimizer.step()

                if global_iter % log_every == 0:
                    writer.add_scalar('train_loss', loss.item(), global_iter)
                    t.set_description("loss: {:.4f}".format(loss.item()))
                    # for tag, param in self.model.named_parameters():
                    #     writer.add_histogram(tag, param.grad.data.cpu().numpy(), epoch)

                global_iter += 1

        writer.add_embedding(self.model.embedding_v.weight.detach().cpu().numpy(),
                             metadata=list(self.batcher.word2index.keys()))
        writer.flush()

    def __getitem__(self, word: str) -> np.ndarray:
        if word in self.batcher.word2index.keys():
            word_index = self.batcher.word2index[word]
            word_vector = self.model.embedding_v(torch.tensor([word_index], dtype=torch.long))
            return word_vector.detach().cpu().numpy()
        else:
            raise ValueError('Word {} not in model vocabulary'.format(word))

    def most_similar_cosine(self, word: str):
        if word in self.batcher.word2index.keys():
            word_index = self.batcher.word2index[word]
            word_vector = self.model.embedding_v(torch.tensor([word_index]))
            word_embeddings = self.model.embedding_v.weight
            word_vector = word_vector / word_vector.norm(dim=1, keepdim=True)
            word_embeddings = word_embeddings / word_embeddings.norm(dim=1, keepdim=True)
            dist = torch.abs(torch.mm(word_vector, word_embeddings.transpose(0, 1)).squeeze())
            top_distances, top_indexes = torch.topk(dist, 11, largest=True)
            top_words = [self.batcher.index2word[i] for i in top_indexes.detach().cpu().numpy()]
            return list(zip(top_words[1:], top_distances[1:].detach().cpu().numpy()))

        else:
            raise ValueError('Word {} not in model vocabulary'.format(word))




