import torch
import torch.nn as nn
from torch_geometric.data import Data
from box import Box
from .tokenizer import Tokenizer
from .embeddings import EmbeddingLayer
from .rnn import ContextualRNN
from .gcn import GCNLayer
from .attention import MultiFacetedAttention
import src.config as config


class MultilingualHypergraphTransformer(nn.Module):
    def __init__(self):
        super(MultilingualHypergraphTransformer, self).__init__()
        self.embedding_layer = EmbeddingLayer()
        self.contextual_rnn = ContextualRNN(
            input_dim=config.EMBEDDING_SIZE, hidden_dim=config.CONTEXTUAL_HIDDEN_SIZE
        )
        self.gcn_layer = GCNLayer(
            in_channels=config.EMBEDDING_SIZE, out_channels=config.HIDDEN_SIZE
        )
        self.attention_layer = MultiFacetedAttention(hidden_size=config.EMBEDDING_SIZE)
        self.classifier = nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES)

    def tokenize(self, texts):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(texts)
        return tokens

    def forward(self, texts):
        tokenized_output = self.tokenize(texts)
        boxed_output = Box(tokenized_output)
        input_ids = boxed_output.input_ids
        attention_mask = tokenized_output.attention_mask
        embeddings = self.embedding_layer(input_ids, attention_mask)
        contextual_embeddings = self.contextual_rnn(embeddings)
        attention_output = self.attention_layer(contextual_embeddings)

        num_nodes = attention_output.size(1)
        edge_index = (
            torch.tensor(
                [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        hypergraph_data = Data(x=attention_output, edge_index=edge_index)
        output = self.gcn_layer(hypergraph_data.x, hypergraph_data.edge_index)

        return self.classifier(output.mean(dim=1))

    def train_model(self, training_data, target_data):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

        for epoch in range(config.EPOCHS):
            optimizer.zero_grad()
            outputs = self(training_data)
            loss = nn.CrossEntropyLoss()(outputs, target_data)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == target_data).sum().item()
            accuracy = correct / target_data.size(0)

            print(
                f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%"
            )

    def predict(self, texts):
        self.eval()
        with torch.no_grad():
            output = self(texts)
            return output.argmax(dim=-1)
