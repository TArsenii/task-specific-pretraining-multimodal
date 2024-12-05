from typing import Dict, Optional

from torch import Tensor
from torch.nn import Embedding, Module
from transformers import BertConfig, BertModel


class LanguageEmbeddingLayer(Module):
    """Language embedding layer supporting both BERT and GloVe embeddings.

    This module provides text embedding functionality using either:
    1. BERT: Contextual embeddings from a pre-trained BERT model
    2. GloVe: Static word embeddings using a simple embedding layer

    The choice between BERT and GloVe is determined by the use_bert parameter
    passed during initialization.

    Args:
        use_bert: Whether to use BERT embeddings instead of GloVe
        word2id: Dictionary mapping words to indices (required if use_bert=False)
        embedding_dim: Dimension of embeddings (required if use_bert=False)
        bert_model_name: Name of the BERT model to use (optional, default='bert-base-uncased')
    """

    def __init__(
        self,
        use_bert: bool,
        word2id: Optional[Dict[str, int]] = None,
        embedding_dim: Optional[int] = None,
        bert_model_name: str = "bert-base-uncased",
    ) -> None:
        """Initialize the language embedding layer.

        Raises:
            ValueError: If using GloVe embeddings and word2id or embedding_dim is None
        """
        super().__init__()
        self.use_bert = use_bert

        if use_bert:
            self._init_bert_model(bert_model_name)
        else:
            if word2id is None or embedding_dim is None:
                raise ValueError("For GloVe embeddings, both word2id and embedding_dim must be provided")
            self._init_glove_embeddings(word2id, embedding_dim)

    def _init_bert_model(self, model_name: str) -> None:
        """Initialize the BERT model with appropriate configuration.

        Args:
            model_name: Name of the pre-trained BERT model to use
        """
        bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)

        self.bert_model = BertModel.from_pretrained(model_name, config=bert_config)

    def _init_glove_embeddings(self, word2id: Dict[str, int], embedding_dim: int) -> None:
        """Initialize the GloVe embedding layer.

        Args:
            word2id: Dictionary mapping words to indices
            embedding_dim: Dimension of the embeddings
        """
        self.embed = Embedding(num_embeddings=len(word2id), embedding_dim=embedding_dim)

    def forward(
        self,
        sentences: Optional[Tensor] = None,
        bert_sent: Optional[Tensor] = None,
        bert_sent_type: Optional[Tensor] = None,
        bert_sent_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for the language embedding layer.

        Args:
            sentences: Input tensor of token indices for GloVe embeddings
                      Shape: (batch_size, sequence_length)
            lengths: Sequence lengths for padding/masking (not used currently)
                    Shape: (batch_size,)
            bert_sent: Input tensor of token indices for BERT
                      Shape: (batch_size, sequence_length)
            bert_sent_type: Token type IDs for BERT
                           Shape: (batch_size, sequence_length)
            bert_sent_mask: Attention mask for BERT
                           Shape: (batch_size, sequence_length)

        Returns:
            Tensor containing the embedded representations:
            - For BERT: Shape (batch_size, sequence_length, hidden_size)
            - For GloVe: Shape (batch_size, sequence_length, embedding_dim)

        Raises:
            ValueError: If using BERT and any required BERT inputs are None
                       Or if using GloVe and sentences input is None
        """
        if self.use_bert:
            return self._forward_bert(bert_sent, bert_sent_type, bert_sent_mask)
        else:
            return self._forward_glove(sentences)

    def _forward_bert(
        self, bert_sent: Optional[Tensor], bert_sent_type: Optional[Tensor], bert_sent_mask: Optional[Tensor]
    ) -> Tensor:
        """Process inputs through BERT model.

        Args:
            bert_sent: Input tensor of token indices
            bert_sent_type: Token type IDs
            bert_sent_mask: Attention mask

        Returns:
            BERT embeddings tensor

        Raises:
            ValueError: If any required inputs are None
        """
        if any(x is None for x in [bert_sent, bert_sent_type, bert_sent_mask]):
            raise ValueError("All BERT inputs must be provided when use_bert=True")

        bert_output = self.bert_model(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)

        # Extract the last hidden state
        # Shape: (batch_size, sequence_length, hidden_size)
        return bert_output[0]

    def _forward_glove(self, sentences: Optional[Tensor]) -> Tensor:
        """Process inputs through GloVe embedding layer.

        Args:
            sentences: Input tensor of token indices

        Returns:
            GloVe embeddings tensor

        Raises:
            ValueError: If sentences input is None
        """
        if sentences is None:
            raise ValueError("Sentences input must be provided when use_bert=False")

        # Shape: (batch_size, sequence_length, embedding_dim)
        return self.embed(sentences)
