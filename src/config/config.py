import yaml
from box import Box

with open("model.yaml", "r") as f:
    boxed_config = Box(yaml.safe_load(f))

EMBEDDING_SIZE = boxed_config.model.embedding_size
CONTEXTUAL_HIDDEN_SIZE = boxed_config.model.contextual_hidden_size
HIDDEN_SIZE = boxed_config.model.hidden_size
NUM_CLASSES = boxed_config.model.num_classes
BATCH_SIZE = boxed_config.model.batch_size
EPOCHS = boxed_config.model.epochs
