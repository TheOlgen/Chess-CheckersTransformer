from Transformers.chess_transformers.play import load_model
from Transformers.chess_transformers.configs import import_config

CONFIG = import_config("CT-E-20")
model = load_model(CONFIG)