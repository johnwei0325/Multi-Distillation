from transformers import WhisperModel

# Load the small Whisper model
model = WhisperModel.from_pretrained("openai/whisper-small")

# Access the 12th encoder layer
encoder_layer_12 = model.encoder.layers[11]  # Index is 0-based, so 11 is the 12th layer

print(len(model.encoder.layers))

