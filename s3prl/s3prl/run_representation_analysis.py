import torch
import argparse
from pathlib import Path
import json
import yaml
from transformers import AutoModel, AutoConfig, Wav2Vec2Model
import torchaudio
import random
import os

from representation_analysis import RepresentationAnalyzer
from util.download import download
from upstream.multi_distiller.model import MultiDistillerModel, MultiDistillerConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_model_path', type=str, required=True,
                      help='Path to the student model checkpoint')
    parser.add_argument('--student_config_path', type=str, required=True,
                      help='Path to the student model config yaml')
    parser.add_argument('--use_distilled_teachers', action='store_true',
                      help='Whether to use distilled teacher models instead of original models')
    parser.add_argument('--distilled_hubert_path', type=str,
                      default='/home/johnwei743251/s3prl/s3prl/result/pretrain/distill_hubert_2_layer_librispeech960/states-epoch-17.ckpt',
                      help='Path to distilled HuBERT model checkpoint')
    parser.add_argument('--distilled_mert_path', type=str,
                      default='/home/johnwei743251/s3prl/s3prl/result/pretrain/distill_mert_2_layer_music4all/states-epoch-43.ckpt',
                      help='Path to distilled MERT model checkpoint')
    parser.add_argument('--speech_data_dir', type=str, required=True,
                      help='Directory containing speech audio files')
    parser.add_argument('--music_data_dir', type=str, required=True,
                      help='Directory containing music audio files')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                      help='Directory to save analysis results')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of samples to analyze')
    parser.add_argument('--sample_rate', type=int, default=16000,
                      help='Sample rate for audio files')
    parser.add_argument('--max_duration', type=float, default=5.0,
                      help='Maximum duration of audio samples in seconds')
    return parser.parse_args()

def load_audio_files(data_dir, num_samples, sample_rate, max_duration, device):
    """Load audio files from directory"""
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    # Randomly sample files
    selected_files = random.sample(audio_files, min(num_samples, len(audio_files)))
    
    # Load and process audio files
    waveforms = []
    for file_path in selected_files:
        try:
            # Load audio file
            waveform, sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # Trim to max duration
            max_samples = int(max_duration * sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            waveforms.append(waveform.squeeze(0))  # Remove channel dimension
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return waveforms

def load_student_model(model_path, config_path):
    """Load the student MultiDistillerModel from checkpoint and config"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = MultiDistillerModel(MultiDistillerConfig(config["multi_distiller"]))
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint["Distiller"])
    
    # Set to eval mode
    model.eval()
    return model

def load_original_teacher_models(device):
    """Load original HuBERT and MERT models"""
    print("Loading original HuBERT model...")
    hubert = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")
    hubert = hubert.to(device)
    
    print("Loading original MERT model...")
    mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M")
    mert = mert.to(device)
    
    return hubert, mert

def load_distilled_teacher_models(hubert_path, mert_path, device):
    """Load distilled HuBERT and MERT models"""
    print("Loading distilled HuBERT model...")
    # Load HuBERT config
    hubert_config_path = os.path.join(os.path.dirname(hubert_path), 'config_model.yaml')
    with open(hubert_config_path, 'r') as f:
        hubert_config = yaml.safe_load(f)
    
    # Create and load HuBERT model
    hubert = MultiDistillerModel(MultiDistillerConfig(hubert_config["multi_distiller"]))
    hubert_checkpoint = torch.load(hubert_path, map_location='cpu')
    hubert.load_state_dict(hubert_checkpoint["Distiller"])
    hubert = hubert.to(device)
    
    print("Loading distilled MERT model...")
    # Load MERT config
    mert_config_path = os.path.join(os.path.dirname(mert_path), 'config_model.yaml')
    with open(mert_config_path, 'r') as f:
        mert_config = yaml.safe_load(f)
    
    # Create and load MERT model
    mert = MultiDistillerModel(MultiDistillerConfig(mert_config["multi_distiller"]))
    mert_checkpoint = torch.load(mert_path, map_location='cpu')
    mert.load_state_dict(mert_checkpoint["Distiller"])
    mert = mert.to(device)
    
    return hubert, mert

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load models
    print("Loading student model...")
    student_model = load_student_model(args.student_model_path, args.student_config_path)
    student_model = student_model.to(device)  # Move student model to device
    
    print("Loading teacher models...")
    if args.use_distilled_teachers:
        speech_teacher, music_teacher = load_distilled_teacher_models(
            args.distilled_hubert_path,
            args.distilled_mert_path,
            device
        )
    else:
        speech_teacher, music_teacher = load_original_teacher_models(device)
    
    # Create analyzer
    analyzer = RepresentationAnalyzer(student_model, speech_teacher, music_teacher)
    
    # Load real audio data
    print("Loading speech data...")
    speech_inputs = load_audio_files(
        args.speech_data_dir, 
        args.num_samples, 
        args.sample_rate, 
        args.max_duration,
        device
    )
    
    print("Loading music data...")
    music_inputs = load_audio_files(
        args.music_data_dir, 
        args.num_samples, 
        args.sample_rate, 
        args.max_duration,
        device
    )
    
    print(f"Loaded {len(speech_inputs)} speech samples and {len(music_inputs)} music samples")
    
    # Run analysis
    print("Running representation analysis...")
    results = analyzer.analyze_representations(speech_inputs, music_inputs)
    
    # Convert tensor values to Python native types
    def convert_tensor_to_native(obj):
        if isinstance(obj, torch.Tensor):
            # Move tensor to CPU first
            obj = obj.detach().cpu()
            return obj.item() if obj.numel() == 1 else obj.numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensor_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensor_to_native(v) for v in obj]
        return obj
    
    # Convert results to native Python types
    results_native = convert_tensor_to_native(results)
    
    # Save results
    print("Saving results...")
    with open(output_dir / 'similarity_results.json', 'w') as f:
        json.dump(results_native, f, indent=2)
    
    # Plot heatmaps
    print("Generating heatmaps...")
    analyzer.plot_similarity_heatmap(results, save_path=output_dir / 'similarity_heatmaps.png')
    
    print("Analysis complete! Results saved to:", output_dir)

if __name__ == '__main__':
    main() 
