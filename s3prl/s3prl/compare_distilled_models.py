import torch
import torchaudio
import argparse
import json
import yaml
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from upstream.multi_distiller.model import MultiDistillerModel, MultiDistillerConfig
from representation_analysis import RepresentationAnalyzer
import torch.nn.functional as F

def load_model(model_path, config_path):
    """Load a model from checkpoint"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = MultiDistillerModel(MultiDistillerConfig(config["multi_distiller"]))
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint["Distiller"])
    
    return model

def load_audio_files(data_dir, max_duration=5.0, sample_rate=16000, num_samples=100):
    """Load and process audio files from directory"""
    audio_files = []
    
    # Get all audio files
    for ext in ['.wav', '.mp3', '.flac']:
        audio_files.extend(list(Path(data_dir).rglob(f'*{ext}')))
    
    # Randomly sample files if we have more than num_samples
    if len(audio_files) > num_samples:
        import random
        audio_files = random.sample(audio_files, num_samples)
    
    waveforms = []
    for audio_file in audio_files:
        try:
            # Load audio - convert PosixPath to string
            waveform, sr = torchaudio.load(str(audio_file))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # Trim to max duration
            max_samples = int(max_duration * sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            waveforms.append(waveform.squeeze(0))
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    return waveforms

def convert_tensor_to_native(obj):
    """Convert PyTorch tensors to native Python types"""
    if isinstance(obj, torch.Tensor):
        # Move tensor to CPU first
        obj = obj.detach().cpu()
        return obj.item() if obj.numel() == 1 else obj.numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_native(v) for v in obj]
    return obj

def plot_model_comparison(results, save_path=None):
    """Create a comparison histogram between all distilled models"""
    # Extract model names and their results
    model_names = list(results.keys())
    
    # Create data arrays for each combination
    data = {
        'Speech Teacher + Speech Audio': [],
        'Speech Teacher + Music Audio': [],
        'Music Teacher + Speech Audio': [],
        'Music Teacher + Music Audio': []
    }
    
    # Collect data for each model
    for model in model_names:
        # Speech teacher with speech audio
        speech_speech = np.mean([
            results[model]['speech']['layer_0']['hubert_layer_0'].cpu().numpy(),
            results[model]['speech']['layer_1']['hubert_layer_0'].cpu().numpy()
        ])
        
        # Speech teacher with music audio
        speech_music = np.mean([
            results[model]['music']['layer_0']['hubert_layer_0'].cpu().numpy(),
            results[model]['music']['layer_1']['hubert_layer_0'].cpu().numpy()
        ])
        
        # Music teacher with speech audio
        music_speech = np.mean([
            results[model]['speech']['layer_0']['mert_layer_0'].cpu().numpy(),
            results[model]['speech']['layer_1']['mert_layer_0'].cpu().numpy()
        ])
        
        # Music teacher with music audio
        music_music = np.mean([
            results[model]['music']['layer_0']['mert_layer_0'].cpu().numpy(),
            results[model]['music']['layer_1']['mert_layer_0'].cpu().numpy()
        ])
        
        data['Speech Teacher + Speech Audio'].append(speech_speech)
        data['Speech Teacher + Music Audio'].append(speech_music)
        data['Music Teacher + Speech Audio'].append(music_speech)
        data['Music Teacher + Music Audio'].append(music_music)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set width of bars
    barWidth = 0.15
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(model_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    
    # Create bars
    ax.bar(r1, data['Speech Teacher + Speech Audio'], width=barWidth, label='Speech Teacher + Speech Audio')
    ax.bar(r2, data['Speech Teacher + Music Audio'], width=barWidth, label='Speech Teacher + Music Audio')
    ax.bar(r3, data['Music Teacher + Speech Audio'], width=barWidth, label='Music Teacher + Speech Audio')
    ax.bar(r4, data['Music Teacher + Music Audio'], width=barWidth, label='Music Teacher + Music Audio')
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Similarity Score')
    ax.set_title('Model Performance Across Different Teacher-Audio Combinations')
    ax.set_xticks([r + barWidth*1.5 for r in range(len(model_names))])
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Add value labels on top of bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    add_value_labels(ax.patches)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add insights as text
    insights = []
    
    # Compare models
    for i, model in enumerate(model_names):
        # Best performing combination for this model
        best_combo = max(data.items(), key=lambda x: x[1][i])
        insights.append(f"{model}: Best performance with {best_combo[0]} (score: {best_combo[1][i]:.3f})")
        
        # Cross-domain performance
        speech_speech = data['Speech Teacher + Speech Audio'][i]
        music_music = data['Music Teacher + Music Audio'][i]
        if abs(speech_speech - music_music) > 0.1:
            insights.append(f"{model}: {'Better at speech' if speech_speech > music_music else 'Better at music'} representation")
    
    # Add insights to plot
    insight_text = "\n".join(insights)
    plt.figtext(0.02, 0.02, insight_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def analyze_representations(student_model, teacher_model, audio_path, layer_idx=0, device='cuda'):
    """Analyze representations from a specific layer"""
    # Get representations (already normalized in get_*_representations)
    student_rep = get_hubert_representations(student_model, audio_path, layer_idx, device)
    teacher_rep = get_hubert_representations(teacher_model, audio_path, layer_idx, device)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(
        student_rep.unsqueeze(0),
        teacher_rep.unsqueeze(0)
    )
    
    return similarity.item()

def get_hubert_representations(model, audio_path, layer_idx, device='cuda'):
    """Get representations from a specific layer of HuBERT model"""
    # Load and preprocess audio
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    # Move to device and add batch dimension
    wav = wav.to(device)
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    
    # Get model outputs
    with torch.no_grad():
        if isinstance(model, MultiDistillerModel):
            # For distilled models
            feat_final, layer_hidden = model(wav)
            
            # Get layer representation
            if layer_idx == 0:
                layer_rep = feat_final
            else:
                layer_rep = layer_hidden[0]  # Only one layer in distilled model
        else:
            # For original HuBERT
            hidden_states = model.extract_features(wav)[1]
            layer_rep = hidden_states[layer_idx]
    
    # Normalize the layer representation
    layer_rep = F.normalize(layer_rep, p=2, dim=-1)
    
    return layer_rep

def get_mert_representations(model, audio_path, layer_idx, device='cuda'):
    """Get representations from a specific layer of MERT model"""
    # Load and preprocess audio
    wav, sr = torchaudio.load(audio_path)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    
    # Move to device and add batch dimension
    wav = wav.to(device)
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    
    # Get model outputs
    with torch.no_grad():
        if isinstance(model, MultiDistillerModel):
            # For distilled models
            feat_final, layer_hidden = model(wav)
            
            # Get layer representation
            if layer_idx == 0:
                layer_rep = feat_final
            else:
                layer_rep = layer_hidden[0]  # Only one layer in distilled model
        else:
            # For original MERT
            hidden_states = model.extract_features(wav)[1]
            layer_rep = hidden_states[layer_idx]
    
    # Normalize the layer representation
    layer_rep = F.normalize(layer_rep, p=2, dim=-1)
    
    return layer_rep

def main():
    parser = argparse.ArgumentParser(description='Compare different distilled models')
    parser.add_argument('--student_models', nargs='+', required=True,
                      help='Paths to student model checkpoints')
    parser.add_argument('--student_configs', nargs='+', required=True,
                      help='Paths to student model configs')
    parser.add_argument('--speech_data_dir', required=True,
                      help='Directory containing speech audio files')
    parser.add_argument('--music_data_dir', required=True,
                      help='Directory containing music audio files')
    parser.add_argument('--output_dir', required=True,
                      help='Directory to save results')
    parser.add_argument('--distilled_hubert_path', type=str,
                      default='/home/johnwei743251/s3prl/s3prl/result/pretrain/distill_hubert_2_layer_librispeech960/states-epoch-17.ckpt',
                      help='Path to distilled HuBERT model checkpoint')
    parser.add_argument('--distilled_mert_path', type=str,
                      default='/home/johnwei743251/s3prl/s3prl/result/pretrain/distill_mert_2_layer_music4all/states-epoch-43.ckpt',
                      help='Path to distilled MERT model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of samples to analyze from each dataset')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load teacher models
    print("Loading teacher models...")
    # Load HuBERT config
    hubert_config_path = os.path.join(os.path.dirname(args.distilled_hubert_path), 'config_model.yaml')
    with open(hubert_config_path, 'r') as f:
        hubert_config = yaml.safe_load(f)
    
    # Create and load HuBERT model
    speech_teacher = MultiDistillerModel(MultiDistillerConfig(hubert_config["multi_distiller"]))
    hubert_checkpoint = torch.load(args.distilled_hubert_path, map_location='cpu')
    speech_teacher.load_state_dict(hubert_checkpoint["Distiller"])
    
    # Load MERT config
    mert_config_path = os.path.join(os.path.dirname(args.distilled_mert_path), 'config_model.yaml')
    with open(mert_config_path, 'r') as f:
        mert_config = yaml.safe_load(f)
    
    # Create and load MERT model
    music_teacher = MultiDistillerModel(MultiDistillerConfig(mert_config["multi_distiller"]))
    mert_checkpoint = torch.load(args.distilled_mert_path, map_location='cpu')
    music_teacher.load_state_dict(mert_checkpoint["Distiller"])
    
    # Load student models
    print("Loading student models...")
    student_models = []
    for model_path, config_path in zip(args.student_models, args.student_configs):
        model = load_model(model_path, config_path)
        student_models.append(model)
    
    # Load audio data
    print("Loading audio data...")
    speech_inputs = load_audio_files(args.speech_data_dir, num_samples=args.num_samples)
    music_inputs = load_audio_files(args.music_data_dir, num_samples=args.num_samples)
    
    print(f"Loaded {len(speech_inputs)} speech samples and {len(music_inputs)} music samples")
    
    # Analyze each student model
    results = {}
    for i, student_model in enumerate(student_models):
        print(f"\nAnalyzing student model {i+1}/{len(student_models)}...")
        
        # Create analyzer
        analyzer = RepresentationAnalyzer(student_model, speech_teacher, music_teacher)
        
        # Analyze representations
        model_results = analyzer.analyze_representations(speech_inputs, music_inputs)
        
        # Get model name from directory path
        model_path = Path(args.student_models[i])
        model_name = model_path.parent.name  # Get the directory name
        
        # Save results
        results[model_name] = model_results
        
        # Plot heatmap
        heatmap_path = os.path.join(args.output_dir, f"{model_name}_similarity_heatmap.png")
        analyzer.plot_similarity_heatmap(model_results, save_path=heatmap_path)
    
    # Create and save model comparison table
    comparison_path = os.path.join(args.output_dir, "model_comparison.png")
    plot_model_comparison(results, save_path=comparison_path)
    
    # Convert results to native Python types
    results_native = convert_tensor_to_native(results)
    
    # Save all results
    results_path = os.path.join(args.output_dir, "similarity_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_native, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
