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
from scipy import stats
from upstream.multi_distiller.model import MultiDistillerModel, MultiDistillerConfig
from torch.nn.utils.rnn import pad_sequence
from scipy.stats import ttest_ind
from sklearn.metrics.pairwise import cosine_similarity

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
            # Load audio
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
            
            # Remove batch dimension and add to list
            waveforms.append(waveform.squeeze(0))
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    return waveforms

def get_representations(model, inputs, layer_idx):
    """Extract representations from model"""
    with torch.no_grad():
        # Get device from model parameters
        device = next(model.parameters()).device
        
        # Create padding mask
        x_lens = torch.LongTensor([wave.shape[-1] for wave in inputs])
        x_pad_batch = pad_sequence(inputs, batch_first=True)
        x_pad_batch = x_pad_batch.to(device)  # Move input to device
        
        # Create initial padding mask
        pad_mask = torch.ones(x_pad_batch.shape[:2], dtype=torch.bool).to(device)
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_lens[idx]:] = 0
        
        # Get hidden states
        if isinstance(model, MultiDistillerModel):
            # For distilled model
            _, feat_final, _, _, layer_hidden = model(
                x_pad_batch, pad_mask=pad_mask, get_hidden=True, no_pred=True
            )
            # For distilled model, we only have 2 layers (0 and 1)
            if layer_idx == 0:
                layer_rep = feat_final
            else:
                layer_rep = layer_hidden[0]  # First layer of distilled model
        else:
            # For original model
            outputs = model(x_pad_batch)
            layer_rep = outputs.hidden_states[layer_idx]
        
        # Create padding mask for the layer representation
        # The sequence length should match the layer representation
        layer_pad_mask = torch.ones(layer_rep.shape[:2], dtype=torch.bool).to(device)
        for idx in range(layer_rep.shape[0]):
            # Calculate the ratio of sequence lengths
            ratio = layer_rep.shape[1] / x_lens[idx]
            # Scale the length accordingly
            scaled_len = int(x_lens[idx] * ratio)
            layer_pad_mask[idx, scaled_len:] = 0
        
        # Average over time dimension (sequence length)
        # Only consider non-padded positions
        mask_expanded = layer_pad_mask.unsqueeze(-1).float()
        masked_rep = layer_rep * mask_expanded
        sum_rep = masked_rep.sum(dim=1)
        count = mask_expanded.sum(dim=1)
        avg_rep = sum_rep / (count + 1e-8)  # Add small epsilon to avoid division by zero
        
        return avg_rep

def perform_ttest_analysis(student_model, teacher_model, audio_inputs, device='cuda'):
    """Perform t-test analysis between student and teacher representations"""
    results = {}
    
    # Analyze both layers
    for layer_idx in [0, 1]:
        student_reps = []
        teacher_reps = []
        
        for audio_input in audio_inputs:
            # Get representations
            student_rep = get_representations(student_model, [audio_input], layer_idx)
            teacher_rep = get_representations(teacher_model, [audio_input], layer_idx)
            
            # Flatten representations
            student_reps.append(student_rep.flatten())
            teacher_reps.append(teacher_rep.flatten())
        
        # Convert to numpy arrays
        student_reps = np.array(student_reps)
        teacher_reps = np.array(teacher_reps)
        
        # Perform t-test for each dimension
        t_stats = []
        p_values = []
        
        for i in range(student_reps.shape[1]):
            t_stat, p_val = stats.ttest_ind(student_reps[:, i], teacher_reps[:, i], equal_var=False)
            t_stats.append(t_stat)
            p_values.append(p_val)
        
        # Store results for this layer
        results[f'layer_{layer_idx}'] = {
            't_statistics': np.array(t_stats),
            'p_values': np.array(p_values),
            'mean_t_stat': np.mean(np.abs(t_stats)),
            'mean_p_value': np.mean(p_values),
            'significant_dims': np.sum(np.array(p_values) < 0.05)
        }
    
    return results

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and scalar types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

def plot_ttest_results(results, save_path=None):
    """Plot t-test results"""
    model_names = list(results.keys())
    combinations = [
        'Speech Teacher + Speech Audio',
        'Speech Teacher + Music Audio',
        'Music Teacher + Speech Audio',
        'Music Teacher + Music Audio'
    ]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('T-test Analysis of Model Differences', fontsize=16)
    
    for idx, combo in enumerate(combinations):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Prepare data
        t_stats = []
        p_values = []
        labels = []
        for model in model_names:
            # Get data for each layer
            for layer in range(2):  # Assuming 2 layers
                t_stats.append(results[model][combo][f'layer_{layer}']['mean_t_stat'])
                p_values.append(results[model][combo][f'layer_{layer}']['mean_p_value'])
                labels.append(f"{model}\nLayer {layer}")
        
        # Set up the bar positions
        x = np.arange(len(labels))
        width = 0.2  # Width of the bars
        
        # Create bar plots
        t_bars = ax.bar(x - width/2, t_stats, width, label='Mean |t-statistic|')
        p_bars = ax.bar(x + width/2, [p * 10 for p in p_values], width, label='Mean p-value (×10)')
        
        # Set ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels for t-statistics
        for bar in t_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        # Add value labels for p-values (original p-values)
        for i, p_val in enumerate(p_values):
            if p_val < 0.001:
                p_text = '<0.001'
            else:
                p_text = f'{p_val:.3f}'
            ax.text(x[i] + width/2, p_val * 10,
                   p_text,
                   ha='center', va='bottom', fontsize=8)
        
        # Add significance stars above p-value bars
        for i, p_val in enumerate(p_values):
            if p_val < 0.001:
                ax.text(x[i] + width/2, p_val * 10 + 0.2, '***',
                       ha='center', va='bottom', fontsize=10)
            elif p_val < 0.01:
                ax.text(x[i] + width/2, p_val * 10 + 0.2, '**',
                       ha='center', va='bottom', fontsize=10)
            elif p_val < 0.05:
                ax.text(x[i] + width/2, p_val * 10 + 0.2, '*',
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_title(combo)
        ax.set_ylabel('Value')
        
        # Add legend
        if idx == 0:  # Only add legend to first subplot
            ax.legend(loc='upper right')
            # Add significance level explanation
            ax.text(0.02, 0.98, 'Significance levels:\n*** p < 0.001\n** p < 0.01\n* p < 0.05',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare model representations using t-tests')
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
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to analyze from each dataset')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load teacher models
    print("Loading teacher models...")
    hubert_config_path = os.path.join(os.path.dirname(args.distilled_hubert_path), 'config_model.yaml')
    mert_config_path = os.path.join(os.path.dirname(args.distilled_mert_path), 'config_model.yaml')
    
    with open(hubert_config_path, 'r') as f:
        hubert_config = yaml.safe_load(f)
    with open(mert_config_path, 'r') as f:
        mert_config = yaml.safe_load(f)
    
    speech_teacher = MultiDistillerModel(MultiDistillerConfig(hubert_config["multi_distiller"]))
    music_teacher = MultiDistillerModel(MultiDistillerConfig(mert_config["multi_distiller"]))
    
    hubert_checkpoint = torch.load(args.distilled_hubert_path, map_location='cpu')
    mert_checkpoint = torch.load(args.distilled_mert_path, map_location='cpu')
    
    speech_teacher.load_state_dict(hubert_checkpoint["Distiller"])
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
        
        # Get model name from directory path
        model_path = Path(args.student_models[i])
        model_name = model_path.parent.name
        
        # Perform t-test analysis for each layer
        model_results = {
            'Speech Teacher + Speech Audio': perform_ttest_analysis(student_model, speech_teacher, speech_inputs),
            'Speech Teacher + Music Audio': perform_ttest_analysis(student_model, speech_teacher, music_inputs),
            'Music Teacher + Speech Audio': perform_ttest_analysis(student_model, music_teacher, speech_inputs),
            'Music Teacher + Music Audio': perform_ttest_analysis(student_model, music_teacher, music_inputs)
        }
        
        results[model_name] = model_results
    
    # Plot results
    comparison_path = os.path.join(args.output_dir, "ttest_comparison.png")
    plot_ttest_results(results, save_path=comparison_path)
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, "ttest_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
