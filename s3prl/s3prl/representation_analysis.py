import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from torch.nn.utils.rnn import pad_sequence
from upstream.multi_distiller.model import MultiDistillerModel, MultiDistillerConfig

class RepresentationAnalyzer:
    def __init__(self, student_model, speech_teacher, music_teacher):
        """
        Args:
            student_model: The distilled student model (MultiDistillerModel)
            speech_teacher: The HuBERT model
            music_teacher: The MERT model
        """
        self.student_model = student_model
        self.speech_teacher = speech_teacher
        self.music_teacher = music_teacher
        
        # Define which layers to analyze
        self.student_layers = [0, 1]  # Student layers (0 is final layer, 1 is first layer)
        self.teacher_layers = [4, 8, 12]  # Teacher layers to analyze
        
        # Move models to device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.student_model = self.student_model.to(self.device)
        self.speech_teacher = self.speech_teacher.to(self.device)
        self.music_teacher = self.music_teacher.to(self.device)
        
        # Set models to eval mode
        self.student_model.eval()
        self.speech_teacher.eval()
        self.music_teacher.eval()
        
    def get_student_representations(self, inputs, layer_idx):
        """Extract representations from student model"""
        with torch.no_grad():
            # Create padding mask
            x_lens = torch.LongTensor([wave.shape[-1] for wave in inputs])
            x_pad_batch = pad_sequence(inputs, batch_first=True)
            x_pad_batch = x_pad_batch.to(self.device)  # Move input to device
            
            # Create initial padding mask
            pad_mask = torch.ones(x_pad_batch.shape[:2], dtype=torch.bool).to(self.device)
            for idx in range(x_pad_batch.shape[0]):
                pad_mask[idx, x_lens[idx]:] = 0
            
            # Get hidden states
            _, feat_final, _, _, layer_hidden = self.student_model(
                x_pad_batch, pad_mask=pad_mask, get_hidden=True, no_pred=True
            )
            
            # Combine final features and layer hidden states
            all_hidden = [feat_final] + layer_hidden
            
            # Get the requested layer representation
            layer_rep = all_hidden[layer_idx]
            
            # Create padding mask for the layer representation
            # The sequence length should match the layer representation
            layer_pad_mask = torch.ones(layer_rep.shape[:2], dtype=torch.bool).to(self.device)
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
    
    def get_hubert_representations(self, inputs, layer_idx):
        """Extract representations from HuBERT model"""
        with torch.no_grad():
            # Create padding mask
            x_lens = torch.LongTensor([wave.shape[-1] for wave in inputs])
            x_pad_batch = pad_sequence(inputs, batch_first=True)
            x_pad_batch = x_pad_batch.to(self.device)  # Move input to device
            
            # Create initial padding mask
            pad_mask = torch.ones(x_pad_batch.shape[:2], dtype=torch.bool).to(self.device)
            for idx in range(x_pad_batch.shape[0]):
                pad_mask[idx, x_lens[idx]:] = 0
            
            # Get hidden states
            if isinstance(self.speech_teacher, MultiDistillerModel):
                # For distilled model
                _, feat_final, _, _, layer_hidden = self.speech_teacher(
                    x_pad_batch, pad_mask=pad_mask, get_hidden=True, no_pred=True
                )
                # For distilled model, we only have 2 layers (0 and 1)
                if layer_idx == 0:
                    layer_rep = feat_final
                else:
                    layer_rep = layer_hidden[0]  # First layer of distilled model
            else:
                # For original HuBERT model
                outputs = self.speech_teacher(x_pad_batch)
                layer_rep = outputs.hidden_states[layer_idx]
            
            # Create padding mask for the layer representation
            # The sequence length should match the layer representation
            layer_pad_mask = torch.ones(layer_rep.shape[:2], dtype=torch.bool).to(self.device)
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
    
    def get_mert_representations(self, inputs, layer_idx):
        """Extract representations from MERT model"""
        with torch.no_grad():
            # Create padding mask
            x_lens = torch.LongTensor([wave.shape[-1] for wave in inputs])
            x_pad_batch = pad_sequence(inputs, batch_first=True)
            x_pad_batch = x_pad_batch.to(self.device)  # Move input to device
            
            # Create initial padding mask
            pad_mask = torch.ones(x_pad_batch.shape[:2], dtype=torch.bool).to(self.device)
            for idx in range(x_pad_batch.shape[0]):
                pad_mask[idx, x_lens[idx]:] = 0
            
            # Get hidden states
            if isinstance(self.music_teacher, MultiDistillerModel):
                # For distilled model
                _, feat_final, _, _, layer_hidden = self.music_teacher(
                    x_pad_batch, pad_mask=pad_mask, get_hidden=True, no_pred=True
                )
                # For distilled model, we only have 2 layers (0 and 1)
                if layer_idx == 0:
                    layer_rep = feat_final
                else:
                    layer_rep = layer_hidden[0]  # First layer of distilled model
            else:
                # For original MERT model
                outputs = self.music_teacher(x_pad_batch)
                layer_rep = outputs.hidden_states[layer_idx]
            
            # Create padding mask for the layer representation
            # The sequence length should match the layer representation
            layer_pad_mask = torch.ones(layer_rep.shape[:2], dtype=torch.bool).to(self.device)
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
    
    def compute_cosine_similarity(self, rep1, rep2):
        """Compute cosine similarity between two representations"""
        # Flatten representations if needed
        rep1 = rep1.view(rep1.size(0), -1)
        rep2 = rep2.view(rep2.size(0), -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(rep1.cpu().numpy(), rep2.cpu().numpy())
        return similarity
    
    def compute_cka(self, rep1, rep2):
        """Compute Centered Kernel Alignment (CKA) between two representations"""
        # Center the representations
        rep1 = rep1 - rep1.mean(dim=0, keepdim=True)
        rep2 = rep2 - rep2.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        gram1 = torch.mm(rep1, rep1.t())
        gram2 = torch.mm(rep2, rep2.t())
        
        # Compute CKA
        hsic = torch.trace(torch.mm(gram1, gram2))
        var1 = torch.sqrt(torch.trace(torch.mm(gram1, gram1)))
        var2 = torch.sqrt(torch.trace(torch.mm(gram2, gram2)))
        
        return hsic / (var1 * var2)
    
    def analyze_representations(self, speech_inputs, music_inputs, batch_size=8):
        """Analyze representation similarities between student and teacher models"""
        results = {
            'speech': {'layer_0': {}, 'layer_1': {}},
            'music': {'layer_0': {}, 'layer_1': {}}
        }
        
        # Determine teacher layers based on model type
        if isinstance(self.speech_teacher, MultiDistillerModel):
            teacher_layers = [0, 1]  # For distilled models
        else:
            teacher_layers = [4, 8, 12]  # For original models
        
        # Process speech inputs in batches
        print("Processing speech inputs...")
        speech_reps = {
            'student': {'layer_0': [], 'layer_1': []},
            'hubert': {f'layer_{l}': [] for l in teacher_layers},
            'mert': {f'layer_{l}': [] for l in teacher_layers}
        }
        
        for i in range(0, len(speech_inputs), batch_size):
            batch = speech_inputs[i:i + batch_size]
            
            # Get student representations
            for layer in [0, 1]:
                reps = self.get_student_representations(batch, layer)
                speech_reps['student'][f'layer_{layer}'].append(reps)
            
            # Get HuBERT representations
            for layer in teacher_layers:
                reps = self.get_hubert_representations(batch, layer)
                speech_reps['hubert'][f'layer_{layer}'].append(reps)
            
            # Get MERT representations
            for layer in teacher_layers:
                reps = self.get_mert_representations(batch, layer)
                speech_reps['mert'][f'layer_{layer}'].append(reps)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        # Concatenate batches
        for model in speech_reps:
            for layer in speech_reps[model]:
                speech_reps[model][layer] = torch.cat(speech_reps[model][layer], dim=0)
        
        # Process music inputs in batches
        print("Processing music inputs...")
        music_reps = {
            'student': {'layer_0': [], 'layer_1': []},
            'hubert': {f'layer_{l}': [] for l in teacher_layers},
            'mert': {f'layer_{l}': [] for l in teacher_layers}
        }
        
        for i in range(0, len(music_inputs), batch_size):
            batch = music_inputs[i:i + batch_size]
            
            # Get student representations
            for layer in [0, 1]:
                reps = self.get_student_representations(batch, layer)
                music_reps['student'][f'layer_{layer}'].append(reps)
            
            # Get HuBERT representations
            for layer in teacher_layers:
                reps = self.get_hubert_representations(batch, layer)
                music_reps['hubert'][f'layer_{layer}'].append(reps)
            
            # Get MERT representations
            for layer in teacher_layers:
                reps = self.get_mert_representations(batch, layer)
                music_reps['mert'][f'layer_{layer}'].append(reps)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        # Concatenate batches
        for model in music_reps:
            for layer in music_reps[model]:
                music_reps[model][layer] = torch.cat(music_reps[model][layer], dim=0)
        
        # Calculate similarities
        print("Calculating similarities...")
        for domain in ['speech', 'music']:
            reps = speech_reps if domain == 'speech' else music_reps
            
            for student_layer in [0, 1]:
                student_rep = reps['student'][f'layer_{student_layer}']
                
                # Compare with HuBERT
                for teacher_layer in teacher_layers:
                    hubert_rep = reps['hubert'][f'layer_{teacher_layer}']
                    similarity = self.compute_cka(student_rep, hubert_rep)
                    results[domain][f'layer_{student_layer}'][f'hubert_layer_{teacher_layer}'] = similarity
                
                # Compare with MERT
                for teacher_layer in teacher_layers:
                    mert_rep = reps['mert'][f'layer_{teacher_layer}']
                    similarity = self.compute_cka(student_rep, mert_rep)
                    results[domain][f'layer_{student_layer}'][f'mert_layer_{teacher_layer}'] = similarity
        
        return results
    
    def plot_similarity_heatmap(self, results, save_path=None):
        """Plot similarity heatmaps for speech and music inputs"""
        # Determine teacher layers based on model type
        if isinstance(self.speech_teacher, MultiDistillerModel):
            teacher_layers = [0, 1]  # For distilled models
            layer_labels = [f'Layer {l}' for l in teacher_layers]
        else:
            teacher_layers = [4, 8, 12]  # For original models
            layer_labels = [f'Layer {l}' for l in teacher_layers]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Representation Similarity Analysis', fontsize=16)
        
        # Plot speech input similarities
        for i, student_layer in enumerate(self.student_layers):
            # Convert tensors to CPU and then to numpy
            hubert_data = [
                results['speech'][f'layer_{student_layer}'][f'hubert_layer_{l}'].cpu().numpy()
                for l in teacher_layers
            ]
            mert_data = [
                results['speech'][f'layer_{student_layer}'][f'mert_layer_{l}'].cpu().numpy()
                for l in teacher_layers
            ]
            data = np.array([hubert_data, mert_data])
            
            sns.heatmap(data, 
                        ax=axes[0, i],
                        xticklabels=layer_labels,
                        yticklabels=['HuBERT', 'MERT'],
                        cmap='YlOrRd',
                        annot=True,
                        fmt='.3f',
                        vmin=0,
                        vmax=1)
            axes[0, i].set_title(f'Speech Input - Student Layer {student_layer}')
        
        # Plot music input similarities
        for i, student_layer in enumerate(self.student_layers):
            # Convert tensors to CPU and then to numpy
            hubert_data = [
                results['music'][f'layer_{student_layer}'][f'hubert_layer_{l}'].cpu().numpy()
                for l in teacher_layers
            ]
            mert_data = [
                results['music'][f'layer_{student_layer}'][f'mert_layer_{l}'].cpu().numpy()
                for l in teacher_layers
            ]
            data = np.array([hubert_data, mert_data])
            
            sns.heatmap(data, 
                        ax=axes[1, i],
                        xticklabels=layer_labels,
                        yticklabels=['HuBERT', 'MERT'],
                        cmap='YlOrRd',
                        annot=True,
                        fmt='.3f',
                        vmin=0,
                        vmax=1)
            axes[1, i].set_title(f'Music Input - Student Layer {student_layer}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 
