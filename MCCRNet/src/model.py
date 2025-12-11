import torch
import torch.nn as nn
import torch.nn.functional as F 


from torchvision.models import resnet18
from modules.encoders import get_encoder, SubNet
from resnet import MyResNet, BasicBlock


# Define the Attention Fusion Module
class FeatureAttention(nn.Module):
    def __init__(self, d_model, hidden_dim):
        """
        A simple attention mechanism applied to the concatenated feature vector.
        It learns attention weights for each feature dimension.
        Args:
            d_model (int): Input and output feature dimension size (d_vout + d_aout).
            hidden_dim (int): Hidden dimension size for the attention scoring network.
        """
        super().__init__()
        self.attention_scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(), # Or nn.ReLU()
            nn.Linear(hidden_dim, d_model) # Output attention scores for each feature dimension
        )
        self.softmax = nn.Softmax(dim=-1) # Apply softmax across the feature dimension

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_model).
        Returns:
            torch.Tensor: Attended output tensor of shape (batch_size, d_model).
        """
        # Compute attention scores
        scores = self.attention_scorer(x) # (batch_size, d_model)

        # Compute attention weights using softmax
        attn_weights = self.softmax(scores) # (batch_size, d_model)

        # Apply attention weights element-wise to the input features
        attended_output = x * attn_weights # (batch_size, d_model)

        return attended_output


class MCCRNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        print('hp:', hp)
        self.hp = hp
        self.add_va = hp.add_va

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ResNet for original visual feature extraction
        self.frame_encoder = MyResNet(BasicBlock, [2, 2, 2, 2]).eval()
        self.load_vision_weight() # This method requires torchvision.models.resnet18
        self.frame_encoder = self.frame_encoder.to(self.device)


        print("--- Inside MCCRNet init ---")
        print(f"hp.d_vh = {getattr(hp, 'd_vh', 'Not Found')}")
        print(f"hp.d_ah = {getattr(hp, 'd_ah', 'Not Found')}")
        print(f"Using hidden size for visual: {hp.d_vh}")
        print(f"Using hidden size for audio: {hp.d_ah}")
        print("------------------------")

 
        self.visual_enc = get_encoder(
            encoder_type=hp.visual_encoder_type,
            in_size=hp.d_vin + hp.d_pose,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        ).to(self.device)

        
        self.audio_enc = get_encoder(
            encoder_type=hp.acoustic_encoder_type,
            in_size=hp.d_audio_in + hp.d_is09_in,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        ).to(self.device)
        
        # --- Fusion Module Control Flags ---
        # NEW: Flag for temporal fusion (Fuse-then-Pool)
        self.use_temporal_fusion = getattr(hp, 'use_temporal_fusion', False)
        # Original flags for pooled fusion (Pool-then-Fuse)
        self.use_transformer_fusion = getattr(hp, 'use_transformer_fusion', False)
        self.use_attention_fusion = getattr(hp, 'use_attention_fusion', False)

        # Ensure only one fusion strategy is active at a time
        num_fusion_strategies = sum([
            self.use_temporal_fusion,
            self.use_transformer_fusion,
            self.use_attention_fusion
        ])
        if num_fusion_strategies > 1:
            raise ValueError("Only one fusion strategy can be enabled at a time (temporal, transformer, or attention).")

        # Initialize all fusion modules to None
        self.temporal_fusion_transformer = None
        self.transformer_fusion = None
        self.attention_fusion = None
        
        fusion_input_size = hp.d_vout + hp.d_aout

        # --- Instantiate the selected fusion module ---
        if self.use_temporal_fusion:
            print("Using TEMPORAL Transformer Fusion (Fuse-then-Pool)")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=fusion_input_size,
                nhead=getattr(hp, 'transformer_nhead', 4),
                dim_feedforward=hp.d_prjh,
                dropout=hp.dropout_prj,
                activation='relu',
                batch_first=True
            )
            transformer_layers = getattr(hp, 'temporal_fusion_layers', 2) # Default to 2 layers for deeper fusion
            self.temporal_fusion_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_layers
            ).to(self.device)

        elif self.use_transformer_fusion:
            print("Using POOLED Transformer Fusion (Pool-then-Fuse)")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=fusion_input_size,
                nhead=getattr(hp, 'transformer_nhead', 4),
                dim_feedforward=hp.d_prjh,
                dropout=hp.dropout_prj,
                activation='relu',
                batch_first=True
            )
            transformer_layers = getattr(hp, 'transformer_fusion_layers', 1)
            self.transformer_fusion = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_layers
            ).to(self.device)

        elif self.use_attention_fusion:
            print("Using POOLED Attention Fusion (Pool-then-Fuse)")
            self.attention_fusion = FeatureAttention(
                d_model=fusion_input_size,
                hidden_dim=hp.d_prjh
            ).to(self.device)
        
        # --- Final Prediction Layer (No change needed!) ---
        # The input size is consistent across all fusion strategies after pooling.
        self.fusion_prj = SubNet(
            in_size=fusion_input_size,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        ).to(self.device)
      
    def load_vision_weight(self):
        try:
            resnet = resnet18(weights='IMAGENET1K_V1')
        except Exception:
            try:
                 resnet = resnet18(pretrained=True)
            except Exception as e2:
                 print(f"Error: Could not load pretrained ResNet18 weights: {e2}")
                 return
        
        pretrained_dict = resnet.state_dict()
        model_dict = self.frame_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        if not pretrained_dict:
            print("Warning: No matching keys found for ResNet weights.")
        else:
            model_dict.update(pretrained_dict)
            self.frame_encoder.load_state_dict(model_dict)
            print(f"Successfully loaded {len(pretrained_dict)} matching keys into frame encoder.")

        for p in self.frame_encoder.parameters():
            p.requires_grad = False

    def _pad_or_truncate(self, features, target_dim):
         current_dim = features.size(-1)
         if current_dim < target_dim:
             padding_size = target_dim - current_dim
             features = F.pad(features, (0, padding_size))
         elif current_dim > target_dim:
             features = features[..., :target_dim]
         return features


    def forward(self, visual, pose_features, audio_features, is09_features, y=None):
        device = self.device
        visual = visual.to(device)
        pose_features = pose_features.to(device)
        audio_features = audio_features.to(device)
        is09_features = is09_features.to(device)

        batch_size, seq_len, channels, height, width = visual.shape

        visual_features = []
        for t in range(seq_len):
            frame = visual[:, t]
            feat = self.frame_encoder(frame).view(batch_size, -1)
            visual_features.append(feat)
        visual_features = torch.stack(visual_features, dim=1)

        pose_features = pose_features.view(batch_size, seq_len, -1)
        pose_features_processed = self._pad_or_truncate(pose_features, self.hp.d_pose)
        combined_visual = torch.cat((visual_features, pose_features_processed), dim=-1)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(device)
        visual_encoded_output = self.visual_enc(combined_visual, lengths)

        if audio_features.dim() == 4:
            audio_features = audio_features.squeeze(-1) if audio_features.shape[-1] == 1 else audio_features.squeeze(-2)
        if audio_features.dim() != 3:
            raise ValueError(f"Audio features must be 3D, but got {audio_features.dim()} dims. Shape: {audio_features.shape}")

        current_seq_len = audio_features.size(1)
        is09_features_processed = is09_features
        if is09_features.dim() == 2:
            is09_features_processed = is09_features.unsqueeze(1).expand(-1, current_seq_len, -1)
        elif is09_features.dim() == 3 and is09_features.size(1) != current_seq_len:
            is09_features_processed = F.interpolate(is09_features.permute(0, 2, 1), size=current_seq_len, mode='nearest').permute(0, 2, 1)

        combined_audio = torch.cat((audio_features, is09_features_processed), dim=-1)
        combined_audio_processed = self._pad_or_truncate(combined_audio, self.hp.d_audio_in + self.hp.d_is09_in)
        audio_lengths = torch.full((batch_size,), current_seq_len, dtype=torch.long).to(device)
        audio_encoded_output = self.audio_enc(combined_audio_processed, audio_lengths)

  
        if self.temporal_fusion_transformer is not None:
            # --- PATH 1: Temporal Fusion (Fuse-then-Pool) ---
            # 1. Fuse: Concatenate sequences along the feature dimension
            concatenated_sequence = torch.cat((visual_encoded_output, audio_encoded_output), dim=2)
            # 2. Deep Fuse: Process the combined sequence with a Transformer
            fused_sequence = self.temporal_fusion_transformer(concatenated_sequence)
            # 3. Pool: Pool the fused sequence to get a final vector
            fused_features = fused_sequence.mean(dim=1)
        else:
            # --- PATH 2: Pooled Fusion (Pool-then-Fuse) - Original Logic ---
            # 1. Pool: Get a single vector for each modality
            visual_pooled = visual_encoded_output.mean(dim=1) if visual_encoded_output.ndim == 3 else visual_encoded_output
            audio_pooled = audio_encoded_output.mean(dim=1) if audio_encoded_output.ndim == 3 else audio_encoded_output
            # 2. Fuse: Concatenate the pooled vectors
            fusion_input = torch.cat((visual_pooled, audio_pooled), dim=1)
            # 3. Apply selected fusion method on the pooled vector
            fused_features = fusion_input  # Default to simple concatenation
            if self.transformer_fusion is not None:
                fused_features = self.transformer_fusion(fusion_input.unsqueeze(1)).squeeze(1)
            elif self.attention_fusion is not None:
                fused_features = self.attention_fusion(fusion_input)
  
        # --- Final Prediction (No change needed!) ---
        output = self.fusion_prj(fused_features)
        
        return output

    # The extract_features method is complex and needs a clear decision on what it should return.
    # For now, I'm keeping the original logic. If you need temporal fusion here, 
    # the return signature might need to change (e.g., return fused_sequence).
    def extract_features(self, visual, pose_features, audio_features, is09_features):
        device = self.device
        visual = visual.to(device)
        pose_features = pose_features.to(device)
        audio_features = audio_features.to(device)
        is09_features = is09_features.to(device)

        batch_size, seq_len, channels, height, width = visual.shape

        visual_features = []
        for t in range(seq_len):
            frame = visual[:, t]
            feat = self.frame_encoder(frame).view(batch_size, -1)
            visual_features.append(feat)
        visual_features = torch.stack(visual_features, dim=1)

        pose_features = pose_features.view(batch_size, seq_len, -1)
        pose_features_processed = self._pad_or_truncate(pose_features, self.hp.d_pose)
        combined_visual = torch.cat((visual_features, pose_features_processed), dim=-1)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(device)
        visual_encoded_output = self.visual_enc(combined_visual, lengths)

        if audio_features.dim() == 4:
            audio_features = audio_features.squeeze(-1) if audio_features.shape[-1] == 1 else audio_features.squeeze(-2)
        if audio_features.dim() != 3:
            raise ValueError(f"Audio features must be 3D, but got {audio_features.dim()} dims. Shape: {audio_features.shape}")

        current_seq_len = audio_features.size(1)
        is09_features_processed = is09_features
        if is09_features.dim() == 2:
            is09_features_processed = is09_features.unsqueeze(1).expand(-1, current_seq_len, -1)
        elif is09_features.dim() == 3 and is09_features.size(1) != current_seq_len:
            is09_features_processed = F.interpolate(is09_features.permute(0, 2, 1), size=current_seq_len, mode='nearest').permute(0, 2, 1)

        combined_audio = torch.cat((audio_features, is09_features_processed), dim=-1)
        combined_audio_processed = self._pad_or_truncate(combined_audio, self.hp.d_audio_in + self.hp.d_is09_in)
        audio_lengths = torch.full((batch_size,), current_seq_len, dtype=torch.long).to(device)
        audio_encoded_output = self.audio_enc(combined_audio_processed, audio_lengths)

        # NOTE: extract_features keeps the original "pool-then-fuse" logic.
        # This is a design choice. If you need the fused sequence from the temporal fusion path,
        # you would need to adapt the logic and return values of this method.
        visual_pooled = visual_encoded_output.mean(dim=1) if visual_encoded_output.ndim == 3 else visual_encoded_output
        audio_pooled = audio_encoded_output.mean(dim=1) if audio_encoded_output.ndim == 3 else audio_encoded_output
        fusion_input = torch.cat((visual_pooled, audio_pooled), dim=1)
        
        fused_features = fusion_input
        if self.transformer_fusion is not None:
            fused_features = self.transformer_fusion(fusion_input.unsqueeze(1)).squeeze(1)
        elif self.attention_fusion is not None:
             fused_features = self.attention_fusion(fusion_input)

        return visual_pooled, audio_pooled, fused_features