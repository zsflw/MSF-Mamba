import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool as gmp
from mamba_ssm import Mamba

class RNAStructureCNN(nn.Module):

    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class PrecomputedRNAFeatureLoader:

    def __init__(self, feature_path):
        self.features = torch.load(feature_path)

    def get_features(self, rna_id):
        return self.features.get(rna_id, torch.zeros(22))

class RNAFeatureExtractor(nn.Module):

    def __init__(self, feature_path, output_dim=128):
        super().__init__()
        self.loader = PrecomputedRNAFeatureLoader(feature_path)
        self.cnn = RNAStructureCNN(output_dim=output_dim)

    def forward(self, rna_ids, device):
        batch_features = []
        for rna_id in rna_ids:
            features = self.loader.get_features(rna_id)
            batch_features.append(features)
        features_tensor = torch.stack(batch_features).to(device)
        return self.cnn(features_tensor)

class HybridMambaBlock(nn.Module):

    def __init__(self, embed_dim, d_state=32, d_conv=4, expand=4):
        super().__init__()
        self.mamba = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expand),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * expand, embed_dim)
        )

    def forward(self, x):

        mamba_out = self.mamba(self.norm1(x))


        attn_out, _ = self.attn(x, x, x)


        fused = self.proj(mamba_out + attn_out)
        return self.norm2(x + fused)

class MultiScaleMamba(nn.Module):

    def __init__(self, embed_dim, output_dim):
        super().__init__()

        self.path1 = nn.Sequential(
            HybridMambaBlock(embed_dim, d_state=16, expand=2),
            HybridMambaBlock(embed_dim, d_state=16, expand=2)
        )
        self.path2 = HybridMambaBlock(embed_dim, d_state=32, expand=4)


        self.local_conv = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )


        self.fusion = nn.Sequential(
            nn.Linear(embed_dim*2 + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):

        p1 = self.path1(x).max(dim=1)[0]

        p2 = self.path2(x).mean(dim=1)

        local = self.local_conv(x.permute(0,2,1)).squeeze(-1)

        return self.fusion(torch.cat([p1, p2, local], dim=-1))

class GraphAugmentor(nn.Module):

    def __init__(self, mask_ratio=0.1, edge_drop_ratio=0.2):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.edge_drop_ratio = edge_drop_ratio
        
    def forward(self, x, edge_index):

        if self.training and self.mask_ratio > 0:
            mask = torch.rand_like(x) > self.mask_ratio
            x = x * mask.float()

        if self.training and self.edge_drop_ratio > 0:
            mask = torch.rand(edge_index.size(1), device=x.device) > self.edge_drop_ratio
            edge_index = edge_index[:, mask]
            
        return x, edge_index

class DifferentialAttention(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat1, feat2):
        q = self.query(feat1)
        k = self.key(feat2)
        v = self.value(feat2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = self.softmax(attn_scores)

        diff_attn = self.gamma * (attn_weights - attn_weights.mean(dim=-1, keepdim=True))
        attended_features = torch.matmul(diff_attn, v)
        return attended_features + feat1

class DrugFeatureFusion(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()
        self.attn = DifferentialAttention(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU()
        )

    def forward(self, graph_feat, smile_feat):
        attended_graph = self.attn(graph_feat, smile_feat)
        attended_smile = self.attn(smile_feat, graph_feat)
        return self.fc(torch.cat([attended_graph, attended_smile], dim=1))

class RNAFeatureFusion(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()
        self.attn = DifferentialAttention(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU()
        )

    def forward(self, seq_feat, struct_feat):
        attended_seq = self.attn(seq_feat, struct_feat)
        attended_struct = self.attn(struct_feat, seq_feat)
        return self.fc(torch.cat([attended_seq, attended_struct], dim=1))

class CrossAttentionFusion(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()
        self.drug_proj = nn.Linear(128, embed_dim)
        self.rna_proj = nn.Linear(128, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU()
        )

    def forward(self, drug_feat, rna_feat):
        drug_proj = self.drug_proj(drug_feat).unsqueeze(0)
        rna_proj = self.rna_proj(rna_feat).unsqueeze(0)

        attn_output, _ = self.attention(
            query=drug_proj,
            key=rna_proj,
            value=rna_proj
        )
        return self.fc(attn_output.squeeze(0) + drug_proj.squeeze(0))

class GCNNetmuti(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                 num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2):
        super(GCNNetmuti, self).__init__()


        self.augmentor = GraphAugmentor(mask_ratio=0.1, edge_drop_ratio=0.2)


        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)
        self.smile_mamba = MultiScaleMamba(embed_dim, output_dim)


        self.gcnv1 = GATConv(num_features_xd, num_features_xd * 2)
        self.gcnv2 = GATConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)


        self.rna_struct_extractor = RNAFeatureExtractor(
            feature_path="/share/home/u2415283017/DLST/data/miRNA_sequences_features.pt",
            output_dim=output_dim
        )


        self.rna_embed = nn.Embedding(num_features_xt + 1, embed_dim)
        self.rna_mamba = MultiScaleMamba(embed_dim, output_dim)


        self.drug_fusion = DrugFeatureFusion(embed_dim=output_dim)
        self.rna_fusion = RNAFeatureFusion(embed_dim=output_dim)
        self.cross_fusion = CrossAttentionFusion(embed_dim=256)


        self.fc1 = nn.Linear(128, 256)
        self.out = nn.Linear(256, n_output)
        self.ac = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        current_device = next(self.parameters()).device
        x, edge_index, batch = data.x, data.edge_index, data.batch
        drugsmile = data.seqdrug
        target = data.target


        xt_struct_features = self.rna_struct_extractor(target, current_device)


        x, edge_index = self.augmentor(x, edge_index)


        x = self.gcnv1(x, edge_index)
        x = self.relu(x)
        x = self.gcnv2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)


        x = self.fc_g1(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)


        embedded_smile = self.smile_embed(drugsmile.long())
        conv_xd = self.smile_mamba(embedded_smile)


        x = self.drug_fusion(x, conv_xd)
        embedded_rna = self.rna_embed(target.long())
        xt_seq_features = self.rna_mamba(embedded_rna)
        rna_features = self.rna_fusion(xt_seq_features, xt_struct_features)
        xc = self.cross_fusion(x, rna_features)


        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.ac(self.out(xc))

        return out