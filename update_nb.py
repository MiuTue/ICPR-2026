import json
import os

file_path = r'c:\Coding Project\ICPR2026\baseline_icpr_2026\notebook\baseline.ipynb'
if not os.path.exists(file_path):
    print("Notebook not found")
    exit(1)

nb = json.load(open(file_path, encoding='utf-8'))
modified = 0

for cell in nb['cells']:
    source_str = "".join(cell['source'])
    
    # Update imports and backbone instantiation
    if 'mobilenet_v3_large' in source_str:
        cell['source'] = [l.replace('mobilenet_v3_large', 'swin_t').replace('MobileNet_V3_Large_Weights', 'Swin_T_Weights') for l in cell['source']]
        modified += 1
        
    # Update Model definition
    elif 'class MultiFrameCRNN' in source_str:
        cell['source'] = [
            'class MultiFrameCRNN(nn.Module):\n',
            '    def __init__(self, num_classes, d_model=512):\n',
            '        super().__init__()\n',
            '        # Backbone Swin Transformer (Tiny version for speed)\n',
            '        swin = swin_t(weights=Swin_T_Weights.DEFAULT)\n',
            '        self.backbone = swin.features # Output channels: 768\n',
            '\n',
            '        self.conv_proj = nn.Sequential(\n',
            '            nn.Upsample(scale_factor=(1, 4), mode=\'bilinear\', align_corners=True),\n',
            '            nn.Conv2d(768, d_model, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),\n',
            '            nn.BatchNorm2d(d_model),\n',
            '            nn.ReLU()\n',
            '        )\n',
            '        self.fusion = ChannelSpatialFusion(channels=d_model)\n',
            '        self.pos_encoder = PositionalEncoding(d_model=d_model)\n',
            '        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)\n',
            '        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)\n',
            '        self.fc = nn.Linear(d_model, num_classes)\n',
            '\n',
            '    def forward(self, x):\n',
            '        b, t, c, h, w = x.size()\n',
            '        x = x.view(b * t, c, h, w)\n',
            '        features = self.backbone(x) # [B*T, 1, 4, 768]\n',
            '        features = features.permute(0, 3, 1, 2) # [B*T, 768, 1, 4]\n',
            '        features = self.conv_proj(features)\n',
            '        fused = self.fusion(features, t)\n',
            '        fused = F.adaptive_avg_pool2d(fused, (1, None))\n',
            '        seq = fused.squeeze(2).permute(0, 2, 1)\n',
            '        seq = self.pos_encoder(seq)\n',
            '        out = self.transformer(seq)\n',
            '        out = self.fc(out)\n',
            '        return out.log_softmax(2)\n'
        ]
        modified += 1

print(f"Modified {modified} cells")
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
