"""Single-stream early-fusion classifier for Fakeddit.

Image grid tokens (CNN) and title tokens (small BERT-style encoder) are concatenated into ONE
sequence and processed by a joint Transformer, so the modalities interact from the first layer on:

    [CLS] img_1..img_49 [SEP | NO_TEXT] txt_1..txt_L  ->  TransformerEncoder  ->  CLS  ->  head

Image-only samples (empty title, or text dropped) mask every text token and use a learned
[NO_TEXT] separator, so the prediction depends on the image alone.

Defaults (ResNet-18 + MiniLM-L6, small fusion Transformer) are sized for free-tier Colab (T4).
"""
import torch
import torch.nn as nn
import torchvision
from transformers import AutoModel

N_IMG_TOKENS = 49  # 7x7 grid at 224x224 (all backbones below have stride 32)

DEFAULT_IMAGE_BACKBONE = "resnet18"
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# name -> (builder returning an nn.Sequential producing [B, C, 7, 7], C)
IMAGE_BACKBONES = {
    "resnet18": (lambda: nn.Sequential(*list(torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).children())[:-2]), 512),
    "resnet50": (lambda: nn.Sequential(*list(torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).children())[:-2]), 2048),
    "efficientnet_b0": (lambda: torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1).features, 1280),
    "mobilenet_v3_large": (lambda: torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).features, 960),
}


def text_layers(text_model):
    """Transformer blocks of a HF encoder (BERT/MiniLM/RoBERTa or DistilBERT layout)."""
    if hasattr(text_model, "encoder"):
        return text_model.encoder.layer
    return text_model.transformer.layer


class EarlyFusionClassifier(nn.Module):
    def __init__(self, n_classes, d_model=256, n_layers=2, n_heads=4, max_text_len=32,
                 dropout=0.1, image_backbone=DEFAULT_IMAGE_BACKBONE, text_model=DEFAULT_TEXT_MODEL,
                 txt_dim=None, load_backbones=True, unfreeze_top=False):
        super().__init__()
        self.unfreeze_top = unfreeze_top
        build_cnn, self.img_dim = IMAGE_BACKBONES[image_backbone]
        self.cnn = self.bert = None
        if load_backbones:
            self.cnn = build_cnn()
            self.bert = AutoModel.from_pretrained(text_model)
            txt_dim = self.bert.config.hidden_size
            for p in self.backbone_parameters():
                p.requires_grad = False
            if unfreeze_top:
                for p in self.cnn[-1].parameters():  # last CNN stage
                    p.requires_grad = True
                for layer in text_layers(self.bert)[-2:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        if txt_dim is None:
            raise ValueError("txt_dim is required when backbones are not loaded")
        self.txt_dim = txt_dim

        self.img_proj = nn.Sequential(nn.Linear(self.img_dim, d_model), nn.LayerNorm(d_model))
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, d_model), nn.LayerNorm(d_model))

        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.no_text_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_emb = nn.Embedding(3, d_model)  # 0 special, 1 image, 2 text
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + N_IMG_TOKENS + 1 + max_text_len, d_model))
        for t in (self.cls_tok, self.sep_tok, self.no_text_tok, self.pos_emb):
            nn.init.trunc_normal_(t, std=0.02)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def backbone_parameters(self):
        for m in (self.cnn, self.bert):
            if m is not None:
                yield from m.parameters()

    def train(self, mode=True):
        super().train(mode)
        # Keep BatchNorm statistics frozen, and the text encoder deterministic unless fine-tuned.
        if self.cnn is not None:
            self.cnn.eval()
        if self.bert is not None and not self.unfreeze_top:
            self.bert.eval()
        return self

    def encode_image(self, pixels):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.unfreeze_top):
            grid = self.cnn(pixels)                    # [B, C, 7, 7]
        return grid.flatten(2).transpose(1, 2)         # [B, 49, C]

    def encode_text(self, input_ids, attention_mask):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.unfreeze_top):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def fuse(self, img_feats, txt_feats, txt_mask, has_text):
        """Early fusion over backbone token features.

        img_feats [B, 49, img_dim], txt_feats [B, L, txt_dim], txt_mask [B, L] (1 = real token),
        has_text [B] bool (False -> image-only prediction).
        """
        B = img_feats.size(0)
        img = self.img_proj(img_feats.float()) + self.type_emb.weight[1]
        txt = self.txt_proj(txt_feats.float()) + self.type_emb.weight[2]
        cls = self.cls_tok.expand(B, -1, -1) + self.type_emb.weight[0]
        sep = torch.where(has_text[:, None, None], self.sep_tok, self.no_text_tok) + self.type_emb.weight[0]

        x = torch.cat([cls, img, sep, txt], dim=1)
        x = x + self.pos_emb[:, :x.size(1)]

        txt_pad = ~(txt_mask.bool() & has_text[:, None])
        keep = torch.zeros(B, 2 + img.size(1), dtype=torch.bool, device=x.device)
        x = self.fusion(x, src_key_padding_mask=torch.cat([keep, txt_pad], dim=1))
        return self.head(self.norm(x[:, 0]))

    def forward(self, pixels, input_ids, attention_mask, has_text):
        return self.fuse(self.encode_image(pixels), self.encode_text(input_ids, attention_mask),
                         attention_mask, has_text)

    def trainable_state_dict(self):
        """State dict without frozen pretrained backbones (they are reloaded from torchvision/HF)."""
        sd = self.state_dict()
        if self.unfreeze_top:
            return sd
        return {k: v for k, v in sd.items() if not k.startswith(("cnn.", "bert."))}
