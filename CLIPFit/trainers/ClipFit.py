import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import random
import sys
#from FedAvg import  FedAvg
# from torchvision.transforms import v2

sys.path.append("..")

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    # "EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        self.dtype = dtype
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features = text_features
            self.clip_tokenized_prompt = prompts_
            self.clip_prompt = clip_model_.token_embedding(prompts_).type(dtype)


    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.clip_prompt
        tokenized_prompts = self.clip_tokenized_prompt
        text_features = self.text_encoder(prompts, tokenized_prompts)

        return  image_features, text_features

    def forward_similarity(self):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)

        prompts = self.clip_prompt
        tokenized_prompts = self.clip_tokenized_prompt
        text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features_old = self.text_features

        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)
        return score


def similarity(text_features, text_features_old):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)

    score = cos(text_features, text_features_old)
    score = 1.0 - torch.mean(score)
    return score

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.freeze_image_encoder = clip_model.visual

    def forward(self, image):
        image_features, text_features = self.prompt_learner(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        """if test:
            text_features_old = self.prompt_learner.text_features
            #text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
            text_features = 0.1*text_features_old + 0.9*text_features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:"""
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp_Fit(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None,scale=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        """
        # calculate the changes in each layer
        with torch.no_grad():
            for name, param in self.model.prompt_learner.text_encoder.named_parameters():
                if "c_proj.bias" in name:
                    print(name,((param.grad)**2).sum())
            for name, param in self.model.prompt_learner.image_encoder.named_parameters():
                if 'ln' in name:
                    print(name,((param.grad)**2).sum())"""
        """
        # calculate the gradient in each layer
        if self.grad==None:
            self.grad = {}
            with torch.no_grad():
                for name, param in self.model.prompt_learner.text_encoder.named_parameters():
                    if "c_proj.bias" in name:
                        self.grad[name] = copy.deepcopy(param.grad)
        else:
            with torch.no_grad():
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                for name, param in self.model.prompt_learner.text_encoder.named_parameters():
                    if "c_proj.bias" in name:
                        print(cos(self.grad[name].reshape(1,-1),copy.deepcopy(param.grad).reshape(1,-1)),((copy.deepcopy(param.grad))**2).mean())
                        self.grad[name] = copy.deepcopy(param.grad)"""
        self.model_update(names)


    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(len(classnames), classnames)
        ##cutmix = v2.CutMix(num_classes=len(classnames))
        # mixup = v2.MixUp(num_classes=len(classnames))
        # self.mixup =  mixup
        # self.cutmix = cutmix

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        w_ln = {}
        w_ln_bias = {}
        w_ln_weight = {}
        w_ln_o = {}
        self.grad = None
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        l = 0

        # unlock bias terms and LN terms.
        for name, param in self.model.prompt_learner.text_encoder.named_parameters():
            if "c_proj.bias" in name:
                param.requires_grad_(True)

        for name, param in self.model.prompt_learner.image_encoder.named_parameters():
            if 'ln' in name:
                param.requires_grad = True
                w_ln[name] = copy.deepcopy(param)


        """
        # analyse changes in parameters
        for name, param in self.model.prompt_learner.text_encoder.named_parameters():
            if "c_proj.bias" in name:
                param.requires_grad_(True)
                with torch.no_grad():
                    w_ln[name] = copy.deepcopy(param)
        for name, param in self.model.prompt_learner.image_encoder.named_parameters():
            if 'ln' in name:
                param.requires_grad = True
                w_ln_bias[name] = copy.deepcopy(param)
                w_ln_o[name] = copy.deepcopy(param)

            if 'ln' in name and 'weight' in name:
                param.requires_grad = True
                w_ln_weight[name] = copy.deepcopy(param)
                w_ln_o[name] = copy.deepcopy(param)


        self.w_bias = w_ln
        self.w_ln_bias = w_ln_bias
        self.w_ln_weight = w_ln_weight
        self.w_ln = w_ln_o
        self.change_store = [[],[],[]]"""


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        """if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)"""

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            # kl_loss = nn.KLDivLoss(reduction="batchmean")
            loss_bias = []
            loss_ln_bias = []
            loss_ln_weight = []
            """
            # L2 loss for less change in each layer
            for name, param in self.model.prompt_learner.text_encoder.named_parameters():
                if "c_proj.bias" in name:
                    loss_bias.append(((self.w_bias[name].cuda()-param)**2).sum())

            for name, param in self.model.prompt_learner.image_encoder.named_parameters():
                if 'ln' in name and 'bias' in name:
                    loss_ln_bias.append(((self.w_ln_bias[name].cuda() - param) ** 2).sum())

                if 'ln' in name and 'weight' in name:
                    loss_ln_weight.append(((self.w_ln_weight[name].cuda() - param) ** 2).sum())

            self.change_store[0].append(loss_bias)
            self.change_store[1].append(loss_ln_weight)
            self.change_store[2].append(loss_ln_bias)"""
            with torch.no_grad():
                pass
                """scale = copy.deepcopy(torch.tensor([t.cpu().item() for t in loss_bias]))
                scale = scale + 0.00001
                scale = scale / torch.max(scale)
                scale = 1 / scale"""
                #print([t.cpu().item() for t in loss_bias])
                #print(scale)
                """print('/n')
                print([t.cpu().item() for t in loss_ln_bias])
                print('/n')
                print([t.cpu().item() for t in loss_ln_weight])"""
            #loss_bias = torch.tensor(loss_bias)

            #print([t.cpu().item() for t in loss_bias])
            #print([t.cpu().item() for t in loss_ln_bias])
            #print([t.cpu().item() for t in loss_ln_weight])
            #loss_bias = sum(loss_bias) / len(loss_bias)
            """loss_bias = sum(loss_bias)/len(loss_bias)
            loss_ln_bias = sum(loss_ln_bias)/len(loss_ln_bias)
            loss_ln_weight = sum(loss_ln_weight)/len(loss_ln_weight)
"""


            lambda_ = 8. # parameter for knowledge distillation
            print(f'lambda:{lambda_}')
            loss = F.cross_entropy(output, label)  + lambda_*self.model.prompt_learner.forward_similarity()
            #+ 2*self.model.prompt_learner.forward_similarity()
            #print(loss_bias)
            #print(loss_bias, loss_ln_bias,loss_ln_weight)
            # + 8*self.model.forward_similarity() + 8*self.model.inter_similarity()
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
 

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        # self.clustering()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        self.model.cpu()
        self.model.cuda()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()

        self.after_train()


@TRAINER_REGISTRY.register()
class ClipFit(CoOp_Fit):

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

