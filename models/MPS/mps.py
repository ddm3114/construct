# import
from transformers import AutoProcessor, AutoModel,CLIPImageProcessor,AutoTokenizer
from io import BytesIO

from PIL import Image
import torch

# load model
device = "cuda"
processor_name_or_path = "/mnt/Devin/benchmark/classify/models/MPS/weight/CLIP-ViT-H-14-laion2B-s32B-b79K"
image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(processor_name_or_path)
model_ckpt_path = "outputs/MPS_overall_checkpoint.pth"
model.eval().to(device)

def infer_example(images, prompt, condition, clip_model, clip_processor, tokenizer, device):
    def _process_image(image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open( image )
        image = image.convert("RGB")
        pixel_values = clip_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def _tokenize(caption):
        input_ids = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    image_inputs = torch.concatenate([_process_image(images[0]).to(device), _process_image(images[1]).to(device)])
    text_inputs = _tokenize(prompt).to(device)
    condition_inputs = _tokenize(condition).to(device)

    with torch.no_grad():
        text_features, image_0_features, image_1_features = clip_model(text_inputs, image_inputs, condition_inputs)
        image_0_features = image_0_features / image_0_features.norm(dim=-1, keepdim=True)
        image_1_features = image_1_features / image_1_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_0_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_1_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_1_features))
        scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)[0]

    return probs.cpu().tolist()

img_0, img_1 = "image1.jpg", "image2.jpg"
# infer the best image for the caption
prompt = "the caption of image" 

# condition for overall
condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things." 

print(infer_example([img_0, img_1], prompt, condition, model, image_processor, tokenizer, device))