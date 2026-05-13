import flair
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pretrained = flair.download_weights_from_hf(model_repo='xiaorui638/flair', filename='flair-cc3m-recap.pt')
model, _, preprocess = flair.create_model_and_transforms('ViT-B-16-FLAIR', pretrained=pretrained)

model.to(device)
model.eval()

tokenizer = flair.get_tokenizer('ViT-B-16-FLAIR')

image = preprocess(Image.open("../assets/puppy.jpg")).unsqueeze(0).to(device)

text = tokenizer(["In the image, a small white puppy with black ears and eyes is the main subject", # ground-truth caption
                  "The white door behind the puppy is closed, and there's a window on the right side of the door", # ground-truth caption
                  "A red ladybug is surrounded by green glass beads", # non-ground-truth caption
                  "Dominating the scene is a white desk, positioned against a white brick wall"]).to(device) # non-ground-truth caption

with torch.no_grad(), torch.cuda.amp.autocast():
    flair_logits = model.get_logits(image=image, text=text)
    clip_logits = model.get_logits_as_clip(image=image, text=text)

    print("logits get using flair's way:", flair_logits) # [4.4062,  6.9531, -20.5000, -18.1719]
    print("logits get using clip's way:", clip_logits) # [12.4609, 15.6797, -3.8535, -0.2281]


