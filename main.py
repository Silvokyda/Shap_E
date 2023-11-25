from flask import Flask, request, jsonify
from PIL import Image
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif, load_image
from diffusers import DiffusionPipeline
from PIL import Image
import torch

app = Flask(__name__)

# Load models
prior_pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
shape_img2img_pipeline = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16, variant="fp16").to("cuda")

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json

    # Check if the request contains a prompt or an image file
    if 'prompt' in data:
        prompt = data['prompt']

        # Generate image using the provided prompt
        image_embeds, negative_image_embeds = prior_pipeline(prompt, guidance_scale=1.0).to_tuple()
        generated_image = pipeline(prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds).images[0]

    elif 'image' in request.files:
        # If an image file is provided, process it
        uploaded_image = Image.open(request.files['image']).resize((256, 256))
        guidance_scale = 3.0

        # Generate 3D image (GIF) from the 2D image
        images = shape_img2img_pipeline(uploaded_image, guidance_scale=guidance_scale, num_inference_steps=64, frame_size=256).images
        generated_image = export_to_gif(images[0], None)

    else:
        return jsonify({'error': 'Invalid request'}), 400

    # Save and return the generated image
    generated_image_path = "generated_image.png"
    generated_image.save(generated_image_path)

    return jsonify({'generated_image_path': generated_image_path})


if __name__ == '__main__':
    app.run(debug=True)