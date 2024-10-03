import os
from PIL import Image
from ultralytics import YOLO

model_path = 'yolov10_fine_tuned.pt'
root_dir = 'cattle muzzle'
model = YOLO(model_path)

output_dir = 'muzzle dataset'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


for example in os.listdir(root_dir):
    ex_path = os.path.join(root_dir, example)
    out_folder_path = os.path.join(output_dir, example)

    if not os.path.exists(out_folder_path):
        os.mkdir(out_folder_path)

    for file in os.listdir(ex_path):
        file_path = os.path.join(ex_path, file)

        image = Image.open(file_path)
        image = image.resize((1400, 1400))
        results = model(image, verbose = False)

        if len(results[0].boxes) >= 1:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cropped_image = image.crop([x1, y1, x2, y2])
  
                cropped_filename = f"{file}.jpg"
                cropped_image_path = os.path.join(out_folder_path, cropped_filename)
                
                cropped_image.save(cropped_image_path)
                print(f"Cropped image saved to {cropped_image_path} ")
                
        
                
   