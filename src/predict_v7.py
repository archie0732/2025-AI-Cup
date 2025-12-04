import os
from ultralytics import YOLO

model_path = "./runs/detect/aortic_run_11x_Human_Corrected/weights/best.pt"

if not os.path.exists(model_path):
    print("❌ no best.pt")
else:
    model = YOLO(model_path)
    
    submission_file = "submission_Human_Corrected.txt"
    base_dir = "./datasets"

    
    with open(submission_file, 'w') as f:
        results = model.predict(
            source=f"{base_dir}/test/images",
            imgsz=640,   
            device=0,    
            conf=0.001,  
            iou=0.65,    
            augment=True,
            verbose=False,
            stream=True
        )
        
        count = 0
        for result in results:
            count += 1
            if count % 1000 == 0: print(f"Finish {count}...")
            
            filename = os.path.basename(result.path).replace(".png", "")
            boxes = result.boxes
            if len(boxes) > 0:
                for k in range(len(boxes)):
                    cls = int(boxes.cls[k].item())
                    conf = boxes.conf[k].item()
                    x1, y1, x2, y2 = boxes.xyxy[k].tolist()
                    line = f"{filename} {cls} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                    f.write(line)

    print(f"✅ Path：{submission_file}")
