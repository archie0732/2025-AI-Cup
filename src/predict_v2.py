import os
from ultralytics import YOLO


best_model_path = "./runs/detect/aortic_run_X_final/weights/best.pt"


model = YOLO(best_model_path)
submission_file = "submission_v8x_optimized.txt"

with open(submission_file, 'w') as f:
    results = model.predict(
        source="./datasets/test/images",
        imgsz=640,
        device=0,
        verbose=False,
        stream=True,
        
        
        augment=True,  
        conf=0.001,    
        iou=0.65,      
        agnostic_nms=True 
    )
    
    for i, result in enumerate(results):
        if (i + 1) % 1000 == 0: print(f"fininshing {i + 1}")
        
        filename = os.path.basename(result.path).replace(".png", "")
        boxes = result.boxes
        if len(boxes) > 0:
            for k in range(len(boxes)):
                cls = int(boxes.cls[k].item())
                conf = boxes.conf[k].item()
                x1, y1, x2, y2 = boxes.xyxy[k].tolist()
                line = f"{filename} {cls} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                f.write(line)

print(f"file path: {submission_file}")
