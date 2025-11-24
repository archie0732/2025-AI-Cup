import os
import shutil
import numpy as np
from sklearn.model_selection import KFold
from ultralytics import YOLO


base_dir = "./datasets"
source_img_dir = f"{base_dir}/train/images" 
source_lbl_dir = f"{base_dir}/train/labels"


K = 5           
EPOCHS = 100    
BATCH = 16     
IMGSZ = 640     
DEVICE = 1     


if not os.path.exists(source_img_dir):
    print(f"❌ cannot find the file {source_img_dir}")
else:
    
    all_files = os.listdir(source_img_dir)
    
    patient_ids = sorted(list(set([f.split('_')[0] for f in all_files if f.startswith('patient')])))
    patient_ids = np.array(patient_ids)

    print(f"📊 all find {len(patient_ids)} people (count: 50 )")
    print(f"🚀 準備開始 {K}-Fold (Use GPU {DEVICE})...")

    
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    
    for k, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        print(f"\n{'='*20} 開始訓練 Fold {k+1}/{K} {'='*20}")
        
        fold_dir = f"./datasets_fold_{k}"
        for split in ['train', 'val']:
            os.makedirs(f"{fold_dir}/{split}/images", exist_ok=True)
            os.makedirs(f"{fold_dir}/{split}/labels", exist_ok=True)
        
        for pid in patient_ids[train_idx]:
            for f in all_files:
                if f.startswith(pid) and f.endswith('.png'):
                    src_img = os.path.join(source_img_dir, f)
                    dst_img = f"{fold_dir}/train/images/{f}"
                    if not os.path.exists(dst_img): shutil.copy(src_img, dst_img)
                    
                    txt_name = f.replace('.png', '.txt')
                    src_lbl = os.path.join(source_lbl_dir, txt_name)
                    dst_lbl = f"{fold_dir}/train/labels/{txt_name}"
                    if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                        shutil.copy(src_lbl, dst_lbl)
        
        for pid in patient_ids[val_idx]:
            for f in all_files:
                if f.startswith(pid) and f.endswith('.png'):
                    src_img = os.path.join(source_img_dir, f)
                    dst_img = f"{fold_dir}/val/images/{f}"
                    if not os.path.exists(dst_img): shutil.copy(src_img, dst_img)
                    
                    txt_name = f.replace('.png', '.txt')
                    src_lbl = os.path.join(source_lbl_dir, txt_name)
                    dst_lbl = f"{fold_dir}/val/labels/{txt_name}"
                    if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                        shutil.copy(src_lbl, dst_lbl)

        yaml_content = f"""
        path: {os.path.abspath(fold_dir)}
        train: train/images
        val: val/images
        names:
          0: aortic_valve
        """
        yaml_file = f"aortic_fold_{k}.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)
            
        model = YOLO('yolov8l.pt')
        
        results = model.train(
            data=yaml_file,
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,        
            project='runs/kfold', 
            name=f'fold_{k}',     
            workers=1,            
            patience=0,          
            cache=True,          
            augment=True,
            close_mosaic=10,      
            cos_lr=True,
            optimizer='auto'
        )
        
        print(f"✅ Fold {k+1} traning finishing!")

    print("\n🏆🏆🏆 all K-Fold traning  complete！ 🏆🏆🏆")
