import cv2
from ocr_utils import PoleOCR
from pipeline import InfrastructurePipeline

def run_test():
    img_path = "uploads/6815587f-7631-4587-b510-13e5be4df393.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    h_orig, w_orig = img.shape[:2]
    print(f"Original image shape: {w_orig}x{h_orig}")
    
    ocr = PoleOCR()
    
    # Run process_pole_tag on Pole 1 box
    print("\n--- Running process_pole_tag on Pole 1 box (4, 794, 214, 1439) ---")
    res = ocr.process_pole_tag(img, [4, 794, 214, 1439])
    print(f"process_pole_tag result: {repr(res)}")
    
    # Run the full pipeline test
    print("\n--- Running InfrastructurePipeline ---")
    pipeline = InfrastructurePipeline(
        comp_model="backup_models/best (2).pt",
        hardware_model="backup_models/channel_12class_v2.pt",
        shed_model="models/shed_model.pt",
        insulator_model="backup_models/insulator_model.pt"
    )
    
    res_pipeline = pipeline.predict(img_path, visualize=False)
    print(f"Pipeline Pole ID detected: {repr(res_pipeline.pole_id)}")
    print(f"Pipeline Final Class: {res_pipeline.final_class}")

if __name__ == "__main__":
    run_test()
