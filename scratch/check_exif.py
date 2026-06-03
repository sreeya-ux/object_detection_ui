from PIL import Image, ExifTags
import cv2

def check_exif():
    img_path = "uploads/0748854a-f0dd-4be5-98b4-3c6a54545831.jpg"
    
    # Check dimensions via PIL
    print("--- PIL info ---")
    try:
        with Image.open(img_path) as pil_img:
            print(f"PIL raw size: {pil_img.size}")
            info = pil_img._getexif()
            if info:
                exif = {ExifTags.TAGS.get(k, k): v for k, v in info.items()}
                print("EXIF Keys:")
                for k, v in exif.items():
                    if k in ("Orientation", "Make", "Model", "ExifImageWidth", "ExifImageHeight"):
                        print(f"  {k}: {v}")
            else:
                print("No EXIF metadata found")
    except Exception as e:
        print(f"Error reading EXIF: {e}")

    # Check dimensions via OpenCV
    print("\n--- OpenCV info ---")
    img = cv2.imread(img_path)
    if img is not None:
        print(f"OpenCV loaded shape: {img.shape}")
    else:
        print("OpenCV failed to read image")

if __name__ == "__main__":
    check_exif()
