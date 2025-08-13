import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(
    source_dir,
    output_root_dir,
    normal_signs_folder="normal_signs",
    damaged_signs_folder="damaged_signs",
    test_size=0.2,
    random_state=42
):
    """
    Splits dataset into train/test folders from raw category folders.

    Args:
        source_dir (str): Path where normal_signs and damaged_signs are located.
        output_root_dir (str): Path where train/test folders will be created.
        normal_signs_folder (str): Name of the normal signs folder.
        damaged_signs_folder (str): Name of the damaged signs folder.
        test_size (float): Fraction for test set.
        random_state (int): Seed for reproducibility.
    """

    def collect_images(category):
        category_path = os.path.join(source_dir, category)
        return [os.path.join(category_path, f)
                for f in os.listdir(category_path)
                if os.path.isfile(os.path.join(category_path, f))]

    normal_images = collect_images(normal_signs_folder)
    damaged_images = collect_images(damaged_signs_folder)

    print(f"Found {len(normal_images)} normal images, {len(damaged_images)} damaged images.")

    # Split into train/test
    train_normal, test_normal = train_test_split(normal_images, test_size=test_size, random_state=random_state)
    train_damaged, test_damaged = train_test_split(damaged_images, test_size=test_size, random_state=random_state)

    # Define output paths
    train_normal_dir = os.path.join(output_root_dir, "train", normal_signs_folder)
    test_normal_dir = os.path.join(output_root_dir, "test", normal_signs_folder)
    train_damaged_dir = os.path.join(output_root_dir, "train", damaged_signs_folder)
    test_damaged_dir = os.path.join(output_root_dir, "test", damaged_signs_folder)

    # Create directories
    for d in [train_normal_dir, test_normal_dir, train_damaged_dir, test_damaged_dir]:
        os.makedirs(d, exist_ok=True)

    # Copy helper
    def copy_files(files, target_dir):
        for f in files:
            shutil.copy(f, target_dir)

    # Copy images
    print(f"Copying {len(train_normal)} normal images to train...")
    copy_files(train_normal, train_normal_dir)
    print(f"Copying {len(test_normal)} normal images to test...")
    copy_files(test_normal, test_normal_dir)
    print(f"Copying {len(train_damaged)} damaged images to train...")
    copy_files(train_damaged, train_damaged_dir)
    print(f"Copying {len(test_damaged)} damaged images to test...")
    copy_files(test_damaged, test_damaged_dir)

    print("\n✅ Dataset split complete!")
    print(f"Train set location: {os.path.join(output_root_dir, 'train')}")
    print(f"Test set location:  {os.path.join(output_root_dir, 'test')}")

if __name__ == "__main__":
    # Change this if your data is somewhere else
    source = r"C:\model\NEWDATA"
    output_root = r"C:\model\NEWDATA"
    split_dataset(source, output_root, test_size=0.2)
