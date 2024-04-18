import os
from PIL import Image


def crop_and_save_images(base_path, file_list, output_path):
    with open(file_list, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        image_name = line.strip()
        image_path = os.path.join(base_path, image_name)
        image = Image.open(image_path)

        img_width, img_height = image.size

        crop_size = 512
        crops = {
            '1': (0, 0, crop_size, crop_size),  # 左上
            '2': (0, img_height - crop_size, crop_size, img_height),  # 左下
            '3': (img_width - crop_size, 0, img_width, crop_size),  # 右上
            '4': (img_width - crop_size, img_height - crop_size, img_width, img_height),  # 右下
        }

        for key, (left, upper, right, lower) in crops.items():
            cropped_image = image.crop((left, upper, right, lower))
            new_image_name = image_name.replace('.tif', f'_{key}.tif')
            cropped_image.save(os.path.join(output_path, new_image_name))
            new_lines.append(new_image_name + '\n')

    return new_lines


def process_images_and_save_list(base_path, input_txt, output_path, output_txt):
    new_lines = crop_and_save_images(base_path, input_txt, output_path)
    with open(output_txt, 'w') as file:
        file.writelines(new_lines)


if __name__ == "__main__":
    base_path = 'path_to_images'
    output_path = 'path_to_cropped_images'
    os.makedirs(output_path, exist_ok=True)
    
    process_images_and_save_list(base_path, 'train.txt', output_path, 'train_aug.txt')
    process_images_and_save_list(base_path, 'val.txt', output_path, 'val_aug.txt')
