from PIL import Image


def resize_image(input_path, output_path, size):
    original_image = Image.open(input_path)

    resized_image = original_image.resize(size)

    resized_image.save(output_path)


def cut_image(input_path, output_path, area):
    original_image = Image.open(input_path)

    original_image = original_image.crop(area)

    original_image.save(output_path)

if __name__ == '__main__':
    cut_image("cz_0.png", "cz_0_fixed.png", area=(158, 112, 1154, 857))
    resize_image("cz_0_fixed.png", "cz_0_fixed.png", size=(64,64))
# Examples
# resize_image(input_image_path='9.png', output_image_path='9.png', size=(28, 28))
# cut_image("cz_0.png", "cz_0_fixed.png", area=(158, 112, 1154, 857))
