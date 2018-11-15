from PIL import Image


def resize_image(input_image_path,
                 output_image_path,
                 size):
    original_image = Image.open(input_image_path)

    resized_image = original_image.resize(size)
    width, height = resized_image.size

    resized_image.save(output_image_path)


# Example
#resize_image(input_image_path='9.png', output_image_path='9.png', size=(28, 28))
