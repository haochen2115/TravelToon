import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def merge_png(titles, image_paths, travel_topic, final_result):
    """
    合并多张PNG图片并添加标题，所有图片会被调整为128x128大小
    
    Args:
        titles (list): 图片标题列表
        image_paths (list): 图片路径列表
        travel_topic (str): 主题标题
        final_result (str): 输出文件路径
    """
    if len(titles) != len(image_paths):
        raise ValueError("titles和image_paths长度不匹配")

    max_width = 228
    current_height = 50
    img_size = 128
    total_height = current_height

    # 计算总高度
    total_height += (img_size + 60) * len(image_paths)

    # 创建画布
    template = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(template)
    font = ImageFont.load_default()

    # 绘制主题标题
    topic_text_bbox = draw.textbbox((0, 0), travel_topic, font=font)
    topic_text_height = topic_text_bbox[3] - topic_text_bbox[1] + 50
    total_height += topic_text_height
    draw.text((50, 10), travel_topic, fill="black", font=font)

    # 粘贴图片并添加标题
    current_height = 50
    for title, image_path in zip(titles, image_paths):
        with Image.open(image_path) as img:
            # 调整图片大小为128x128
            img_resized = img.resize((img_size, img_size), Image.LANCZOS)
            template.paste(img_resized, (50, current_height))
            draw.text((50, current_height + img_size + 10), title, fill="black", font=font)
            current_height += img_size + 60

    template.save(final_result)

def convert_to_sketch(image_path, output_path):
    """
    将图片转换为素描风格
    
    Args:
        image_path (str): 输入图片路径
        output_path (str): 输出图片路径
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blurred = 255 - blurred
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    sketch = 255 - sketch

    thickness = 3
    kernel = np.ones((thickness, thickness), np.uint8)
    sketch = cv2.dilate(sketch, kernel, iterations=1)

    cv2.imwrite(output_path, sketch)

if __name__ == "__main__":
    # 这里可以添加测试代码
    pass
