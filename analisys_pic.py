import os
import fitz  # PyMuPDF

def extract_images_from_pdf(pdf_path, output_dir):
    """
    从 PDF 文件中提取所有嵌入图像，并保存到指定目录。

    参数:
        pdf_path (str): 输入 PDF 文件的路径。
        output_dir (str): 图像保存的目标目录（会自动创建）。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 打开 PDF 文档
    doc = fitz.open(pdf_path)
    
    image_count = 0
    for page_index, page in enumerate(doc):
        # 获取当前页的所有图像（full=True 包含被引用多次的图像）
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                # img[0] 是 XREF（图像对象的引用 ID）
                xref = img[0]
                # 提取图像数据
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]  # 如 'png', 'jpg', 'jp2' 等
                
                # 构造文件名，避免重复（按页和图编号）
                filename = f"page{page_index:03d}_img{img_index:03d}.{image_ext}"
                filepath = os.path.join(output_dir, filename)
                
                # 写入文件
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                image_count += 1
                print(f"已保存: {filepath}")
            
            except Exception as e:
                print(f"提取 page{page_index} 的图像 {img_index} 时出错: {e}")
                continue

    doc.close()
    print(f"\n✅ 共提取并保存 {image_count} 张图像到 '{output_dir}'")

# ===== 使用示例 =====
if __name__ == "__main__":
    path1 = "pdfs/1.pdf"      # 替换为你的 PDF 路径
    path2 = "picts"    # 替换为你想保存图像的文件夹路径

    extract_images_from_pdf(path1, path2)