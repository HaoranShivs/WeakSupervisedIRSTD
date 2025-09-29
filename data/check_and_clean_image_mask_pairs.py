#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
from PIL import Image

# 支持的图像格式
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')

def check_image_file(filepath):
    """检查单个图像文件是否可正常打开"""
    try:
        with Image.open(filepath) as img:
            img.verify()  # 快速验证结构
        with Image.open(filepath) as img:
            img.load()    # 确保像素可加载
        return True
    except Exception as e:
        return False

def get_image_files(dir_path):
    """获取目录下所有支持格式的图像文件（不含子目录）"""
    files = {}
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(SUPPORTED_EXT):
            name, ext = os.path.splitext(fname)
            files[name] = fname  # 保留原始文件名（含扩展名）
    return files

def main(root_path, dry_run=False):
    images_dir = os.path.join(root_path, "images")
    masks_dir = os.path.join(root_path, "masks")

    if not os.path.exists(images_dir):
        print(f"❌ images 目录不存在: {images_dir}")
        return
    if not os.path.exists(masks_dir):
        print(f"❌ masks 目录不存在: {masks_dir}")
        return

    # 获取 images 和 masks 中的文件名（不含扩展名）作为键
    image_files = get_image_files(images_dir)
    mask_files = get_image_files(masks_dir)

    # 找出共有的文件名（交集）
    common_names = set(image_files.keys()) & set(mask_files.keys())
    total_pairs = len(common_names)
    deleted_count = 0

    log_file = "deleted_pairs.log"
    if not dry_run:
        # 清空日志
        open(log_file, 'w').close()

    print(f"🔍 扫描中... 共发现 {total_pairs} 对图像-掩码")

    for name in sorted(common_names):
        img_path = os.path.join(images_dir, image_files[name])
        mask_path = os.path.join(masks_dir, mask_files[name])

        img_ok = check_image_file(img_path)
        mask_ok = check_image_file(mask_path)

        if not (img_ok and mask_ok):
            deleted_count += 1
            reason = []
            if not img_ok: reason.append("image损坏")
            if not mask_ok: reason.append("mask损坏")
            print(f"🗑️  [{deleted_count}] 删除对: {name} | 原因: {' + '.join(reason)}")

            if not dry_run:
                try:
                    os.remove(img_path)
                    os.remove(mask_path)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{name}\t{img_path}\t{mask_path}\n")
                except Exception as e:
                    print(f"⚠️  删除失败: {e}")

    print(f"\n✅ 扫描完成!")
    print(f"📊 总对数: {total_pairs}")
    print(f"🗑️  删除数: {deleted_count}")
    if dry_run:
        print("💡 此为预览模式，未实际删除文件。使用 --delete 执行真实删除。")
    else:
        print(f"📄 已删除文件列表保存至: {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查并清理损坏的 image-mask 对")
    parser.add_argument("path", help="包含 images/ 和 masks/ 的根目录")
    parser.add_argument("--delete", action="store_true",
                        help="执行真实删除（默认仅预览）")
    args = parser.parse_args()

    main(args.path, dry_run=not args.delete)