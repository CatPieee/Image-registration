import kornia as K
import torch
import cv2
import numpy as np
import os

# 1. 加载图片
def load_image(path, size=(640, 480)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法找到图片: {path}")
    # 调整大小以确保 LoFTR 正常工作且可视化方便
    img = cv2.resize(img, size)
    # 转换为 tensor, 范围 [0, 1], 格式为 (C, H, W)
    img_tensor = K.image_to_tensor(img, keepdim=False).float() / 255.0
    # BGR 转 RGB
    img_tensor = K.color.bgr_to_rgb(img_tensor)
    # LoFTR 通常需要单通道（灰度图）输入，且需要 Batch 维度 (1, 1, H, W)
    img_gray = K.color.rgb_to_grayscale(img_tensor)
    return img_gray, img

def apply_random_transform(img, mode='perspective'):
    """
    对图像进行随机变换以测试配准效果
    mode: 'affine' (仿射) 或 'perspective' (透视)
    """
    h, w = img.shape[:2]
    
    if mode == 'affine':
        # 随机旋转、缩放和平移
        center = (w // 2, h // 2)
        angle = np.random.uniform(-15, 15)  # 旋转 -15 到 15 度
        scale = np.random.uniform(0.8, 1.2) # 缩放 0.8 到 1.2
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # 添加一些随机平移
        M[0, 2] += np.random.uniform(-20, 20)
        M[1, 2] += np.random.uniform(-20, 20)
        transformed = cv2.warpAffine(img, M, (w, h))
    else:
        # 随机透视变换 (移动四个角)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = 0.1 * min(h, w)
        pts2 = pts1 + np.random.uniform(-offset, offset, size=pts1.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(img, M, (w, h))
        
    return transformed

# --- 测试模式：如果你想测试配准效果，可以取消下面三行的注释 ---
# _, img1_cv = load_image("data/TNO_Image_Fusion_Dataset-master/Marne_11/Marne_11_IR.bmp")
# img2_cv = apply_random_transform(img1_cv, mode='perspective')
# img1, _ = load_image("data/TNO_Image_Fusion_Dataset-master/Marne_11/Marne_11_IR.bmp") # 重新获取 tensor
# img2 = K.image_to_tensor(img2_cv, keepdim=False).float() / 255.0
# img2 = K.color.rgb_to_grayscale(K.color.bgr_to_rgb(img2))
# -------------------------------------------------------

img1, img1_cv = load_image("data/TNO_Image_Fusion_Dataset-master/Marne_11/Marne_11_IR.bmp")
img2, img2_cv = load_image("data/TNO_Image_Fusion_Dataset-master/Marne_11/Marne_11_REF.bmp")

# --- 对 img2 进行随机变换以测试配准效果 ---
print("正在对 img2 应用随机透视变换...")
img2_cv = apply_random_transform(img2_cv, mode='perspective')
# 重新将变换后的 img2_cv 转换为 tensor 供 LoFTR 使用
img2_tensor = K.image_to_tensor(img2_cv, keepdim=False).float() / 255.0
img2 = K.color.rgb_to_grayscale(K.color.bgr_to_rgb(img2_tensor))
# ---------------------------------------

# 2. 调用预训练的 LoFTR
matcher = K.feature.LoFTR(pretrained='outdoor')

# 3. 计算匹配
input_dict = {"image0": img1, "image1": img2}
with torch.no_grad():
    results = matcher(input_dict)

# 4. 获取匹配点并可视化
mkpts0 = results['keypoints0'].cpu().numpy()
mkpts1 = results['keypoints1'].cpu().numpy()

# 拼接图片进行可视化 (此时 img1_cv 和 img2_cv 已经是相同大小)
h1, w1 = img1_cv.shape[:2]
vis_img = cv2.hconcat([img1_cv, img2_cv])

# 绘制匹配线
for pt0, pt1 in zip(mkpts0, mkpts1):
    p1 = (int(pt0[0]), int(pt0[1]))
    p2 = (int(pt1[0] + w1), int(pt1[1]))
    cv2.line(vis_img, p1, p2, (0, 255, 0), 1)
    cv2.circle(vis_img, p1, 2, (0, 0, 255), -1)
    cv2.circle(vis_img, p2, 2, (0, 0, 255), -1)

# 5. 使用 RANSAC 估算单应性矩阵 (Homography)
# mkpts0 是源图像点，mkpts1 是目标图像点
# 我们找出一个 3x3 矩阵 H，使得 mkpts1 = H * mkpts0
if len(mkpts0) > 4:
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    print("单应性矩阵 (3x3 Homography Matrix):")
    print(H)

    # 6. 图像配准 (Warping)
    # 将第一张图 (img1_cv) 变换到第二张图 (img2_cv) 的坐标系下
    h, w = img2_cv.shape[:2]
    img1_warped = cv2.warpPerspective(img1_cv, H, (w, h))

    # 7. 效果展示与保存
    # 确保输出目录存在
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 保存变换后的图片
    cv2.imwrite(os.path.join(output_dir, "warped_img1.png"), img1_warped)
    cv2.imwrite(os.path.join(output_dir, "matching_result.png"), vis_img)

    # 8. 交互式预览：带有滑动条的叠加显示
    def on_trackbar(val):
        alpha = val / 100.0
        beta = 1.0 - alpha
        blended = cv2.addWeighted(img1_warped, alpha, img2_cv, beta, 0)
        cv2.imshow("Registration Preview (Drag slider to adjust alpha)", blended)

    # 创建窗口和滑动条
    cv2.namedWindow("Registration Preview (Drag slider to adjust alpha)")
    cv2.createTrackbar("Alpha", "Registration Preview (Drag slider to adjust alpha)", 50, 100, on_trackbar)

    # 初始显示
    on_trackbar(50)

    print("配准完成！")
    print(f"- 结果已保存至 {output_dir}/ 目录下")
    print("- 正在打开交互式预览窗口，请拖动滑块查看效果。按任意键退出窗口。")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("匹配点不足，无法计算变换矩阵。")