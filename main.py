import kornia as K
import torch
import cv2
import numpy as np
import os
import json
import random

# 固定随机种子以保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
        center = (w // 2, h // 2)
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += np.random.uniform(-20, 20)
        M[1, 2] += np.random.uniform(-20, 20)
        transformed = cv2.warpAffine(img, M, (w, h))
    else:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = 0.1 * min(h, w)
        pts2 = pts1 + np.random.uniform(-offset, offset, size=pts1.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(img, M, (w, h))
        
    return transformed

def run_sift_matching(img1_cv, img2_cv):
    """
    传统的 SIFT 匹配作为 Baseline
    """
    sift = cv2.SIFT_create()
    gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 比例测试 (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    inliers = 0
    if len(pts1) > 4:
        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        inliers = np.sum(mask) if mask is not None else 0
        
    return len(good_matches), int(inliers)

def save_comparison_result(img1_cv, img2_cv_warped, mkpts0, mkpts1, H, output_path, title_prefix):
    """
    生成 3 倍宽度的对比图：左&中是匹配连线，右边是配准结果
    """
    h, w = img1_cv.shape[:2]
    
    # 创建 3 倍宽度的画布
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # 左和中：原始匹配图
    vis_match = cv2.hconcat([img1_cv, img2_cv_warped])
    if mkpts0 is not None and mkpts1 is not None:
        for pt0, pt1 in zip(mkpts0, mkpts1):
            cv2.line(vis_match, (int(pt0[0]), int(pt0[1])), (int(pt1[0] + w), int(pt1[1])), (0, 255, 0), 1)
    
    canvas[:, :w*2] = vis_match
    
    # 右：配准结果 (Alpha Blending)
    if H is not None:
        img1_registered = cv2.warpPerspective(img1_cv, H, (w, h))
        blended = cv2.addWeighted(img1_registered, 0.5, img2_cv_warped, 0.5, 0)
        canvas[:, w*2:] = blended
    else:
        # 如果配准失败，显示黑色并标注文字
        cv2.putText(canvas, "Registration Failed", (w*2 + 20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # 添加标题标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"{title_prefix}: Correspondence", (20, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"{title_prefix}: Registration", (w*2 + 20, 30), font, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, canvas)

# --- 主程序逻辑 ---
def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 定义多个图像对 (RGB-IR 对)
    image_pairs = [
        {
            "name": "Marne_11",
            "ir": "data/TNO_Image_Fusion_Dataset-master/Marne_11/Marne_11_IR.bmp",
            "vis": "data/TNO_Image_Fusion_Dataset-master/Marne_11/Marne_11_REF.bmp"
        },
        {
            "name": "Athena_2men",
            "ir": "data/TNO_Image_Fusion_Dataset-master/Athena_images/2_men_in_front_of_house/IR_meting003_g.bmp",
            "vis": "data/TNO_Image_Fusion_Dataset-master/Athena_images/2_men_in_front_of_house/VIS_meting003_r.bmp"
        },
        {
            "name": "Kaptein_01",
            "ir": "data/TNO_Image_Fusion_Dataset-master/Kaptein_01/IR01.bmp",
            "vis": "data/TNO_Image_Fusion_Dataset-master/Kaptein_01/Vis01.bmp"
        },
        {
            "name": "Marne_01",
            "ir": "data/TNO_Image_Fusion_Dataset-master/Marne_01/Marne_01_IR.bmp",
            "vis": "data/TNO_Image_Fusion_Dataset-master/Marne_01/Marne_01_Vis.bmp"
        },
        {
            "name": "Reek",
            "ir": "data/TNO_Image_Fusion_Dataset-master/Reek/Reek_IR.bmp",
            "vis": "data/TNO_Image_Fusion_Dataset-master/Reek/Reek_Vis.bmp"
        }
    ]

    all_stats = {}

    print(f"开始批量实验，共 {len(image_pairs)} 个图像对...")
    
    # 初始化模型
    print("正在加载 LoFTR 模型...")
    loftr_matcher = K.feature.LoFTR(pretrained='outdoor')
    
    for pair in image_pairs:
        name = pair["name"]
        ir_path = pair["ir"]
        vis_path = pair["vis"]
        
        print(f"\n正在处理 [{name}]: {ir_path}")
        
        # 创建子目录
        pair_output_dir = os.path.join(output_dir, name)
        os.makedirs(pair_output_dir, exist_ok=True)

        # 加载图片
        img1_loftr, img1_cv = load_image(ir_path)
        img2_loftr, img2_cv = load_image(vis_path)

        # 应用随机透视变换
        img2_cv_warped = apply_random_transform(img2_cv, mode='perspective')
        img2_tensor = K.image_to_tensor(img2_cv_warped, keepdim=False).float() / 255.0
        img2_loftr_warped = K.color.rgb_to_grayscale(K.color.bgr_to_rgb(img2_tensor))

        # 1. 运行 LoFTR
        print("运行 LoFTR 匹配...")
        input_dict = {"image0": img1_loftr, "image1": img2_loftr_warped}
        with torch.no_grad():
            results = loftr_matcher(input_dict)
        
        mkpts0_loftr = results['keypoints0'].cpu().numpy()
        mkpts1_loftr = results['keypoints1'].cpu().numpy()
        
        loftr_matches = len(mkpts0_loftr)
        loftr_inliers = 0
        H_loftr = None
        if loftr_matches > 4:
            H_loftr, mask = cv2.findHomography(mkpts0_loftr, mkpts1_loftr, cv2.RANSAC, 5.0)
            loftr_inliers = int(np.sum(mask)) if mask is not None else 0

        # 2. 运行 SIFT (Baseline)
        print("运行 SIFT 匹配 (Baseline)...")
        sift_matches, sift_inliers = run_sift_matching(img1_cv, img2_cv_warped)

        # 保存统计数据
        all_stats[name] = {
            "LoFTR": {"matches": loftr_matches, "inliers": loftr_inliers},
            "SIFT": {"matches": sift_matches, "inliers": sift_inliers}
        }

        # 保存 3 倍宽度的可视化对比图
        print(f"正在保存可视化结果至 {pair_output_dir}...")
        save_comparison_result(img1_cv, img2_cv_warped, mkpts0_loftr, mkpts1_loftr, H_loftr, 
                               os.path.join(pair_output_dir, "loftr_full_result.png"), "LoFTR")
        
        # 记录 SIFT 的简单匹配结果（SIFT 通常连点都对不上，所以只存一个简单的）
        vis_sift = cv2.hconcat([img1_cv, img2_cv_warped])
        cv2.putText(vis_sift, f"SIFT Inliers: {sift_inliers}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(pair_output_dir, "sift_matches_baseline.png"), vis_sift)

    # 保存所有统计结果
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(all_stats, f, indent=4)
    
    print(f"\n批量实验完成！所有数据已保存至 {output_dir}/ 目录下")

if __name__ == "__main__":
    main()