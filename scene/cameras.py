#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scipy.spatial.transform import Rotation as R

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", objects=None, style_transfer=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device) # clamp()로 픽셀 값을 0~1로 제한(0~255를 0~1로 정규화)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 투명도 마스크이용해서 원본이미지에 alpha값을 곱해서 할당
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
         #기본 클리핑 거리설정정
        self.zfar = 100.0
        self.znear = 0.01
        #변환 행렬 저장(크기조정+이동)
        self.trans = trans
        self.scale = scale

        #카메라 변환행렬 계산하는 부분
        #월드 -> 뷰 변환행렬
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        #투영 변환행렬
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        #최종 변환 행렬 = view × projection
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        #역행렬연산을 통해서 월드 좌표계에서의 카메라 중심위치 찾아냄
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if objects is not None:
            self.objects = objects.to(self.data_device)
        else:
            self.objects = None
        
        if style_transfer:
            self.transfer_image = self.original_image.clone()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

