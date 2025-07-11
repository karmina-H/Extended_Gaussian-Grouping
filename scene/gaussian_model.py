# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial import KDTree

class GaussianModel:

    def setup_functions(self):
        #스케일링 + 로테이션으로부터 공분산행렬 생성하는 함수
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree   # 최대 SH 차수
        self._xyz = torch.empty(0) # 가우시안 위치
        self._features_dc = torch.empty(0) # 색상 DC(0차) 계수 (N×C) — SH 표현의 기본 성분. RGB라면 C=3.
        self._features_rest = torch.empty(0) # 고차 SH 계수 (N×(C × (β−1))). 여기서 β는 (max_sh_degree+1)^2(계수 개수).
        self._scaling = torch.empty(0) #가우시안 스케일링
        self._rotation = torch.empty(0) #가우시안 회전
        self._opacity = torch.empty(0) #가우시안 투명도
        self._objects_dc = torch.empty(0) #가우시안 그룹핑의 핵심속성으로, classifier를 위해 개개인에 고유하게 할당된 벡터값
        self.num_objects = 16 # segmentation할 최대 object개수
        self.max_radii2D = torch.empty(0) # 각 가우시안이 투영 평면에서 차지할 최대 2D 반지름
        self.xyz_gradient_accum = torch.empty(0) #최근 여러 step 동안 누적한 위치( _x, y, z ) 그래디언트 합
        self.denom = torch.empty(0) # 누적 그래디언트 제곱의 정규화 분모. RMSProp·Adam처럼 2차 모멘트를 저장하거나, 가중치 보정에 쓰임
        self.optimizer = None
        self.percent_dense = 0  #전체 가우시안 중 밀집(dense) 영역이 차지하는 비율(0 ~ 1). 전처리·압축에서 유효
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._objects_dc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._objects_dc,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_objects(self):
        return self._objects_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1): #밑에 있는 매개변수 3개주면 가우시안 공분산을 반환해주는 함수
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self): # sh차수가 max보다 작을 때 차수 하나 올리는 함수 (점진적인 학습을 위해서)
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    #포인트클라우드 데이터로부터 가우시안 초기값들을 초기화하는 함수
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        # pcd의 xyz좌표 그리고 색상을 넘파이배열로 변환후 텐서로 변환해서 gpu에 할당.
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        #(전체 포인트 수 , 3색상(RGB) , SH 계수 수 )만큼의 비어있는 SH 텐서 생성
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # sh의 0차(SH의 DC 성분) 에 RGB 값 직접 할당
        features[:, :3, 0 ] = fused_color
         # 나머지는 0으로 초기화
        features[:, 3:, 1:] = 0.0

        # random init obj_id now
        # [pcd개수, num_objects]크기 텐서를 랜덤값이 할당된 텐서로 생성후 gpu에 올림. 그리고 이 텐서가지고 rgb2sh사용해서 fused_objects의 초기값으로 사용
        '''
        def RGB2SH(rgb):
            return (rgb - 0.5) / C0
        '''
        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda"))
        # fused_objects에 차원하나 추가 -> [pcd개수, num_objects, 1]
        fused_objects = fused_objects[:,:,None]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 포인트클라우드의 각 포인트들과 카메라의 거리제곱을 계산
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 거리 기반으로 초기 스케일(σ값) 계산 - 그래서 초기 가우시안은 거리가 멀수록 크기가 큼.
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 회전 쿼터니언(4*4)를 포인트클라우드 개수만큼 생성 후 0으로 초기화
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # w,x,y,z 에서 w = 1 즉 항등회전으로 초기화
        rots[:, 0] = 1

        #초기 투명도는 모든 포인트들에 대해서 0.1로 설정해줌
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        #가우시안 속성들 + 노출값을을학습파라미터로 등록 ->
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # max_radii2D 즉 2d에서 보이는 가우시안 크기를 pcd개수만큼텐서로 0으로 초기화해줌 -> [num] 
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # 이게 가우시안 그룹핑에서 사용하는 Gaussin_id
        # fused_objects를 [pcd개수, num_objects, 1] 에서 [pcd개수, 1, num_objects]로 변환하고 학습가능한 파라미터로 만들어줌.
        self._objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def finetune_setup(self, training_args, mask3d): 
        # Define a function that applies the mask to the gradients
        def mask_hook(grad):
            return grad * mask3d
        def mask_hook2(grad):
            return grad * mask3d.squeeze(-1)
        
        # 그레디언트 후크를 사용하여 파라미터의 그레디언트 계산 과정에 개입
        # 이 메서드들은 해당 파라미터에 대한 그레디언트가 계산될 때마다 hook_function을 호출하도록 등록하는 것
        # 특정 영역 밖의 가우시안에 대해 0 값을 가진다면, 해당 가우시안의 파라미터(xyz, opacity 등)는 업데이트되지 않도록 그레디언트를 0으로 만드는 효과를 줌
        # 특정 영역의 가우시안만 미세 조정하거나, 정적인 배경 영역은 학습에서 제외할 때 유용
        # Register the hook to the parameter (only once!)
        hook_xyz = self._xyz.register_hook(mask_hook2)
        hook_dc = self._features_dc.register_hook(mask_hook)
        hook_rest = self._features_rest.register_hook(mask_hook)
        hook_opacity = self._opacity.register_hook(mask_hook2)
        hook_scaling = self._scaling.register_hook(mask_hook2)
        hook_rotation = self._rotation.register_hook(mask_hook2)

        self._objects_dc.requires_grad = False

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def removal_setup(self, training_args, mask3d):
        # 특정 조건에 해당하지 않는 가우시안들만 남기고, 이 남은 가우시안들의 모든 파라미터를 학습 불가능(고정) 상태로 변경하는 함수

        # mask3d값들을 boolean값들로 변경하고 크기가1인차원 모두제거 후
        # 원래 mask3d가 "유지할" 가우시안을 True (또는 1)로, "제거할" 가우시안을 False (또는 0)로 나타냈다면, 이 줄을 거친 mask3d는 이제 "제거할" 가우시안을 True로, "유지할" 가우시안을 False로 표시
        mask3d = ~mask3d.bool().squeeze()

        # 마스크를 이용해서 변경하지 않을 가우시안 들만 추출
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        objects_dc_sub = self._objects_dc[mask3d].detach()


        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            # 입력 텐서를 연산 그래프에서 분리하고, 분리된 텐서에서 독립적인 복사본 만들고 requires_grad_로 연산여부 결정
            return tensor.detach().clone().requires_grad_(requires_grad)

        # 제거하지 않을 가우시안들에 대해서 gradient연산에서 제외함.
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, False))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, False))
        self._features_rest = nn.Parameter(set_requires_grad(features_rest_sub, False))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, False))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, False))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, False))
        self._objects_dc = nn.Parameter(set_requires_grad(objects_dc_sub, False))


    def inpaint_setup(self, training_args, mask3d):
        '''
        num_new_points: 새로 생성할 가우시안의 개수.
        mask_xyz_values: 인페인팅이 필요한 영역 내의 3D 좌표 (마스크 영역의 xyz 값).
        '''

        def initialize_new_features(features, num_new_points, mask_xyz_values, distance_threshold=0.25, max_distance_threshold=1, k=5):
            """Initialize new points for multiple features based on neighbouring points in the remaining area."""
            new_features = {}
            
            # 새로 생성할 가우시안의 개수가 0이면, 모두 비어있는 텐서로 생성
            if num_new_points == 0:
                for key in features:
                    new_features[key] = torch.empty((0, *features[key].shape[1:]), device=features[key].device)
                return new_features

            # 기존 가우시안들의 위치좌표값 가져오고 gpu -> cpu로 할당후 넘파이배열로 변환해줌
            remaining_xyz_values = features["xyz"]
            remaining_xyz_values_np = remaining_xyz_values.cpu().numpy()
            
            # 그리고 해당 위치값들을 이용해서 KD-Tree를 구축함.
            #  KD-트리는 다차원 공간에서 가장 가까운 이웃(Nearest Neighbor)을 효율적으로 검색할 수 있게 해주는 자료 구조
            kdtree = KDTree(remaining_xyz_values_np)
            
            # 인페인팅이 필요한 영역 내의 3D 좌표를 마찬가지로 cpu 그리고 넘파이배열로 변환
            mask_xyz_values_np = mask_xyz_values.cpu().numpy()
            query_points = mask_xyz_values_np

            # query_points에 있는 각 점에 대해, KD-트리를 사용하여 remaining_xyz_values에서 가장 가까운 k개의 이웃 가우시안을 찾음
            distances, indices = kdtree.query(query_points, k=k)
            selected_indices = indices

            # Initialize new points for each feature
            for key, feature in features.items():
                # Convert feature to numpy array
                feature_np = feature.cpu().numpy()
                
                # If we have valid neighbors, calculate the mean of neighbor points
                # selected_indices가 inpaint될 가우시안들이고 이걸 kd-tree를 이용해서 주변점인 neighbor_points의 평균을 이용해서 초기 가우시안파라미터값들을 초기화해줌
                if feature_np.ndim == 2:
                    neighbor_points = feature_np[selected_indices]
                elif feature_np.ndim == 3:
                    neighbor_points = feature_np[selected_indices, :, :]
                else:
                    raise ValueError(f"Unsupported feature dimension: {feature_np.ndim}")
                new_points_np = np.mean(neighbor_points, axis=1)
                
                # Convert back to tensor
                new_features[key] = torch.tensor(new_points_np, device=feature.device, dtype=feature.dtype)
            
            return new_features['xyz'], new_features['features_dc'], new_features['scaling'], new_features['objects_dc'], new_features['features_rest'], new_features['opacity'], new_features['rotation']
        
        mask3d = ~mask3d.bool().squeeze()
        # mask3d에서 inpaint할 것이 1으로
        mask_xyz_values = self._xyz[~mask3d]

        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        objects_dc_sub = self._objects_dc[mask3d].detach()

        # Add new points with random initialization
        sub_features = {
            'xyz': xyz_sub,
            'features_dc': features_dc_sub,
            'scaling': scaling_sub,
            'objects_dc': objects_dc_sub,
            'features_rest': features_rest_sub,
            'opacity': opacity_sub,
            'rotation': rotation_sub,
        }

        num_new_points = len(mask_xyz_values)
        with torch.no_grad():
            new_xyz, new_features_dc, new_scaling, new_objects_dc, new_features_rest, new_opacity, new_rotation = initialize_new_features(sub_features, num_new_points, mask_xyz_values)


        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Construct nn.Parameters with specified gradients
        # 기울기계산 고정해야할 것과, 기울기계산을 할 inpaint할 영역을 concat해줌
        self._xyz = nn.Parameter(torch.cat([set_requires_grad(xyz_sub, False), set_requires_grad(new_xyz, True)]))
        self._features_dc = nn.Parameter(torch.cat([set_requires_grad(features_dc_sub, False), set_requires_grad(new_features_dc, True)]))
        self._features_rest = nn.Parameter(torch.cat([set_requires_grad(features_rest_sub, False), set_requires_grad(new_features_rest, True)]))
        self._opacity = nn.Parameter(torch.cat([set_requires_grad(opacity_sub, False), set_requires_grad(new_opacity, True)]))
        self._scaling = nn.Parameter(torch.cat([set_requires_grad(scaling_sub, False), set_requires_grad(new_scaling, True)]))
        self._rotation = nn.Parameter(torch.cat([set_requires_grad(rotation_sub, False), set_requires_grad(new_rotation, True)]))
        self._objects_dc = nn.Parameter(torch.cat([set_requires_grad(objects_dc_sub, False), set_requires_grad(new_objects_dc, True)]))

        # for optimize
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Setup optimizer. Only the new points will have gradients.
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"}  # Assuming there's a learning rate for objects_dc in training_args
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    #iteration에 따라서 학습률 업데이트 하는 함수
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # PLY 파일에 저장할 속성 이름 리스트를 구성하는 함수(틀만 구성 실제 값은 안넣고)
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._objects_dc.shape[1]*self._objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        return l

    # 실제 값들을 ply파일로 저장하는 함수
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path): # ply파일 읽어서 가우시안의 각 속성들에 할당하는 함수
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        for idx in range(self.num_objects):
            objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        #특정 이름(name)을 가진 optimizer 파라미터 그룹의 tensor를 새로운 tensor로 교체하고,
        # 기존의 옵티마이저 내부 상태값(모멘텀 등)은 초기화해서 유지
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name: # 옵티마이저에 있는 모든 파라미터 속성에 대해서 순회하면서 해당하는 name에 대해서 수행
                stored_state = self.optimizer.state.get(group['params'][0], None) # 현재 옵티마이저에 있는 해당 파라미터 값들을 가져옴
                stored_state["exp_avg"] = torch.zeros_like(tensor) # 빈 상태 텐서 선언(크기는 기존 입력텐서)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]] # 기존 옵티마이져에 있는 해당 파라미터 정보삭제
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True)) # 새로운 tensor를 nn.Parameter로 매핑
                self.optimizer.state[group['params'][0]] = stored_state # 새로운 텐서를 할당 
                #기존 옵티마이저에 새로 만든거 할당
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        # 텐서에서 특정 요소를 마스킹(mask) 하여 prune하고,
        # optimizer 내부의 상태(exp_avg, exp_avg_sq)도 같이 잘라내는 함수
        optimizable_tensors = {}
        for group in self.optimizer.param_groups: # 옵티마이저 내 모든 파라미터그룹 순회
            stored_state = self.optimizer.state.get(group['params'][0], None) # 해당 그룹에 대응하는 상태가져오기
            if stored_state is not None: # 옵티마이져가 이 파라미터 상태값 관리하고 있으면 -> 마스크로 처리된 부분만 남김
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:# 새로생긴 파라미터라면 새로 파라미터 만들고 마스크부분만 짤라서 저장
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask): # 주어진 마스크에 기반해서 가우시안 점들을 제거하는 함수
        #삭제할 점을 마스크로 받아서- 관련 텐서들과 optimizer 상태를 깔끔하게 잘라냄
        valid_points_mask = ~mask # 유지할 점들 = true
        optimizable_tensors = self._prune_optimizer(valid_points_mask) # 일단 옵티마이저에서 prune

        #위의 옵티마이저에서 prune했으니 옵티마이저 텐서 최신화 되었을거고, 그걸 현재 상태에 적용
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
         #- 기존 optimizer에 등록된 tensor에 대해, 새로운 데이터를 연결해서 점 추가하는 함수입니다.
        # - Gaussian Splatting에서 새로운 점을 scene에 추가할 때 사용됨.
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1 # 각 파라미터그룹에는 객체 하나만 있어야함
            extension_tensor = tensors_dict[group["name"]]  # 해당 그룹 텐서 가져오고(추가할거)
            stored_state = self.optimizer.state.get(group['params'][0], None)  #기존 해당 그룹 텐서 가져옴 - 여기다가 위의거 추가하는 방식

            if stored_state is not None:# 상태값이 존재할 경우 즉 이미 학습중이던 파라미터라면
                #기존 학습중이던 파라미터에 새로운 0으로된 크기동일한 텐서 추가해줌
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                # 해당 파라미터의 옵티마이저 이전 상태를 삭제하고
                del self.optimizer.state[group['params'][0]]
                # 새로운 파라미터로 교체해줌.
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else: # 상태가 없는 경우 그냥 새로 하나 만들어서(크기늘려서) 반환.
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "obj_dc": new_objects_dc}
        # 새로운 가우시안 속성으로 옵티마이저에 있는 상태 업데이트 하고 현재속성도 해당 상태 업데이트 된거 반영
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        # 새로운 점들이 추가 및 삭제 되었으니위치 gradient의 누적 버퍼를 모두 0으로 초기화
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 정규화 변수도 초기화(점 개수에 맞게 확장도)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 각 점의 2D 화면상 최대 반지름 모두 0으로 초기화(개수 맞춰서)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
         # threshold에 해당하는 점들을 2개씩 복제 후 가우시안샘플링 기반 위치로 퍼뜨리는 함수
        # 현재 존재하는 가우시안의 개수
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition - graident조건을 만족하는 점들을 추출
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # gradient가 임계값보다 높은 것만을 선택
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 그리고 해당 점의 scale이 self.percent_dense*scene_extent 이거보다 커야함(특정 장면에서 일정비율 이상 차지 해야하는거)
        # 최종논리 마스크 ->
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 새로운 가우시안에 대한 위치설정하는 부분
        # 해당 스케일링 N개로 복제
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # std사이즈(표준편차(퍼짐정도) 텐서크기) 만큼 제로텐서 생성
        means =torch.zeros((stds.size(0), 3),device="cuda")
         # 해당 텐서로 분포 가우시안 분포 생성
        samples = torch.normal(mean=means, std=stds)
        # 로테이션 텐서도 마스크로 필터 및 복제
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # 위치텐서도 동일하게
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)) # 점이 나뉠때 각각의 영향을 줄이기 위해서 표준편차를 나눠줌
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1)

        # 새로 후처리 이후 새로 생성된 점들에 대한 추가 및 옵티마이저 후처리 진행
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # gradient가 큰데 크기가 작은거 선택 후 add 즉 가우시안 복제하는 함수
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_objects_dc = self._objects_dc[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 업데이트 당 평균 gradient계산
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 투명도가 일정임계값보다 낮으면 삭제
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # 화면상 너무 크기가 크게 보이는 점이나 공간상 크기가 너무 큰점 제거
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size # 화면상 너무 크기가 크게 보이는 점(2D 정보를 가지고 계산)
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # 공간상 크기가 너무 큰점(표준편차 즉 sacling기준으로 계산)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 마스크 이용해서 해당 가우시안들 제거
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    # 누적 2D 상에서의 gradient 저장하는함수
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1