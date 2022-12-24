import cv2 as cv
import numpy as np
import torch
import tqdm


def get_screen_batch(
    height: int,
    width: int,
    render_batch_size: int,
    bias: int,
    device=torch.device("cpu"),
) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0, height - 1, height, device=device),
            torch.linspace(0, width - 1, width, device=device)
        ),
        dim=-1
    )
    coords = torch.reshape(coords, (-1, 2))
    coords = coords[bias: bias + render_batch_size].long()
    return coords


def save_img(img: np.ndarray, path):
    cv.imwrite(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))


def resize_img(img: np.ndarray, H, W):
    return cv.resize(img, (W, H), interpolation=cv.INTER_AREA)


class ForegroundRenderer:
    def __init__(self, nerf, width, height, num_samples, num_isamples, background_w, ray_chunk, sample5d_chunk, is_train, device) -> None:
        self.nerf = nerf                        # Neural Radiance Fields (contains encoding, coarse net and fine net)
        self.width = width                      # width of the rendering scene
        self.height = height                    # height of the rendering scene
        self.num_samples = num_samples          # number of sampling
        self.num_isamples = num_isamples        # number of Hierarchical volume sampling
        self.ray_chunk = ray_chunk              # chunk of ray casting
        self.sample5d_chunk = sample5d_chunk    # chunk of net
        self.background_w = background_w        # whether or not transform image's background to white
        self.device = device                    # device of the whole volume renderer

        self.nerf.to(device)                    # to device(CPU or GPU)
        
        self.train(is_train)                    # train or eval(just rendering or test)

    def train(self, is_train):
        """ Train or Eval(just rendering or test) """
        self.is_train = is_train
        self.nerf.train(is_train)

    def _intersect_sphere(self, rays_o, rays_d):
        '''
        ray_o, ray_d: [..., 3]
        compute the depth of the intersection point between this ray and unit sphere
        '''
        # note: d1 becomes negative if this mid point is behind camera
        d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
        p = rays_o + d1.unsqueeze(-1) * rays_d
        # consider the case where the ray does not intersect the sphere
        ray_d_cos = 1. / torch.norm(rays_d, dim=-1)
        p_norm_sq = torch.sum(p * p, dim=-1)
        if (p_norm_sq >= 1.).any():
            raise Exception('Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
        d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

        return d1 + d2

    def _perturb_samples(self, z_vals):
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

        return z_vals

    def _sample(self, rays):
        """ 
        Volume Sampling 
        ===============
        **we partition [tn,tf ] into N evenly-spaced bins and
        then draw one sample uniformly at random from within each bin**
        Input:
            rays: (rays_o, rays_d) tuple[tensor, tensor]
        Output:
            pos_locs: tensor [num_rays, num_samples, 3]  spatial locations used for the input of Coarse Net
            t: tensor       [num_rays, num_samples]  t of sampling 
        """
        # rays_o.shape=[num_rays, 3], rays_d.shape=[num_rays, 3]
        rays_o, rays_d = rays
        fg_far_depth = self._intersect_sphere(rays_o, rays_d)
        fg_near_depth = 1e-4 * torch.ones_like(rays_d[..., 0])
        step = (fg_far_depth - fg_near_depth) / (self.num_samples - 1)
        fg_depth = torch.stack([fg_near_depth + i * step for i in range(self.num_samples)], dim=-1)
        
        if self.is_train:
            fg_depth = self._perturb_samples(fg_depth)
        
        fg_locs = rays_o[..., None, :] + rays_d[..., None, :]  * fg_depth[..., :, None]
        
        return fg_locs, fg_depth
        
    def _parse_voxels(self, voxels, t_vals, rays_d) -> dict:
        """
        The volume rendering integral equation
        was calculated by Monte Carlo integral method
        Inputs:
            voxels: tensor [num_rays, num_samples, 4]   results of NN forward
            t_vals: tensor [num_rays, num_samples]      t of sampling 
            rays_d: tensor [num_rays, 3]                rays' directions
        Output:
            rbg_map and cdf_map
            rbg_map: tensor [num_rays, 3]               RGB map of the rendering scene
            cdf_map: tensor [num_rays, num_samples - 1] CDF map (Cumulative Distribution Function)
        """
        t_delta = t_vals[..., 1:] - t_vals[..., :-1]
        t_delta = torch.cat(
            (t_delta, torch.tensor(
                [1e10], device=self.device).expand_as(t_delta[..., :1])),
            dim=-1
        ) * torch.norm(rays_d[..., None, :], dim=-1) # [num_rays, num_samples]
        
        c_i = torch.sigmoid(voxels[..., :3]) # [num_rays, num_samples, 3]
        alpha_i = 1 - torch.exp(-torch.relu(voxels[..., 3]) * t_delta)
        # exp(a + b) == exp(a) * exp(b)
        w_i = torch.cumprod(
            torch.cat(
                (torch.ones((*alpha_i.shape[:-1], 1), device=self.device), 1.0 - alpha_i + 1e-10),
                dim=-1
            ),
            dim=-1
        )[:, :-1]  
        Lambda = w_i[..., -1]
        w_i = alpha_i * w_i
        # [:, :-1]   [num_rays, num_samples + 1] => [num_rays, num_samples]
        rgb_map = torch.sum(
            w_i[..., None] * c_i, # [num_rays, num_samples, 1] * [num_rays, num_samples, 3]
            dim=-2,  # num_samples
            keepdim=False
        )  # [num_rays, 3]
        
        # Normalizing these weights as ˆwi = wi/∑Ncj=1 wj 
        # produces a piecewise-constant PDF along the ray. **5.2**
        pdf_map = w_i[..., 1:-1] + 1e-5  
        # prevent nans
        pdf_map = pdf_map / torch.sum(pdf_map, -1, keepdim=True)
        cdf_map = torch.cumsum(pdf_map, dim=-1)
        cdf_map = torch.cat(
            (torch.zeros_like(cdf_map[..., :1]), cdf_map), dim=-1
        )

        if self.background_w:
            rgb_map = rgb_map + (1.0 - torch.sum(w_i, dim=-1, keepdim=False)[..., None])

        return dict(
            rgb_map=rgb_map,
            cdf_map=cdf_map,
            Lambda=Lambda
        )

    def _hierarchical_sample(self, rays, t_vals: torch.Tensor, cdf_map: torch.Tensor):
        """
        Hierarchical volume sampling in paper
        =====================================
        **We sample a second set of Nf locations from this distribution
        using inverse transform sampling, evaluate our “fine” network at the union of the
        first and second set of samples, and compute the final rendered color of the ray Cf (r) using Eqn. 3 but using all Nc + Nf samples.**
        Follow the official approach. 
        Inputs:
            rays: (rays_o, rays_d) tuple[tensor, tensor]
            t_vals: tensor [num_rays, num_samples] t of the result of uniform sampling (the function `_sample`)
            cdf_map: tensor [num_rays, num_samples - 1] CDF map (Cumulative Distribution Function)
        Output:
            pos_locs: tensor [num_rays, num_samples + num_isamples, 3]  spatial locations used for the input of Fine Net
            t_vals: tensor [num_rays, num_samples + num_isamples] t of sampling and hierarchical sampling
        """
        rays_o, rays_d = rays
        t_vals_mid = (t_vals[..., :-1] + t_vals[..., 1:]) * 0.5 
        # [num_rays, num_samples - 1] == cdf_map.shape

        u = torch.rand((*cdf_map.shape[:-1], self.num_isamples), device=self.device).contiguous() # [num_rays, num_isamples]
        
        index = torch.searchsorted(cdf_map, u, right=True) # [num_rays, num_isamples] find > u
        index = torch.stack((
            torch.max(torch.zeros_like(index, device=self.device), index - 1), 
            torch.min(torch.full_like(index, fill_value=(cdf_map.shape[-1] - 1) * 1.0, device=self.device), index) 
            ), dim=-1) # [num_rays, num_isamples, 2]
        
        shape_m = [index.shape[0], index.shape[1], cdf_map.shape[-1]] # [num_rays, num_isamples, num_samples - 1]
        cdf_map_gather = torch.gather(cdf_map.unsqueeze(1).expand(shape_m), dim=2, index=index) # [num_rays, num_isamples, 2]
        t_vals_gather = torch.gather(t_vals_mid.unsqueeze(1).expand(shape_m), dim=2, index=index) # [num_rays, num_isamples, 2]
        
        denom = cdf_map_gather[..., 1] - cdf_map_gather[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=self.device), denom) # [num_rays, num_isamples]
        
        t_vals_fine = torch.lerp(t_vals_gather[..., 0], t_vals_gather[..., 1], (u - cdf_map_gather[..., 0]) / denom).detach()
        t_vals, _ = torch.sort(torch.cat((t_vals, t_vals_fine), dim=-1), dim=-1)

        pos_locs = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None] # [num_rays, num_samples + num_isamples, 3]

        return pos_locs, t_vals

    def cast_rays(self, rays):
        """
        Rays Casting
        ============
        1. sample spatial locations and t for coarse net of NeRF
        2. using spatial locations and view directions as inputs of the coarse net of NeRF 
            to get voxels (rgb density) of the spatial locations
        3. parse voxels (rgb density) to rgb map of the scene (calculate integral) 
            and cdf of the sampling t
        4. hierarchical sample spatial locations and t for fine net of NeRF
        5. using spatial locations obtained by [4] and view directions as inputs of the fine net of NeRF 
            to get voxels (rgb density) of the spatial locations
        6. return the results (rgb map) of [2] and [5]
        Inputs:
            rays: (rays_o, rays_d) 
            view_dirs: tensor  view directions [num_rays, 3]
        Outputs:
            rgb map of [2] and [5]
        """
        rays_o, rays_d = rays
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        coarse_nerf_locs, t_coarse = self._sample((rays_o, rays_d))
        coarse_voxels = self._voxel_sample5d(
            torch.cat((coarse_nerf_locs,
                       view_dirs[..., None, :].expand_as(coarse_nerf_locs)), dim=-1),
            "coarse"
        )
        coarse_info = self._parse_voxels(coarse_voxels, t_coarse, rays_d)
        fine_nerf_locs, t_fine = self._hierarchical_sample(
            (rays_o, rays_d), t_coarse, coarse_info["cdf_map"]
        )
        fine_voxels = self._voxel_sample5d(
            torch.cat((fine_nerf_locs,
                       view_dirs[..., None, :].expand_as(fine_nerf_locs)), dim=-1),
            "fine"
        )
        fine_info = self._parse_voxels(fine_voxels, t_fine, rays_d)
        return coarse_info["rgb_map"], fine_info["rgb_map"], coarse_info["Lambda"], fine_info["Lambda"]

    def _voxel_sample5d(self, x: torch.Tensor, net_type) -> torch.Tensor:
        """
        sample voxels (rgb density) using NeRF
        =======================================
        Inputs:
            x : spatial locations and view directions
            net_type: "coarse" or "fine" to select coarse net or fine net to forward
        Outputs:
            rgb + density
        """
        self.nerf.net_(net_type)
        if self.sample5d_chunk is None or self.sample5d_chunk <= 1:
            return self.nerf(x)
        return torch.cat(
            [self.nerf(x[i:i + self.sample5d_chunk])
             for i in range(0, x.shape[0], self.sample5d_chunk)],
            dim=0
        )
        

class BackgroundRenderer:
    def __init__(self, nerf, width, height, num_samples, num_isamples, background_w, ray_chunk, sample5d_chunk, is_train, device) -> None:
        self.nerf = nerf                        # Neural Radiance Fields (contains encoding, coarse net and fine net)
        self.width = width                      # width of the rendering scene
        self.height = height                    # height of the rendering scene
        self.num_samples = num_samples          # number of sampling
        self.num_isamples = num_isamples        # number of Hierarchical volume sampling
        self.ray_chunk = ray_chunk              # chunk of ray casting
        self.sample5d_chunk = sample5d_chunk    # chunk of net
        self.background_w = background_w        # whether or not transform image's background to white
        self.device = device                    # device of the whole volume renderer

        self.nerf.to(device)                    # to device(CPU or GPU)
        
        self.train(is_train)                    # train or eval(just rendering or test)

    def train(self, is_train):
        """ Train or Eval(just rendering or test) """
        self.is_train = is_train
        self.nerf.train(is_train)

    def _perturb_samples(self, z_vals):
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

        return z_vals

    def _depth2pts_outside(self, ray_o, ray_d, depth):
        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''
        # note: d1 becomes negative if this mid point is behind camera
        d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
        p_mid = ray_o + d1.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)
        
        ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
        d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
        p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm)

        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                    torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                    rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        # now calculate conventional depth
        depth_real = 1. / (depth + 1e-6) * torch.cos(theta) * ray_d_cos + d1
        return pts, depth_real
    
    def _sample(self, rays):
        """ 
        Volume Sampling 
        ===============
        **we partition [tn,tf ] into N evenly-spaced bins and
        then draw one sample uniformly at random from within each bin**
        Input:
            rays: (rays_o, rays_d) tuple[tensor, tensor]
        Output:
            pos_locs: tensor [num_rays, num_samples, 3]  spatial locations used for the input of Coarse Net
            t: tensor       [num_rays, num_samples]  t of sampling 
        """
        # rays_o.shape=[num_rays, 3], rays_d.shape=[num_rays, 3]
        rays_o, rays_d = rays
        dots_sh = list(rays_d.shape[:-1])
        bg_depth = torch.linspace(0., 1., self.num_samples, device=self.device).view(
            [1, ] * len(dots_sh) + [self.num_samples]
        ).expand(dots_sh + [self.num_samples])
        if self.is_train:
            bg_depth = self._perturb_samples(bg_depth)
        bg_depth = torch.flip(bg_depth, dims=[-1])

        bg_rays_o = rays_o.unsqueeze(-2).expand(dots_sh + [self.num_samples, 3])
        bg_rays_d = rays_d.unsqueeze(-2).expand(dots_sh + [self.num_samples, 3])
        
        bg_locs, _ = self._depth2pts_outside(bg_rays_o, bg_rays_d, bg_depth)

        return bg_locs, bg_depth
        
    def _parse_voxels(self, voxels, t_vals, rays_d) -> dict:
        """
        The volume rendering integral equation
        was calculated by Monte Carlo integral method
        Inputs:
            voxels: tensor [num_rays, num_samples, 4]   results of NN forward
            t_vals: tensor [num_rays, num_samples]      t of sampling 
            rays_d: tensor [num_rays, 3]                rays' directions
        Output:
            rbg_map and cdf_map
            rbg_map: tensor [num_rays, 3]               RGB map of the rendering scene
            cdf_map: tensor [num_rays, num_samples - 1] CDF map (Cumulative Distribution Function)
        """
        t_delta = t_vals[..., :-1] - t_vals[..., 1:]
        t_delta = torch.cat(
            (t_delta, torch.tensor(
                [1e10], device=self.device).expand_as(t_delta[..., :1])),
            dim=-1
        ) * torch.norm(rays_d[..., None, :], dim=-1) # [num_rays, num_samples]
        
        c_i = torch.sigmoid(voxels[..., :3]) # [num_rays, num_samples, 3]
        alpha_i = 1 - torch.exp(-torch.relu(voxels[..., 3]) * t_delta)
        # exp(a + b) == exp(a) * exp(b)
        w_i = torch.cumprod(
            torch.cat(
                (torch.ones((*alpha_i.shape[:-1], 1), device=self.device), 1.0 - alpha_i + 1e-10),
                dim=-1
            ),
            dim=-1
        )[:, :-1]  
        w_i = alpha_i * w_i
        # [:, :-1]   [num_rays, num_samples + 1] => [num_rays, num_samples]
        rgb_map = torch.sum(
            w_i[..., None] * c_i, # [num_rays, num_samples, 1] * [num_rays, num_samples, 3]
            dim=-2,  # num_samples
            keepdim=False
        )  # [num_rays, 3]
        
        # Normalizing these weights as ˆwi = wi/∑Ncj=1 wj 
        # produces a piecewise-constant PDF along the ray. **5.2**
        pdf_map = w_i[..., 1:-1] + 1e-5  
        # prevent nans
        pdf_map = pdf_map / torch.sum(pdf_map, -1, keepdim=True)
        cdf_map = torch.cumsum(pdf_map, dim=-1)
        cdf_map = torch.cat(
            (torch.zeros_like(cdf_map[..., :1]), cdf_map), dim=-1
        )

        if self.background_w:
            rgb_map = rgb_map + (1.0 - torch.sum(w_i, dim=-1, keepdim=False)[..., None])

        return dict(
            rgb_map=rgb_map,
            cdf_map=cdf_map
        )

    def _hierarchical_sample(self, rays, t_vals: torch.Tensor, cdf_map: torch.Tensor):
        """
        Hierarchical volume sampling in paper
        =====================================
        **We sample a second set of Nf locations from this distribution
        using inverse transform sampling, evaluate our “fine” network at the union of the
        first and second set of samples, and compute the final rendered color of the ray Cf (r) using Eqn. 3 but using all Nc + Nf samples.**
        Follow the official approach. 
        Inputs:
            rays: (rays_o, rays_d) tuple[tensor, tensor]
            t_vals: tensor [num_rays, num_samples] t of the result of uniform sampling (the function `_sample`)
            cdf_map: tensor [num_rays, num_samples - 1] CDF map (Cumulative Distribution Function)
        Output:
            pos_locs: tensor [num_rays, num_samples + num_isamples, 3]  spatial locations used for the input of Fine Net
            t_vals: tensor [num_rays, num_samples + num_isamples] t of sampling and hierarchical sampling
        """
        rays_o, rays_d = rays
        t_vals_mid = (t_vals[..., :-1] + t_vals[..., 1:]) * 0.5 
        # [num_rays, num_samples - 1] == cdf_map.shape

        u = torch.rand((*cdf_map.shape[:-1], self.num_isamples), device=self.device).contiguous() # [num_rays, num_isamples]
        
        index = torch.searchsorted(cdf_map, u, right=True) # [num_rays, num_isamples] find > u
        index = torch.stack((
            torch.max(torch.zeros_like(index, device=self.device), index - 1), 
            torch.min(torch.full_like(index, fill_value=(cdf_map.shape[-1] - 1) * 1.0, device=self.device), index) 
            ), dim=-1) # [num_rays, num_isamples, 2]
        
        shape_m = [index.shape[0], index.shape[1], cdf_map.shape[-1]] # [num_rays, num_isamples, num_samples - 1]
        cdf_map_gather = torch.gather(cdf_map.unsqueeze(1).expand(shape_m), dim=2, index=index) # [num_rays, num_isamples, 2]
        t_vals_gather = torch.gather(t_vals_mid.unsqueeze(1).expand(shape_m), dim=2, index=index) # [num_rays, num_isamples, 2]
        
        denom = cdf_map_gather[..., 1] - cdf_map_gather[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=self.device), denom) # [num_rays, num_isamples]
        
        t_vals_fine = torch.lerp(t_vals_gather[..., 0], t_vals_gather[..., 1], (u - cdf_map_gather[..., 0]) / denom).detach()
        t_vals, _ = torch.sort(torch.cat((t_vals, t_vals_fine), dim=-1), dim=-1)

        dots_sh = list(rays_d.shape[:-1])
        
        bg_depth = torch.flip(t_vals, dims=[-1])

        bg_rays_o = rays_o.unsqueeze(-2).expand(dots_sh + [self.num_samples + self.num_isamples, 3])
        bg_rays_d = rays_d.unsqueeze(-2).expand(dots_sh + [self.num_samples + self.num_isamples, 3])
        
        bg_locs, _ = self._depth2pts_outside(bg_rays_o, bg_rays_d, bg_depth)

        return bg_locs, bg_depth

    def cast_rays(self, rays):
        """
        Rays Casting
        ============
        1. sample spatial locations and t for coarse net of NeRF
        2. using spatial locations and view directions as inputs of the coarse net of NeRF 
            to get voxels (rgb density) of the spatial locations
        3. parse voxels (rgb density) to rgb map of the scene (calculate integral) 
            and cdf of the sampling t
        4. hierarchical sample spatial locations and t for fine net of NeRF
        5. using spatial locations obtained by [4] and view directions as inputs of the fine net of NeRF 
            to get voxels (rgb density) of the spatial locations
        6. return the results (rgb map) of [2] and [5]
        Inputs:
            rays: (rays_o, rays_d) 
            view_dirs: tensor  view directions [num_rays, 3]
        Outputs:
            rgb map of [2] and [5]
        """
        rays_o, rays_d = rays
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        coarse_nerf_locs, t_coarse = self._sample((rays_o, rays_d))
        coarse_voxels = self._voxel_sample5d(
            torch.cat((coarse_nerf_locs,
                       view_dirs[..., None, :].expand((*coarse_nerf_locs.shape[:-1],3))), dim=-1),
            "coarse"
        )
        coarse_info = self._parse_voxels(coarse_voxels, t_coarse, rays_d)
        fine_nerf_locs, t_fine = self._hierarchical_sample(
            (rays_o, rays_d), t_coarse, coarse_info["cdf_map"]
        )
        fine_voxels = self._voxel_sample5d(
            torch.cat((fine_nerf_locs,
                       view_dirs[..., None, :].expand((*fine_nerf_locs.shape[:-1],3))), dim=-1),
            "fine"
        )
        fine_info = self._parse_voxels(fine_voxels, t_fine, rays_d)
        return coarse_info["rgb_map"], fine_info["rgb_map"]

    def _voxel_sample5d(self, x: torch.Tensor, net_type) -> torch.Tensor:
        """
        sample voxels (rgb density) using NeRF
        =======================================
        Inputs:
            x : spatial locations and view directions
            net_type: "coarse" or "fine" to select coarse net or fine net to forward
        Outputs:
            rgb + density
        """
        self.nerf.net_(net_type)
        if self.sample5d_chunk is None or self.sample5d_chunk <= 1:
            return self.nerf(x)
        return torch.cat(
            [self.nerf(x[i:i + self.sample5d_chunk])
             for i in range(0, x.shape[0], self.sample5d_chunk)],
            dim=0
        )
        

class VolumeRendererPlusPlus:
    
    def __init__(self, fg_nerf, bg_nerf, width, height, num_samples, num_isamples, background_w, ray_chunk, sample5d_chunk, is_train, device) -> None:
        self.fg_nerf = fg_nerf                          
        self.bg_nerf = bg_nerf
        self.width = width                      # width of the rendering scene
        self.height = height                    # height of the rendering scene
        self.num_samples = num_samples          # number of sampling
        self.num_isamples = num_isamples        # number of Hierarchical volume sampling
        self.ray_chunk = ray_chunk              # chunk of ray casting
        self.sample5d_chunk = sample5d_chunk    # chunk of net
        self.background_w = background_w        # whether or not transform image's background to white
        self.device = device                    # device of the whole volume renderer

        self.is_train = is_train
        
        self.fg_renderer = ForegroundRenderer(
            nerf=fg_nerf,
            width=width,
            height=height,
            num_samples=num_samples,
            num_isamples=num_isamples,
            background_w=background_w,
            ray_chunk=ray_chunk,
            sample5d_chunk=sample5d_chunk,
            is_train=is_train,
            device=device
        )
        
        self.bg_renderer = BackgroundRenderer(
            nerf=bg_nerf,
            width=width,
            height=height,
            num_samples=num_samples,
            num_isamples=num_isamples,
            background_w=background_w,
            ray_chunk=ray_chunk,
            sample5d_chunk=sample5d_chunk,
            is_train=is_train,
            device=device
        )
        

    def train(self, is_train):
        """ Train or Eval(just rendering or test) """
        self.is_train = is_train
        self.fg_renderer.train(is_train)
        self.bg_renderer.train(is_train)
        

    def _generate_rays(self, c2w, intrinsics):
        '''
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        '''
        H, W = self.height, self.width
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32) + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

        rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
        rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
        rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

        rays_o = c2w[:3, 3].reshape((1, 3))
        rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)
        
        rays_o = torch.tensor(rays_o, device=self.device).reshape(H, W, 3)
        rays_d = torch.tensor(rays_d, device=self.device).reshape(H, W, 3)

        # depth = np.linalg.inv(c2w)[2, 3]
        # depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)
        
        return rays_o, rays_d
    
    def _cast_rays(self, rays):
        fg_c_rgb, fg_f_rgb, fg_c_lambda, fg_f_lambda = self.fg_renderer.cast_rays(rays)
        bg_c_rgb, bg_f_rgb = self.bg_renderer.cast_rays(rays)
        bg_c_rgb = fg_c_lambda[..., None] * bg_c_rgb
        bg_f_rgb = fg_f_lambda[..., None] * bg_f_rgb
        return fg_c_rgb + bg_c_rgb, fg_f_rgb + bg_f_rgb
        
    
    def render(self, pose, intrinsics, select_coords=None):
        """
        render some pixels of the scene 
        (mainly used in training NeRF and `self.render_image`)
        Inputs:
            pose: camera pose (camera2world matrix)
            select_coords: some random pixels of the scene
        Outputs:
            rgb maps of coarse net and fine net 
        """
        rays_o, rays_d = self._generate_rays(pose, intrinsics)

        if select_coords is not None:
            rays_o = rays_o[select_coords[..., 0], select_coords[..., 1]]
            rays_d = rays_d[select_coords[..., 0], select_coords[..., 1]]

        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        
        if self.ray_chunk is None or self.ray_chunk <= 1:
            return self._cast_rays((rays_o, rays_d))

        coarse_result, fine_result = [], []

        for i in range(0, rays_o.shape[0], self.ray_chunk):
            j = i + self.ray_chunk

            c, f = self._cast_rays((rays_o[i:j], rays_d[i:j]))
            coarse_result.append(c)
            fine_result.append(f)

        return torch.cat(coarse_result, dim=0), torch.cat(fine_result, dim=0)

    def render_image(self, pose, intrinsic, render_batch_size=1024, use_tqdm=True) -> np.ndarray:
        """
        render a scene (image)
        ======================
        * work in eval state (just rendering not for training)
        * NumPy.NDArray => torch.Tensor(CPU) => torch.Tensor(GPU) => torch.Tensor(CPU) => NumPy.NDArray
        Inputs:
            pose                : NumPy.NDArray         camera pose (camera2world matrix)
            render_batch_size   : int                   batch size of rendering default is 1024 
            use_tqdm            : bool                  whether or not use `tqdm` module
        Outputs:
            img: NumPy.NDArray
        """
        self.train(False)
        img_block_list = []
        if use_tqdm:
            for epoch in tqdm.trange(0, self.height * self.width, render_batch_size):
                coords = get_screen_batch(self.height, self.width, render_batch_size, epoch, device=self.device)
                _, image_fine = self.render(pose, intrinsic, select_coords=coords)
                img_block_list.append(image_fine.detach().cpu())
        else:
            for epoch in range(0, self.height * self.width, render_batch_size):
                coords = get_screen_batch(self.height, self.width, render_batch_size, epoch, device=self.device)
                _, image_fine = self.render(pose, intrinsic, select_coords=coords)
                img_block_list.append(image_fine.detach().cpu())
        img = np.concatenate(img_block_list, axis=0)[:self.height * self.width].reshape(self.height, self.width, 3)
        return img