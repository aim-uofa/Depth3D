import torch
import torch.nn.functional as F

pixel_coords = None

def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords.double()).reshape(b, 3, h, w)
    out = depth.unsqueeze(1) * cam_coords
    return out


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    angle = angle.double()
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zeros = zeros.double()
    ones = ones.double()

    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


# def mat2euler(rotMat):
#     '''
#     convert rotation_matrix to vector
#     rotMat: [B, 3, 3] 
#     '''
#     rotMat = rotMat.double()
#     b = rotMat.shape[0]
#     cy_thresh = torch.finfo(torch.double).eps * 4
#     (r11, r12, r13, r21, r22, r23, r31, r32, r33) = rotMat.contiguous().view(-1, 9).split(1, dim=1)
#     cy = torch.sqrt(r33*r33 + r23*r23)

#     angle_list = []
#     for i in range(b):
#         if cy[i] > cy_thresh: # cos(y) not close to zero, standard form
#                 z = torch.atan2(-r12[i],  r11[i]) # atan2(cos(y)*sin(z), cos(y)*cos(z))
#                 y = torch.atan2(r13[i],  cy[i]) # atan2(sin(y), cy)
#                 x = torch.atan2(-r23[i], r33[i]) # atan2(cos(y)*sin(x), cos(x)*cos(y))
#         else: # cos(y) (close to) zero, so x -> 0.0 (see above)
#                 # so r21 -> sin(z), r22 -> cos(z) and
#                 z = torch.atan2(r21[i],  r22[i])
#                 y = torch.atan2(r13[i],  cy[i]) # atan2(sin(y), cy)
#                 x = 0.0
        
#         # while not ((x >= -np.pi/2) and (x <= np.pi/2)):
#         #     if x < 0:
#         #         x += np.pi
#         #         y = np.pi-y
#         #     elif x > 0:
#         #         x -= np.pi
#         #         y = np.pi-y

#         # while not ((z >= -np.pi/2) and (z <= np.pi/2)):
#         #     if z < 0:
#         #         z += np.pi
#         #         y = np.pi-y
#         #     elif z > 0:
#         #         z -= np.pi
#         #         y = np.pi-y
        
#         # # limit rotation of y
#         # # while not ((y >= -np.pi/2) and (y <= np.pi/2)):
#         # while not ((x >= -np.pi/2) and (x <= np.pi/2)):
#         #     while not ((y >= -np.pi) and (y <= np.pi)):
#         #         if y < 0:
#         #             y += np.pi * 2
#         #         elif y > 0:
#         #             y -= np.pi * 2
#         #     y = np.pi - y

#         #     if (x + np.pi).abs() < (x - np.pi).abs():
#         #         x = (x + np.pi)
#         #     else:
#         #         x = (x - np.pi)
#         #     if (z + np.pi).abs() < (z - np.pi).abs():
#         #         z = (z + np.pi)
#         #     else:
#         #         z = (z - np.pi)


#         angle_list.append(torch.cat((x,y,z), dim=0))
    
#     angle = torch.stack(angle_list, dim=0)
    
#     return angle
    

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    vec = vec.double()
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

# def pose_mat2vec(transform_mat, rotation_mode='euler'):
#     transform_mat = transform_mat.double()
#     rot_mat = transform_mat[:, :3, :3] # [B, 3, 3]
#     translation = transform_mat[:, :3, 3] # [B, 3]

#     if rotation_mode == 'euler':
#         rot = mat2euler(rot_mat)  # [B, 3]
#     elif rotation_mode == 'quat':
#         assert False, 'not implemented...'
    
#     vec = torch.cat((translation, rot), dim=1)
#     return vec


# def pose_vec_inverse(pose_vec):

#     pose_mat = pose_vec2mat(pose_vec)
#     pose_ones = torch.tensor([[[0, 0, 0, 1]]]).float().cuda()
#     pose_mat = torch.cat((pose_mat, pose_ones), dim=1)
#     pose_inv_mat = pose_mat.inverse()
#     pose_inv_vec = pose_mat2vec(pose_inv_mat)

#     return pose_inv_vec


def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    # mask coords with depth=0
    X_norm[Z == proj_c2p_tr[:, 2]] = 2
    Y_norm[Z == proj_c2p_tr[:, 2]] = 2
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)


def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose)
    # pose_mat = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]).cuda()

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat.double()  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img.double(), src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    projected_depth = F.grid_sample(ref_depth.double(), src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    # plt.imsave('depth_tgt.png', depth[0, 0].detach().cpu().numpy(), cmap='rainbow')
    # plt.imsave('depth_ref.png', ref_depth[0, 0].detach().cpu().numpy(), cmap='rainbow')
    # # plt.imsave('depth_computed.png', computed_depth[0, 0].detach().cpu().numpy(), cmap='rainbow')
    
    # ref_img_warped_viz = projected_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    # plt.imsave('ref_img_warped_viz.png', ref_img_warped_viz)
    # assert False

    return projected_img, projected_depth, computed_depth


def inverse_rotation_warp(img, rot, intrinsics, padding_mode='zeros'):

    b, _, h, w = img.size()
    cam_coords = pixel2cam(torch.ones(b, h, w).type_as(img), intrinsics.inverse()) # [B,3,H,W]

    rot_mat = euler2mat(rot) # [B, 3, 3]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ rot_mat # [B, 3, 3]

    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, proj_cam_to_src_pixel, None, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=True)

    return projected_img

def grid_to_flow(grid):
    b, h, w, _ = grid.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(grid) # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(grid) # [1, H, W]
    image_coords = torch.stack((j_range, i_range), dim=1) # [1, 2, H, W]

    flow = torch.zeros_like(grid).type_as(grid)
    flow[:, :, :, 0] = (grid[:, :, :, 0]+1) / 2 * (w-1)
    flow[:, :, :, 1] = (grid[:, :, :, 1]+1) / 2 * (h-1)
    flow = flow.permute([0, 3, 1, 2])

    flow -= image_coords

    return flow

def compute_translation_flow(depth, pose, intrinsics):
    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse()) # [B,3,H,W]

    pose_mat = pose_vec2mat(pose) # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]

    grid_all, _ = cam2pixel2(cam_coords, rot, tr, padding_mode='zeros') # [B,H,W,2]
    grid_rot, _ = cam2pixel2(cam_coords, rot, None, padding_mode='zeros') # [B,H,W,2]

    flow_all = grid_to_flow(grid_all)
    flow_rot = grid_to_flow(grid_rot)
    flow_tr = (flow_all - flow_rot)

    return flow_tr


