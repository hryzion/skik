from utils import *
import torch

view = {
    'position' : [[-4.188032002484747, 2.414213562373095, 1.4683720691864368]],
    'center' : [[-5.602245564857842, 1.0, 1.4683720691864368]],
    "fov" : 90
}


def transform_pos(mtx, pos): 
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx #4x4
    if len(t_mtx.shape)==2:
        t_mtx = t_mtx[None]
    # (x,y,z) -> (x,y,z,1)
    if pos.shape[-1]==3:
        posw = torch.nn.functional.pad(pos, pad=(0,1), mode='constant', value=1.0)
    else:
        posw=pos
    if len(t_mtx.shape)==2:
        return torch.matmul(posw, t_mtx.t())[None,...]
    else:
        return torch.matmul(posw, t_mtx.permute(0,2,1))

vmtx = get_camera_matrix(view,'cpu', aspect= 3/2)
print(vmtx['camera_mtx'])
print(vmtx['perspective_mtx'])
print(vmtx['view_mtx'])

points = torch.tensor(
        [[-4.28619073 , 0.   ,       1.84198066],
        [-5.33618927 , 0.    ,      1.84198066],
        [-5.33618927  ,0.   ,       0.79197934],
        [-4.28619073  ,0.     ,     0.79197934],
        [-4.28619073 , 0.442863 ,   1.84198066],
        [-5.33618927  ,0.442863 ,   1.84198066],
        [-5.33618927  ,0.442863 ,   0.79197934],
        [-4.28619073  ,0.442863,    0.79197934]]
    )

pt_cl = transform_pos(vmtx['view_mtx'],points)

print(pt_cl)

