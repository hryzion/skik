import json
import numpy as np
import os


object_base_dir = r"D:\zhx_workspace\3DScenePlatformDev\dataset\object"

def load_json(file_path):
    with open(file_path, 'r') as f:
        scene_data = json.load(f)
    return scene_data


def operate_vertex(vertex,translate,scale,rotate,rotateorder,orient):
    vertex = np.array(vertex) * np.array(scale)

    rx, ry, rz = rotate
    R={}
    R['Z'] = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])
    
    R['Y'] = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0,          1, 0         ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    R['X'] = np.array([[1, 0,            0           ],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    
    rotation_matrix = R[rotateorder[2]] @ R[rotateorder[1]] @ R[rotateorder[0]]
    vertex = rotation_matrix @ vertex

    # orient_matrix=np.array([[np.cos(orient), -np.sin(orient), 0],
    #                [np.sin(orient),  np.cos(orient), 0],
    #                [0,           0,          1]])
    # vertex = orient_matrix @ vertex

    vertex = vertex + np.array(translate)
    return vertex.tolist()

def operate_vn(vn,translate,scale,rotate,rotateorder,orient):
    rx, ry, rz = rotate
    R={}
    R['Z'] = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])
    
    R['Y'] = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0,          1, 0         ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    R['X'] = np.array([[1, 0,            0           ],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    
    rotation_matrix = R[rotateorder[2]] @ R[rotateorder[1]] @ R[rotateorder[0]]
    vn = rotation_matrix @ vn

    # orient_matrix=np.array([[np.cos(orient), -np.sin(orient), 0],
    #                [np.sin(orient),  np.cos(orient), 0],
    #                [0,           0,          1]])
    # vn = orient_matrix @ vn

    return vn.tolist()

def readmodel(model_path,vertex_cnt,vt_cnt,vn_cnt,translate,scale,rotate,rotateorder,orient):
    v = []
    vn = []
    vt = []
    f = []
    use_mtl=[]
    cnt1=0
    cnt2=0
    cnt3=0
    fcnt=0

    with open(model_path, 'r') as file:
        for line in file:
            if line.startswith('#') or line.startswith('m'):
                #m对应mtllib .mtl这类行 
                continue
            line=line.strip()  
            if line.startswith('u'):
                #对应usemtl行,fcnt用于记录此usemtl应用于哪些面
                use_mtl.append((line,fcnt))

            parts = line.split()
            if len(parts)==0 or len(parts)==1:
                continue
            if parts[0] == 'v':
                vertex = list(map(float, parts[1:4]))
                vertex=operate_vertex(vertex,translate,scale,rotate,rotateorder,orient)
                v.append(vertex)
                cnt1+=1
            elif parts[0] == 'vn':
                normal = list(map(float, parts[1:4]))
                normal=operate_vn(normal,translate,scale,rotate,rotateorder,orient)
                vn.append(normal)
                cnt3+=1
            elif parts[0] == 'vt':
                tex_coord = list(map(float, parts[1:3]))
                vt.append(tex_coord)
                cnt2+=1
            elif parts[0] == 'f':
                face = []
                fcnt+=1
                for vertex in parts[1:]:
                    indices = [int(x) for x in vertex.split('/')]
                    face.append((indices[0]+vertex_cnt,indices[1]+vt_cnt,indices[2]+vn_cnt))
                f.append(face)
    file.close()

    model={}
    model['v']=v
    model['vn']=vn
    model['vt']=vt
    model['f']=f
    model['use_mtl']=use_mtl
    return model,cnt1,cnt2,cnt3

def combine_mtl(mid_list):
    totline=[]
    for mid in mid_list:
        mtl_path=os.path.join(object_base_dir,mid,mid+".mtl")
        with open(mtl_path,'r') as f:
            for line in f:
                totline.append(line.strip()+'\n')
        totline.append('\n')
    with open(".\\total.mtl",'w') as f:
        for line in totline:
            f.write(line)

def create_roomobj(r,vertex_cnt,vt_cnt,vn_cnt):
    #还没加vertexcnt
    roomshape=r["roomShape"]
    vertices = []
    faces = []
    height = 2.8 
    cnt1=0
    cnt2=0
    cnt3=0
    l1=[]
    l2=[]

    for point in roomshape:
        x, z = point
        vertices.append([x,0,z])
        cnt1+=1
        l1.append(cnt1+vertex_cnt)
        vertices.append([x, height, z])
        cnt1+=1
        l2.append(cnt1+vertex_cnt)

    #加入顶面和底面
    faces.append([(x,) for x in l1])
    faces.append([(x,) for x in l2])

    for i in range(len(roomshape)):
        #加入侧面
        next_index = (i + 1) % len(roomshape)
        
        v1 = 2 * i       
        v2 = 2 * next_index  
        v3 = v2 + 1      
        v4 = v1 + 1      
        #侧面一定是四边形，就不分解成三角形了
        faces.append([(v1 + 1+vertex_cnt,), (v2 + 1+vertex_cnt,), (v3 + 1+vertex_cnt,), (v4+1+vertex_cnt,)])  

    model={}
    model["use_mtl"]=""
    model["v"]=vertices
    model['vn']=[]
    model['vt']=[]
    model['f']=faces
    return model,cnt1,cnt2,cnt3

def write_obj(model_list,filename):
    # 整合所有model中元素，输出到total.obj文件中
    # 注意不要用连等号初始化
    vlist=[]
    vnlist=[]
    vtlist=[]
    flist=[]
    for model in model_list:
        vlist+=model['v']
        vnlist+=model['vn']
        vtlist+=model['vt']
        flist+=model['f']

    with open(filename,'w') as file:
        file.write("mtllib total.mtl\n")
        
        for v in vlist:
            file.write('v ')
            file.write(' '.join(map(str,v)))
            file.write('\n')
        for vn in vnlist:
            file.write('vn ')
            file.write(' '.join(map(str,vn)))
            file.write('\n')
        for vt in vtlist:
            file.write('vt ')
            file.write(' '.join(map(str,vt)))
            file.write('\n')
        for f in flist:
            file.write('f ')
            for ele in f:
                if len(ele)==1:
                    file.write(str(ele[0]))
                else:
                    file.write('/'.join(map(str,ele)))
                file.write(' ')
            file.write('\n')

def operate(file_path):
    data=load_json(file_path)
    roomList=data.get("rooms")

    mid_list=[]
    model_list=[]
    wall_model_list=[]
    vertex_cnt=0
    vt_cnt=0
    vn_cnt=0
    wall_vertex_cnt=0
    wall_vt_cnt=0
    wall_vn_cnt=0

    for r in roomList:
        # 根据roomshape等参数构造room obj
        model,cnt1,cnt2,cnt3=create_roomobj(r,wall_vertex_cnt,wall_vt_cnt,wall_vn_cnt)
        wall_model_list.append(model)
        wall_vertex_cnt+=cnt1
        wall_vt_cnt+=cnt2
        wall_vn_cnt+=cnt3

        #处理room.objlist中的物体
        objlist=r["objList"]
        for o in objlist:
            if o["inDatabase"]:
                #获取变换数据
                translate=o["translate"]
                scale=o["scale"]
                rotate=o["rotate"]
                rotateorder=o["rotateOrder"]
                orient=o["orient"]

                #读取在数据库内的obj文件，对顶点编号、参数作相应处理
                mid=o["modelId"]
                mid_list.append(mid)
                
                model={}
                model_path=os.path.join(object_base_dir,mid,mid+".obj")
                model,cnt1,cnt2,cnt3=readmodel(model_path,vertex_cnt,vt_cnt,vn_cnt,translate,scale,rotate,rotateorder,orient)
                vertex_cnt+=cnt1
                vt_cnt+=cnt2
                vn_cnt+=cnt3
                model_list.append(model)
    
    #将所有model的mtl文件整合，方便引用
    combine_mtl(mid_list)

    write_obj(model_list,"model.obj")
    write_obj(wall_model_list,"wall.obj")


if __name__ == '__main__':
    operate(rf"D:\zhx_workspace\3DScenePlatformDev\dataset\Levels2021\000ecb5b-b877-4f9a-ab6f-90f385931658.json")