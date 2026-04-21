import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from pathlib import Path
import random, time, uuid, os
import numpy as np
import shutil
import cv2
system = random.SystemRandom()


class TexturedCube:
    def __init__(self, texture_path):
        self.vertices = [
            [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
            [1, -1, 1], [1, 1, 1], [-1, -1, 1], [-1, 1, 1]
        ]

        self.faces = [
            [0, 1, 2, 3],  # 后面
            [4, 5, 1, 0],  # 右面
            [7, 6, 4, 5],  # 前面
            [3, 2, 7, 6],  # 左面
            [1, 5, 7, 2],  # 顶面
            [6, 4, 0, 3]  # 底面
        ]

        # 纹理坐标
        self.tex_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]

        # 每个面的法线
        self.normals = [
            [0, 0, -1],  # 后面：朝向-Z
            [1, 0, 0],  # 右面：朝向+X
            [0, 0, 1],  # 前面：朝向+Z
            [-1, 0, 0],  # 左面：朝向-X
            [0, 1, 0],  # 顶面：朝向+Y
            [0, -1, 0]  # 底面：朝向-Y
        ]

    def draw(self, texture):
        """绘制带纹理的立方体（带法线）"""
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)

        for num, face in enumerate(self.faces):
            glBegin(GL_QUADS)

            # 设置这个面的法线
            glNormal3fv(self.normals[num])

            # 绘制四个顶点
            for j, vertex_index in enumerate(face):
                glTexCoord2f(*self.tex_coords[j])
                glVertex3fv(self.vertices[vertex_index])

            glEnd()

        glDisable(GL_TEXTURE_2D)

    def get_coordinates_and_visibility(self, camera_pos, position=(0, 0, 0), display_size=(640, 640)):
        """
        判定顶点坐标和可见性
        """
        w, h = display_size
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        camera_pos = np.array(camera_pos)
        pos = np.array(position)

        pixel_points = []
        visibility = []

        all_face_visible = []
        for i, face_idx in enumerate(self.faces):
            normal = np.array(self.normals[i])

            face_ref_point = np.array(self.vertices[face_idx[0]]) + pos
            view_vector = face_ref_point - camera_pos

            visible_face = np.dot(normal, view_vector) <= 0
            all_face_visible.append(visible_face)

        for i, v_local in enumerate(self.vertices):
            # 计算世界坐标
            world_v = np.array(v_local) + pos

            win_x, win_y, win_z = gluProject(world_v[0], world_v[1], world_v[2],
                                             modelview, projection, viewport)
            px, py = win_x, h - win_y

            if 0 <= win_x <= w and 0 <= win_y <= h:
                any_face_visible = False
                for face_i, face_indices in enumerate(self.faces):
                    if i in face_indices: # 如果点在该面中
                        if all_face_visible[face_i]: # 只要相连的面里有一个可见，就认为这个顶点可见
                            any_face_visible = True
                            break

                v_status = 2 if any_face_visible else 1
                pixel_points.append((px, py))
                visibility.append(v_status)
            else:
                pixel_points.append((px, py))
                visibility.append(0)

        return pixel_points, visibility


def setup_lighting():
    """设置光照参数 - 保证足够亮"""
    # 启用光照和深度测试
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    # 设置环境光亮度
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0, 0, 0, 1.0])

    # 启用4个光源
    for i in range(4):
        glEnable(GL_LIGHT0 + i)

    # 光源位置（放在物体周围，距离适中）
    position = random.uniform(-1, 1) * 2
    light_positions = [
        [3 +  position, 3, 3 +  position, 1.0],  # 光源0：右前上
        [-3 +  position, 3, 3 +  position, 1.0],  # 光源1：左前上
        [3 +  position, 3, -3 +  position, 1.0],  # 光源2：右前下
        [-3 +  position, 3, -3 +  position, 1.0],  # 光源3：左前下
    ]

    # 光源颜色 - 提高亮度！
    ambient = [0.6, 0.6, 0.6, 1.0]  # 环境光
    diffuse = [0.45, 0.45, 0.45, 1.0]  # 漫反射设为最亮！
    specular = [0.2, 0.2, 0.2, 1.0]  # 镜面反射提高

    for num, pos in enumerate(light_positions):
        light = GL_LIGHT0 + num
        glLightfv(light, GL_POSITION, pos)
        glLightfv(light, GL_AMBIENT, ambient)
        glLightfv(light, GL_DIFFUSE, diffuse)
        glLightfv(light, GL_SPECULAR, specular)

        glLightf(light, GL_CONSTANT_ATTENUATION, 0.05)
        glLightf(light, GL_LINEAR_ATTENUATION, 0.015)
        glLightf(light, GL_QUADRATIC_ATTENUATION, 0.03)

    # 设置材质属性
    glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE, [0.2, 0.2, 0.2, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 75.0)

def draw_ground():
    color_lib = [[0.5661, 0.6510, 0.3137],
                 [0.1647, 0.4431, 0.2196],
                 [0.1608, 0.3216, 0.0627]]
    color = random.choice(color_lib)

    # 设置地面材质
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [c * 0.4 for c in color] + [1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 20.0)

    # 法线朝上（Y轴正方向）
    glNormal3f(0, 1, 0)

    # 绘制地面（放在Y=-1处，立方体下方）
    glBegin(GL_QUADS)
    glVertex3f(-5, -1, -5)
    glVertex3f(5, -1, -5)
    glVertex3f(5, -1, 5)
    glVertex3f(-5, -1, 5)
    glEnd()

def pre_load_all_textures(img_paths):
    """
    一次性加载所有贴图到显存
    """
    tex_id_map = {}

    for class_id, img_p in enumerate(img_paths):
        image_path = str(img_p)
        try:
            # 1. 加载图片
            textureSurface = pygame.image.load(image_path)
            # 2. 转换数据（True 表示翻转，适配 OpenGL 坐标系）
            textureData = pygame.image.tostring(textureSurface, "RGBA", True)
            width = textureSurface.get_width()
            height = textureSurface.get_height()

            # 3. 创建 OpenGL 纹理
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)

            # 设置数据传输方式
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)

            # 设置拉伸过滤参数
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # 存入字典
            tex_id_map[class_id] = tex_id
            print(f"成功预加载类别 {class_id}: {img_p.stem}")

        except Exception as e:
            print(f"警告：无法加载贴图 {image_path}, 错误: {e}")

    return tex_id_map

def get_true_pnp(eye, center, up):
    f_cv = center - eye
    f_cv = f_cv / np.linalg.norm(f_cv)
    s_cv = np.cross(f_cv, up)
    s_cv = s_cv / np.linalg.norm(s_cv)
    u_cv = - np.cross(s_cv, f_cv)
    u_cv = u_cv / np.linalg.norm(u_cv)

    R = np.array([
        [s_cv[0], s_cv[1], s_cv[2]],
        [u_cv[0], u_cv[1], u_cv[2]],
        [f_cv[0], f_cv[1], f_cv[2]]
    ])

    tvec = R.dot(- eye)
    rvec, _ = cv2.Rodrigues(R)

    return rvec, tvec

def generate_dataset(num_per_texture=5):
    base_dir = Path("../textures/KFS")
    output_dir = Path("mono_cube_dataset_vec")
    img_save_dir = output_dir / "images"
    lbl_save_dir = output_dir / "labels"
    vec_dir = output_dir / "vectors"

    if output_dir.exists():
        print(f"已清理旧数据: {output_dir}")
        shutil.rmtree(output_dir)

    for p in [img_save_dir, lbl_save_dir, vec_dir]: p.mkdir(parents=True, exist_ok=True)

    extensions = {'.png'}

    img_paths = [
        p for p in Path(base_dir).rglob("*")
        if p.suffix.lower() in extensions
    ]

    img_paths.sort()

    image_dict = {i: p.stem for i, p in enumerate(img_paths)}
    # class_ids = image_dict.keys()
    # class_names = image_dict.values()

    # 初始化一次 Pygame 窗口
    pygame.init()
    display_size = (640, 640)

    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    pygame.display.set_mode(display_size, DOUBLEBUF | OPENGL)

    cube = TexturedCube(img_paths)
    texture = pre_load_all_textures(img_paths)

    for class_id, img_p in enumerate(img_paths):
        tex = texture[class_id]
        cnt = 0

        while cnt < num_per_texture:
            pygame.event.pump() # 防止卡死
            # 渲染逻辑
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, 1.0, 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # 随机相机位姿
            eye_x = random.uniform(-10, 10)
            eye_y = random.uniform(0, 6) + 3
            eye_z = random.uniform(0, 10) + 2

            center_x, center_y, center_z = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)

            up_x, up_y, up_z = 0, 1, 0

            gluLookAt(eye_x, eye_y, eye_z,
                      center_x, center_y, center_z,
                      up_x, up_y, up_z)

            eye = np.array((eye_x, eye_y, eye_z), dtype=float)
            center = np.array((center_x, center_y, center_z), dtype=float)
            up = np.array((up_x, up_y, up_z), dtype=float)

            rvec, tvec = get_true_pnp(eye, center, up)
            vectors_str = [" ".join(map(str, np.array(v).ravel())) for v in [rvec, tvec]]

            setup_lighting() # 设置光照
            glClearColor(random.random() * 0.8, random.random() * 0.8, random.random() * 0.8, 1.0) # 背景颜色
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # 绘制KFS和地面
            cube.draw(tex)
            draw_ground()

            # 获取标签信息
            coords, vis = cube.get_coordinates_and_visibility(eye)

            vis_cnt = 0
            for point in vis:
                if point != 0:
                    vis_cnt += 1

            if vis_cnt < 4: continue
            else: cnt += 1

            # YOLO标签处理
            pts_array = np.array(coords)
            x_min, y_min = np.min(pts_array[:, 0]), np.min(pts_array[:, 1])  # 取第一列和第二列
            x_max, y_max = np.max(pts_array[:, 0]), np.max(pts_array[:, 1])

            w, h = display_size

            x_min_clipped = max(0, min(w, x_min))
            x_max_clipped = max(0, min(w, x_max))
            y_min_clipped = max(0, min(h, y_min))
            y_max_clipped = max(0, min(h, y_max))

            xc = ((x_min_clipped + x_max_clipped) / 2) / w
            yc = ((y_min_clipped + y_max_clipped) / 2) / h
            bw = (x_max_clipped - x_min_clipped) / w
            bh = (y_max_clipped - y_min_clipped) / h

            label_line = f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" # 写入bbox
            for (px, py), v in zip(coords, vis):
                if not(0 <= px <= w and 0 <= py <= h):
                    px, py = 0.0, 0.0
                label_line += f" {px / w:.6f} {py / h:.6f} {v}" # 写入每个顶点坐标

            glReadBuffer(GL_BACK)
            # time.sleep(0.5)
            data = glReadPixels(0, 0, 640, 640, GL_RGB, GL_UNSIGNED_BYTE)
            surface = pygame.image.fromstring(data, (640, 640), 'RGB')
            surface = pygame.transform.flip(surface, False, True)
            pygame.display.flip()

            # 写入和保存
            file_id = f"{uuid.uuid4().hex[:8]}"
            if class_id < 10:
                img_filename = f"0{class_id}_{img_p.stem}_{file_id}.png" # 保存图片
                with open(lbl_save_dir / f"0{class_id}_{img_p.stem}_{file_id}.txt", "w") as f:  # 写入标签文件
                    f.write(label_line)
                with open(vec_dir / f"0{class_id}_{img_p.stem}_{file_id}.txt", "w") as f:  # 写入vec文件
                    f.write("\n".join(vectors_str))
            else:
                img_filename = f"{class_id}_{img_p.stem}_{file_id}.png"
                with open(lbl_save_dir / f"{class_id}_{img_p.stem}_{file_id}.txt", "w") as f:  # 写入标签文件
                    f.write(label_line)
                with open(vec_dir / f"{class_id}_{img_p.stem}_{file_id}.txt", "w") as f:  # 写入vec文件
                    f.write("\n".join(vectors_str))

            pygame.image.save(surface, str(img_save_dir / img_filename))

    print(f"合成完毕！数据保存在: {output_dir.absolute()}")

if __name__ == "__main__":
    generate_dataset(num_per_texture=20)