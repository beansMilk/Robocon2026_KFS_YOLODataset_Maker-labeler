# 此脚本用于生成随机角度KFS的数据集（yolo-pose)
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from pathlib import Path
import random, time, uuid, os
import numpy as np
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

    def draw(self, texture_selected, position = (0, 0, 0)):
        """绘制带纹理的立方体（带法线）"""
        glEnable(GL_TEXTURE_2D)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1, 0.1, 0.1, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 10.0)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        glPushMatrix()

        glTranslatef(position[0], position[1], position[2])

        glBindTexture(GL_TEXTURE_2D, texture_selected)

        # 绘制每个面
        for num, face in enumerate(self.faces):
            glBegin(GL_QUADS)

            # 设置这个面的法线
            glNormal3fv(self.normals[num])

            # 绘制四个顶点
            for j, vertex_index in enumerate(face):
                glTexCoord2f(*self.tex_coords[j])
                glVertex3fv(self.vertices[vertex_index])

            glEnd()

        glPopMatrix()

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

        # 2. 遍历每个顶点，判定其可见性状态
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

def pre_load_all_textures(img_paths, num_per_texture=6):
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

    texture_pool = []
    for c_id, t_id in tex_id_map.items():
        texture_pool.extend([(c_id, t_id)] * num_per_texture)
    random.shuffle(texture_pool)

    return texture_pool


def setup_lighting():
    """设置光照参数 - 增加随机熄灯增强"""
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.1, 0.1, 0.1, 1.0])

    # 1. 核心修改：决定哪些灯是亮着的
    # 总共 4 盏灯 (0,1,2,3)，随机熄灭 1 到 2 盏，剩下的就是亮着的
    num_to_turn_off = random.randint(0, 2)
    off_lights = random.sample(range(4), num_to_turn_off)

    light_positions = [
        [8, random.randint(7, 10), -2, 1.0],
        [-8, random.randint(7, 10), -2, 1.0],
        [8, random.randint(7, 10), 12, 1.0],
        [-8, random.randint(7, 10), 12, 1.0],
    ]

    for i in range(4):
        light_id = GL_LIGHT0 + i
        if i in off_lights:
            glDisable(light_id)  # 熄灭选中的灯
        else:
            glEnable(light_id)  # 开启剩余的灯

            # 只有开启的灯才需要配置参数
            light_diffuse = random.uniform(0.7, 1.0)
            diffuse = [light_diffuse, light_diffuse, light_diffuse, 1.0]

            glLightfv(light_id, GL_POSITION, light_positions[i])
            glLightfv(light_id, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
            glLightfv(light_id, GL_DIFFUSE, diffuse)
            glLightfv(light_id, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])

            # 设置衰减，模拟真实光线感
            glLightf(light_id, GL_CONSTANT_ATTENUATION, 1)
            glLightf(light_id, GL_LINEAR_ATTENUATION, 0.01)
            glLightf(light_id, GL_QUADRATIC_ATTENUATION, 0.005)

    # 设置材质属性
    glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE, [0.2, 0.2, 0.2, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 85.0)

def draw_ground(position=(0, -1, 0)):
    color_lib = [[0.5661, 0.6510, 0.3137],
                 [0.1647, 0.4431, 0.2196],
                 [0.1608, 0.3216, 0.0627]]
    color = random.choice(color_lib)
    size = 10

    # 设置地面材质
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [c * 0.4 for c in color] + [1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.6, 0.6, 0.6, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 30.0)

    # 法线朝上（Y轴正方向）
    glNormal3f(0, 1, 0)

    # 绘制地面（放在Y=-1处，立方体下方）
    glBegin(GL_QUADS)
    glVertex3f(-size//2+position[0], position[1]-1, size//2+position[2])
    glVertex3f(size//2+position[0], position[1]-1, size//2+position[2])
    glVertex3f(size//2+position[0], position[1]-1, -size//2+position[2])
    glVertex3f(-size//2+position[0], position[1]-1, -size//2+position[2])
    glEnd()

def generate_dataset(num_per_texture=6):
    base_dir = Path("../textures/KFS")
    output_dir = Path("multi_cube_dataset_g2")
    img_save_dir = output_dir / "images"
    lbl_save_dir = output_dir / "labels"

    print(f"合成数量：{num_per_texture // 6 * 62}张")

    for p in [img_save_dir, lbl_save_dir]: p.mkdir(parents=True, exist_ok=True)

    extensions = {'.png'}

    img_paths = [
        p for p in Path(base_dir).rglob("*")
        if p.suffix.lower() in extensions
    ]

    img_paths.sort()

    position = [(0, 0, 0), (10, 1.143, 0), (-10, 1.143, 0), (0, 1.143, 10), (10, 0, 10), (-10, 2.286, 10)]

    # 初始化一次 Pygame 窗口
    pygame.init()
    display_size = (640, 640)
    pygame.display.set_mode(display_size, DOUBLEBUF | OPENGL)

    # 设置OpenGL视图
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 50.0)
    glEnable(GL_MULTISAMPLE)

    # GL_MULTISAMPLEBUFFERS 设为 1 启用多重采样
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    # GL_MULTISAMPLESAMPLES 设为 4 (4倍采样)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    pygame.display.set_mode(display_size, DOUBLEBUF | OPENGL)

    cube = TexturedCube(img_paths)
    texture_pool = pre_load_all_textures(img_paths, num_per_texture)
    process_flag = 0

    while process_flag < len(texture_pool):
        # 渲染逻辑
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1.0, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 随机相机位姿
        eye_x = random.uniform(-15, 15)
        eye_y = random.uniform(3, 15)
        eye_z = random.uniform(-10, 0) + 3

        if -8 < eye_x < 8:
            eye_y = random.uniform(3, 7)
            eye_z = random.uniform(-5, -13)

        gluLookAt(eye_x, eye_y, eye_z,
                  random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3),
                  0, 1, 0)
        eye_pos = (eye_x, eye_y, eye_z)

        setup_lighting() # 设置光照
        glClearColor(random.random() * 0.8, random.random() * 0.8, random.random() * 0.8, 1.0) # 背景颜色
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        all_labels = []

        for j in range(len(position)):
            if process_flag >= len(texture_pool):
                print("贴图已耗尽")
                break

            pygame.event.pump()

            # 绘制KFS和地面
            flag_0 = process_flag
            cube.draw(texture_pool[flag_0][1], position[j])
            draw_ground(position[j])

            # 获取标签信息
            coords, vis = cube.get_coordinates_and_visibility(eye_pos, position[j])

            vis_cnt = 0
            for point in vis:
                if point != 0:
                    vis_cnt += 1

            if vis_cnt < 4: continue

            process_flag += 1

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

            label_line = f"{texture_pool[flag_0][0]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" # 写入bbox

            for (px, py), v in zip(coords, vis):
                if not(0 <= px <= w and 0 <= py <= h):
                    px, py = 0.0, 0.0
                label_line += f" {px / w:.6f} {py / h:.6f} {v}" # 写入每个顶点坐标

            all_labels.append(label_line)

        final_content = "\n".join(all_labels)

        glFlush()
        glReadBuffer(GL_BACK)
        data = glReadPixels(0, 0, 640, 640, GL_RGB, GL_UNSIGNED_BYTE)
        surface = pygame.image.fromstring(data, (640, 640), 'RGB')
        surface = pygame.transform.flip(surface, False, True)
        pygame.display.flip()

        # time.sleep(1)

        # 写入和保存
        file_id = f"{uuid.uuid4().hex[:8]}"
        img_filename = f"{file_id}.png" # 保存图片
        pygame.image.save(surface, str(img_save_dir / img_filename))

        with open(lbl_save_dir / f"{file_id}.txt", "w") as f: # 写入标签文件
            f.write(final_content)

    pygame.quit()
    print(f"合成完毕！数据保存在: {output_dir.absolute()}")

if __name__ == "__main__":
    generate_dataset(num_per_texture=6* 5)  # 填入6的倍数（每张图六个KFS）