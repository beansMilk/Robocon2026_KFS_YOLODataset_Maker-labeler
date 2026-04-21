import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from pathlib import Path
import random, time, uuid, os
import numpy as np
import shutil
import cv2
from OpenGL.GL import shaders
system = random.SystemRandom()


class TexturedCube:
    def __init__(self, texture_path):
        self.vertices = [
            [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
            [1, -1, 1], [1, 1, 1], [-1, -1, 1], [-1, 1, 1]
        ]

        self.faces = [
            [0, 3, 2, 1],  # 后面 (Back)
            [4, 0, 1, 5],  # 右面 (Right)
            [6, 4, 5, 7],  # 前面 (Front)
            [3, 6, 7, 2],  # 左面 (Left)
            [2, 7, 5, 1],  # 顶面 (Top)
            [0, 4, 6, 3]  # 底面 (Bottom)
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

        self.shader_program = create_shader_program()

    def draw(self, texture):
        glUseProgram(self.shader_program)

        # 1. 获取所有位置
        use_tex_loc = glGetUniformLocation(self.shader_program, "useTexture")
        tex_uni_loc = glGetUniformLocation(self.shader_program, "texture1")
        obj_col_loc = glGetUniformLocation(self.shader_program, "objectColor")
        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")

        # 2. 矩阵同步
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, np.identity(4, dtype=np.float32))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, modelview.astype(np.float32))
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.astype(np.float32))

        # 3. 纹理绑定
        glUniform1i(use_tex_loc, 1)  # 开启纹理
        glUniform1i(tex_uni_loc, 0)
        glUniform3f(obj_col_loc, 1, 1, 1)  # 基础色设为白色，不干扰纹理

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)

        for num, face in enumerate(self.faces):
            glBegin(GL_QUADS)
            glNormal3fv(self.normals[num])
            for j, vertex_index in enumerate(face):
                glTexCoord2f(*self.tex_coords[j])
                glVertex3fv(self.vertices[vertex_index])
            glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)

    def get_seg_coordinates(self, camera_pos, position=(0, 0, 0), display_size=(640, 640), threshold=-0.2):
        """
        判定可见面并输出符合 YOLO-seg 逻辑的顶点数组
        :param threshold: 可辨识阈值。dot_product 越小（越接近-1），表示面越正对着相机。
        """
        w, h = display_size
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        camera_pos = np.array(camera_pos)
        pos = np.array(position)

        # 1. 预先计算所有 8 个顶点的 2D 投影坐标
        all_projected_2d = []
        for v_local in self.vertices:
            world_v = np.array(v_local) + pos
            win_x, win_y, _ = gluProject(world_v[0], world_v[1], world_v[2],
                                         modelview, projection, viewport)
            all_projected_2d.append([win_x, h - win_y])  # 转换为图像坐标系

        visible_faces_pts = []

        # 2. 遍历面，判定可见性
        for i, face_indices in enumerate(self.faces):
            normal = np.array(self.normals[i])

            # 计算面中心的世界坐标（用于计算视线向量）
            face_center = np.mean([self.vertices[idx] for idx in face_indices], axis=0) + pos

            # 视线向量 v = 面中心 - 相机位置
            view_vector = face_center - camera_pos
            # 归一化视线向量，使阈值不受距离影响
            view_vector_norm = view_vector / np.linalg.norm(view_vector)

            # 计算法线与视线的点积
            # dot <= 0 表示面向相机；越接近 -1 表示越垂直
            dot_val = np.dot(normal, view_vector_norm)

            if dot_val < threshold:
                # 提取该面对应的 4 个 2D 顶点
                face_pts = [all_projected_2d[idx] for idx in face_indices]
                visible_faces_pts.append(face_pts)

        # 返回格式：可见面个数, ([面1顶点], [面2顶点], ...)
        return len(visible_faces_pts), tuple(visible_faces_pts)

    def setup_lighting(self, current_cam_pos, is_random=True):
        """
        所有赛场光照与材质参数
        """
        glUseProgram(self.shader_program)

        # 1. 获取变量位置
        light_pos_loc = glGetUniformLocation(self.shader_program, "lightPositions")
        light_color_loc = glGetUniformLocation(self.shader_program, "lightColors")
        view_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")
        shininess_loc = glGetUniformLocation(self.shader_program, "shininess")
        sat_loc = glGetUniformLocation(self.shader_program, "saturationFactor")
        spec_loc = glGetUniformLocation(self.shader_program, "specularStrength")
        num_lights_loc = glGetUniformLocation(self.shader_program, "numLights")

        # pos_shift_x = random.uniform(-3, 3)
        # pos_shift_z = random.uniform(-3, 3)
        location = np.random.uniform(-5, 5, size=(4, 2))

        light_pool = [
            {"pos": [location[0][0], 10, location[0][1]], "color": [0.8, 0.8, 0.8]},
            {"pos": [location[1][0], 10, location[1][1]], "color": [0.8, 0.8, 0.8]},
            {"pos": [location[2][0], 10, location[2][1]], "color": [0.8, 0.8, 0.8]},
            {"pos": [location[3][0], 10, location[3][1]], "color": [0.8, 0.8, 0.8]},
        ]

        # chosen = random.sample(light_pool, 2)

        pos_data = np.array([c["pos"] for c in light_pool], dtype=np.float32).flatten()
        col_data = np.array([c["color"] for c in light_pool], dtype=np.float32).flatten()

        glUniform3fv(light_pos_loc, 4, pos_data)
        glUniform3fv(light_color_loc, 4, col_data)
        glUniform1i(num_lights_loc, 4)

        if is_random:
            # 饱和度随机化
            rand_saturation = random.uniform(0.8, 1.2)
            glUniform1f(sat_loc, rand_saturation)

            # 高光强度随机化
            rand_specular = random.uniform(0.2, 0.5)
            glUniform1f(spec_loc, rand_specular)

            # 高光聚拢度随机化
            rand_shininess = random.uniform(20.0, 50.0)
            glUniform1f(shininess_loc, rand_shininess)
        else:
            # 非随机模式下保持原色
            glUniform1f(sat_loc, 1.0)

        # 视点位置
        glUniform3f(view_pos_loc, current_cam_pos[0], current_cam_pos[1], current_cam_pos[2])

from PIL import Image


class BackgroundManager:
    def __init__(self, folder_path):
        path_obj = Path(folder_path)
        self.image_files = [
            str(p) for p in path_obj.rglob('*')
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ]
        if not self.image_files:
            raise FileNotFoundError("未找到图片。")

        self.background_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.background_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # 解决非4倍数宽度图片崩溃
        glBindTexture(GL_TEXTURE_2D, 0)

        self.background_shader_program = create_background_shader()

    def load_background(self):
        # 更新已有ID内容
        img_path = random.choice(self.image_files)

        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                img_data = np.array(img, dtype=np.uint8)
                width, height = img.size

            # 绑定同一个 ID，覆盖旧数据
            glBindTexture(GL_TEXTURE_2D, self.background_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, img_data)

            glFinish()

            glBindTexture(GL_TEXTURE_2D, 0)

        except Exception as e:
            print(f"图片加载失败: {e}")

def create_shader_program():
    """
    着色器
    """
    vertex_shader_source = """
    #version 330 compatibility

    out vec3 FragPos;
    out vec2 TexCoord;
    out vec3 Normal;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        // gl_Vertex: 自动接收 glVertex3fv 的数据
        FragPos = vec3(model * gl_Vertex);

        // gl_Normal: 自动接收 glNormal3fv 的数据
        Normal = mat3(transpose(inverse(model))) * gl_Normal;

        // gl_MultiTexCoord0: 自动接收 glTexCoord2f 的数据
        TexCoord = gl_MultiTexCoord0.xy;

        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
    """

    fragment_shader_source = """
        #version 330 compatibility
        out vec4 FragColor;

        in vec3 FragPos;
        in vec2 TexCoord;
        in vec3 Normal;

        uniform sampler2D texture1;
        uniform vec3 viewPos;

        // --- 多光源 Uniform ---
        uniform vec3 lightPositions[5]; // 支持最多 5 个光源
        uniform vec3 lightColors[5];    // 每个光源的颜色
        uniform int numLights;          // 实际激活的光源数量

        uniform bool useTexture;
        uniform vec3 objectColor;

        // --- 调节参数 ---
        uniform float saturationFactor;
        uniform float specularStrength = 0.45;
        uniform float shininess = 40.0;
        uniform float ambientStrength = 0.3; // 基础环境光强度

        // --- 工具函数 1: RGB 转 HSL ---
        vec3 rgb2hsl(vec3 c) {
            vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
            vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
            vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
            float d = q.x - min(q.w, q.y);
            float e = 1.0e-10;
            return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
        }

        // --- 工具函数 2: HSL 转 RGB ---
        vec3 hsl2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        // --- 工具函数 3: 单个光源的光照计算 ---
        vec3 calculatePointLight(vec3 pos, vec3 color, vec3 normal, vec3 fragPos, vec3 viewDir) {
            // 1. 漫反射
            vec3 lightDir = normalize(pos - fragPos);
            float diff = max(dot(normal, lightDir), 0.0);
            vec3 diffuse = diff * color;

            // 2. 高光 (Blinn-Phong)
            vec3 halfwayDir = normalize(lightDir + viewDir);
            float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
            vec3 specular = specularStrength * spec * color;

            // 3. 简单的距离衰减 (可选，防止多光源叠加强度过大)
            float distance = length(pos - fragPos);
            float attenuation = 1.0 / (1.0 + 0.04 * distance + 0.0045 * (distance * distance));

            return (diffuse + specular) * attenuation;
        }

        void main() {
            // --- 1. 基础准备 ---
            vec3 norm = normalize(Normal);
            vec3 viewDir = normalize(viewPos - FragPos);

            // 初始光照累加值 (环境光只计算一次基础值)
            // 这里使用第一个光源颜色作为环境光基准，或统一使用白光
            vec3 totalLighting = vec3(ambientStrength);

            // --- 2. 遍历累加所有光源 ---
            for(int i = 0; i < numLights; i++) {
                totalLighting += calculatePointLight(lightPositions[i], lightColors[i], norm, FragPos, viewDir);
            }

            // --- 3. 采样纹理并调节饱和度 ---
            vec4 baseColor;
            if (useTexture) {
                vec4 texColor = texture(texture1, TexCoord);
                vec3 hsl = rgb2hsl(texColor.rgb);
                hsl.y = clamp(hsl.y * saturationFactor, 0.0, 1.0);
                baseColor = vec4(hsl2rgb(hsl), texColor.a);
            } else {
                baseColor = vec4(objectColor, 1.0);
            }

            // --- 4. 最终颜色合成 ---
            // 将累加后的总光照乘以物体的固有色
            vec3 result = totalLighting * baseColor.rgb;
            FragColor = vec4(result, baseColor.a);
        }
    """

    vs = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
    fs = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    return shaders.compileProgram(vs, fs)

def create_background_shader():
    vertex_source = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        TexCoord = aTexCoord;
        gl_Position = vec4(aPos, 0.999, 1.0); // 放在远平面附近
    }
    """
    fragment_source = """
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;
    uniform sampler2D bgTexture;
    void main() {
        FragColor = texture(bgTexture, TexCoord);
    }
    """

    vs = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
    fs = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
    return shaders.compileProgram(vs, fs)

def draw_ground(shader_program):
    """
    保留原有随机颜色逻辑，并适配 Shader 渲染环境
    """
    # 1. 保留原有的颜色库随机逻辑
    color_lib = [[0.5661, 0.6510, 0.3137],
                 [0.1647, 0.4431, 0.2196],
                 [0.1608, 0.3216, 0.0627]]
    color = random.choice(color_lib)

    # 2. 激活 Shader 并设置地面参数
    glUseProgram(shader_program)

    # 获取 Uniform 位置
    use_tex_loc = glGetUniformLocation(shader_program, "useTexture")
    obj_col_loc = glGetUniformLocation(shader_program, "objectColor")
    shine_loc = glGetUniformLocation(shader_program, "shininess")

    # --- 关键设置 ---
    glUniform1i(use_tex_loc, 0)  # 不使用贴图，使用纯色
    glUniform3f(obj_col_loc, *color)  # 传入随机选中的颜色
    glUniform1f(shine_loc, 1000.0)  # 极大值消除地面高光，保持哑光感

    # 3. 同步矩阵
    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    proj_loc = glGetUniformLocation(shader_program, "projection")

    view_mat = glGetDoublev(GL_MODELVIEW_MATRIX)
    proj_mat = glGetDoublev(GL_PROJECTION_MATRIX)

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, np.identity(4, dtype=np.float32))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_mat.astype(np.float32))
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj_mat.astype(np.float32))

    # 4. 执行绘制
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)

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
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

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

def random_spherical_coords(r_range=(4, 6), theta_range=(40, 100), phi_range=(-45, 45)): # 输入为角度值
    r = random.uniform(*r_range)

    theta_min_rad = np.radians(theta_range[0])
    theta_max_rad = np.radians(theta_range[1])

    cos_min = np.cos(theta_min_rad)
    cos_max = np.cos(theta_max_rad)

    random_cos = random.uniform(cos_max, cos_min)
    theta_rad = np.arccos(random_cos)

    phi_rad = np.radians(random.uniform(*phi_range))

    return r, theta_rad, phi_rad

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(theta) * np.sin(phi)
    y = r * np.cos(theta)
    return x, y, z

def setup_bg_quad():
    # 坐标范围 -1 到 1，纹理范围 0 到 1

    vertices = np.array([
        # 位置(x,y)   # 纹理(u,v)
        -1.0, 1.0, 0.0, 1.0,
        -1.0, -1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 0.0,

        -1.0, 1.0, 0.0, 1.0,
        1.0, -1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0
    ], dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # 位置
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
    # 纹理
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(8))

    glBindVertexArray(0)
    return vao

def generate_dataset(num_per_texture=5):
    base_dir = Path("../textures/KFS")
    background_folder = r"D:/Python_Projects/YOLO/KFS_locating/KFS_backgrounds"
    output_dir = Path("mono_cube_dataset_demo")
    train_img_save_dir = output_dir / "images/train"
    train_lbl_save_dir = output_dir / "labels/train"
    val_img_save_dir = output_dir / "images/val"
    val_lbl_save_dir = output_dir / "labels/val"

    if output_dir.exists():
        print(f"已清理旧数据: {output_dir}")
        shutil.rmtree(output_dir)

    for p in [train_img_save_dir, train_lbl_save_dir, val_img_save_dir, val_lbl_save_dir]:
        p.mkdir(parents=True, exist_ok=True)

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

    bg_manager = BackgroundManager(background_folder)
    bg_vao = setup_bg_quad()
    bg_shader = create_background_shader()
    bg_tex_loc = glGetUniformLocation(bg_shader, "bgTexture")

    cube = TexturedCube(img_paths)
    texture = pre_load_all_textures(img_paths)

    train_num = int(num_per_texture * 0.8)

    for class_id, img_p in enumerate(img_paths):
        tex = texture[class_id]
        cnt = 0

        while cnt < num_per_texture:

            for event in pygame.event.get(): # 防止卡死
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # 加载背景
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)
            bg_manager.load_background()
            glUseProgram(bg_shader)

            if bg_manager.background_id is not None:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, bg_manager.background_id)
                glUniform1i(bg_tex_loc, 0)

                glBindVertexArray(bg_vao)
                glDrawArrays(GL_TRIANGLES, 0, 6)
                glBindVertexArray(0)
                glBindTexture(GL_TEXTURE_2D, 0)

            # KFS和地面渲染逻辑
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # glClearColor(random.random() * 0.8, random.random() * 0.8, random.random() * 0.8, 1.0)  # 背景颜色
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, 1.0, 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # 随机相机位姿
            eye_x, eye_y, eye_z = spherical_to_cartesian(*random_spherical_coords())
            eye = eye_x, eye_y, eye_z

            # # 有遮挡
            # pos_x, pos_y, pos_z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
            # gluLookAt(eye_x, eye_y, eye_z,
            #           pos_x, pos_y, pos_z,
            #           0, 1, 0)

            # 无遮挡
            gluLookAt(eye_x, eye_y, eye_z,
                      0, 0, 0,
                      0, 1, 0)

            cube.setup_lighting(eye, is_random=True)

            draw_ground(cube.shader_program)

            cube.draw(tex)

            # 获取标签信息
            faces_num, face_coords = cube.get_seg_coordinates(eye)

            glReadBuffer(GL_BACK)
            data = glReadPixels(0, 0, 640, 640, GL_RGB, GL_UNSIGNED_BYTE)
            surface = pygame.image.fromstring(data, (640, 640), 'RGB')
            surface = pygame.transform.flip(surface, False, True)
            pygame.display.flip()

            if faces_num >= 2:

                # 写入和保存
                file_id = f"{uuid.uuid4().hex[:8]}"

                if cnt <= train_num:
                    img_save_dir = train_img_save_dir
                    lbl_save_dir = train_lbl_save_dir
                else:
                    img_save_dir = val_img_save_dir
                    lbl_save_dir = val_lbl_save_dir

                # 定义图片尺寸
                W, H = 640, 640

                img_filename = f"{class_id:02d}_{img_p.stem}_{file_id}.png"
                lbl_filename = f"{class_id:02d}_{img_p.stem}_{file_id}.txt"

                lbl_path = lbl_save_dir / lbl_filename

                # 写入标签文件
                with open(lbl_path, "a") as f:
                    for i in range(faces_num):

                        normalized_coords = []
                        for point in face_coords[i]:
                            nx = point[0] / W
                            ny = point[1] / H

                            normalized_coords.extend([nx, ny])

                        coords_str = " ".join(map(lambda x: f"{x:.6f}", normalized_coords))

                        label_line = f"{class_id} {coords_str}\n"

                        f.write(label_line)

                cnt += 1

                pygame.image.save(surface, str(img_save_dir / img_filename))

            time.sleep(0.01)

    print(f"合成完毕！数据保存在: {output_dir.absolute()}")

if __name__ == "__main__":
    generate_dataset(num_per_texture=5)