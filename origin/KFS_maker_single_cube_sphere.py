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
            [0, 3, 2, 1],  # 后面 (Back)
            [4, 0, 1, 5],  # 右面 (Right)
            [6, 4, 5, 7],  # 前面 (Front)
            [3, 6, 7, 2],  # 左面 (Left)
            [1, 5, 7, 2],  # 顶面 (Top)
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

        pos_shift_x = random.uniform(-3, 3)
        pos_shift_z = random.uniform(-3, 3)

        light_pool = [
            {"pos": [-5 + pos_shift_x, 7, -5 + pos_shift_z], "color": [0.9, 0.9, 0.9]},
            {"pos": [5 + pos_shift_x, 7, 5 + pos_shift_z], "color": [0.9, 0.9, 0.9]},
            {"pos": [-5 + pos_shift_x, 7, 5 + pos_shift_z], "color": [0.9, 0.9, 0.9]},
            {"pos": [5 + pos_shift_x, 7, 5 + pos_shift_z], "color": [0.9, 0.9, 0.9]}
        ]

        chosen = random.sample(light_pool, 2)

        pos_data = np.array([c["pos"] for c in chosen], dtype=np.float32).flatten()
        col_data = np.array([c["color"] for c in chosen], dtype=np.float32).flatten()

        glUniform3fv(light_pos_loc, 2, pos_data)
        glUniform3fv(light_color_loc, 2, col_data)
        glUniform1i(num_lights_loc, 2)

        if is_random:
            # 饱和度随机化
            rand_saturation = random.uniform(0.8, 1.2)
            glUniform1f(sat_loc, rand_saturation)

            # 高光强度随机化
            rand_specular = random.uniform(0.2, 0.9)
            glUniform1f(spec_loc, rand_specular)

            # 高光聚拢度随机化
            rand_shininess = random.uniform(50.0, 128.0)
            glUniform1f(shininess_loc, rand_shininess)
        else:
            # 非随机模式下保持原色
            glUniform1f(sat_loc, 1.0)

        # 视点位置
        glUniform3f(view_pos_loc, current_cam_pos[0], current_cam_pos[1], current_cam_pos[2])

from OpenGL.GL import *
from OpenGL.GL import shaders


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
        uniform float ambientStrength = 0.35; // 基础环境光强度

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

    # 3. 同步矩阵 (防止地面消失或错位)
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

def random_spherical_coords(r_range=(4, 6), theta_range=(30, 105), phi_range=(-180, 180)): # 输入为角度值
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

def generate_dataset(num_per_texture=5):
    base_dir = Path("../textures/KFS")
    output_dir = Path("mono_cube_dataset_sphere")
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
            for event in pygame.event.get(): # 防止卡死
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # KFS和地面渲染逻辑
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(random.random() * 0.8, random.random() * 0.8, random.random() * 0.8, 1.0)  # 背景颜色
            glEnable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, 1.0, 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # 随机相机位姿
            eye_x, eye_y, eye_z = spherical_to_cartesian(*random_spherical_coords())
            eye = eye_x, eye_y, eye_z

            # # 有遮挡
            # pos_x, pos_y, pos_z = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)
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
            # time.sleep(1)
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
            else:
                img_filename = f"{class_id}_{img_p.stem}_{file_id}.png"
                with open(lbl_save_dir / f"{class_id}_{img_p.stem}_{file_id}.txt", "w") as f:  # 写入标签文件
                    f.write(label_line)

            pygame.image.save(surface, str(img_save_dir / img_filename))

    print(f"合成完毕！数据保存在: {output_dir.absolute()}")

if __name__ == "__main__":
    generate_dataset(num_per_texture=5)