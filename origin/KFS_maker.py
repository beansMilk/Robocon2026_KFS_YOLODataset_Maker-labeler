import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from pathlib import Path
import random, time, uuid, os
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

        pygame.init()
        display = (640, 640)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("单帧立方体 - 按任意键退出")

        # 设置OpenGL视图
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

        self.load_texture(texture_path)

    def load_texture(self, image_path):
        """加载纹理"""
        try:
            textureSurface = pygame.image.load(image_path)
        except:
            print(f"无法加载图片: {image_path}")
            exit(0)

        textureData = pygame.image.tostring(textureSurface, "RGBA", True)
        width = textureSurface.get_width()
        height = textureSurface.get_height()

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    def draw(self):
        """绘制带纹理的立方体（带法线）"""
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)

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
    light_positions = [
        [3, 3, 3, 1.0],  # 光源0：右前上
        [-3, 3, 3, 1.0],  # 光源1：左前上
        [3, 3, -3, 1.0],  # 光源2：右前下
        [-3, 3, -3, 1.0],  # 光源3：左前下
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
        glLightf(light, GL_LINEAR_ATTENUATION, 0.02)
        glLightf(light, GL_QUADRATIC_ATTENUATION, 0.035)

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

def show_single_frame(cube):
    """只显示一帧的版本（用于测试）"""

    # 设置投影矩阵
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (640 / 640), 0.1, 50.0)

    # 设置模型视图矩阵
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 应用简单旋转，让立方体看起来更立体
    gluLookAt(system.random() * 12 + 3, system.random() * 12 + 3, system.random() * 6 - 2,
              system.random() * 4 - 2, system.random() * 4 - 2, system.random() * 4 - 2,
              0, 1, 0)

    # 设置光源
    setup_lighting()

    # 清屏
    glClearColor(random.random(), random.random(), random.random(), 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 绘制立方体
    cube.draw()
    draw_ground()

    # 刷新显示
    pygame.display.flip()
    glReadBuffer(GL_FRONT)

    # 读取像素数据
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, 640, 640, GL_RGB, GL_UNSIGNED_BYTE)

    # 转换并保存
    surface = pygame.image.fromstring(data, (640, 640), 'RGB')
    surface = pygame.transform.flip(surface, False, True)

    return surface

if __name__ == "__main__":
    texture_paths = list(Path("textures/KFS").rglob("*.png"))[1:]

    # 创建立方体
    for texture_path in texture_paths:
        if texture_path.stem[-2:] == "R1":
            output_path = Path("origin_data") / texture_path.stem / Path("images") / f"{uuid.uuid4()}.png"
        else:
            output_path = Path("origin_data") / texture_path.stem[:-2] / Path("images") / f"{uuid.uuid4()}.png"
        os.makedirs(output_path.parent, exist_ok=True)
        cube = TexturedCube(texture_path)

        for i in range(1):
            surface = show_single_frame(cube) # pygame.image.save(surface, "screenshot_guaranteed.png")
            time.sleep(0.7)
#             pygame.image.save(surface, output_path.parent /  f"{output_path.parent.parent.stem}_{uuid.uuid4()}.png")
#
#         classes_txt_path = output_path.parent.parent / Path("labels") / "classes.txt"
#         os.makedirs(classes_txt_path.parent, exist_ok=True)
#         if not classes_txt_path.exists():
#             with open(classes_txt_path, "w") as f:
#                 f.write("""R_R1
# B_R1
# T_03
# T_04
# T_05
# T_06
# T_07
# T_08
# T_09
# T_10
# T_11
# T_12
# T_13
# T_14
# T_15
# T_16
# T_17
# F_18
# F_19
# F_20
# F_21
# F_22
# F_23
# F_24
# F_25
# F_26
# F_27
# F_28
# F_29
# F_30
# F_31
# F_32""")